import os

import time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from config import (
    CANCER_TYPES, NUM_CANCERS, DEVICE, EPOCHS, LR,
    WEIGHT_DECAY, BATCH_SIZE, FREEZE_BACKBONE_EPOCHS,
)
from v2lightning_model import CancerV2Model as CancerDetectionModel
from dataset import TCGADataModule
from loss import compute_class_weights


class Trainer:
    def __init__(self, model, data_module, device=DEVICE):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.best_val_acc = 0.0
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()
        self.test_loader = data_module.test_dataloader()

        self.class_weights = compute_class_weights(
            data_module.train_dataset
        ).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            ce_loss = self.criterion(logits, labels)
            orth_loss = self.model.pdm.orthogonality_loss()
            loss = ce_loss + 0.1 * orth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, loader, name="Val"):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(loader, desc=f"[{name}]"):
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        per_class_auc = {}
        for i, cancer in enumerate(CANCER_TYPES):
            try:
                per_class_auc[cancer] = roc_auc_score(
                    (np.array(all_labels) == i).astype(int),
                    all_probs[:, i],
                )
            except ValueError:
                per_class_auc[cancer] = 0.0

        return {
            "loss": avg_loss,
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class_auc": per_class_auc,
        }

    def print_metrics(self, metrics, phase="Val"):
        print(f"\n{phase} — Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.4f} | "
              f"F1-macro: {metrics['f1_macro']:.4f} | F1-weighted: {metrics['f1_weighted']:.4f}")
        mean_auc = np.mean(list(metrics['per_class_auc'].values()))
        print(f"Mean AUC: {mean_auc:.4f}")
        for cancer in CANCER_TYPES:
            auc = metrics['per_class_auc'][cancer]
            if auc < 0.7:
                print(f"  ⚠ {cancer}: AUC={auc:.4f}")

    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def resume_from_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
            else:
                return 0
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.best_val_acc = ckpt['val_acc']
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']+1}/{EPOCHS} (val_acc={ckpt['val_acc']:.4f})")
        return start_epoch

    def _save_checkpoint(self, epoch, val_acc, val_f1, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
        }, path)

    def fit(self, start_epoch=0):
        if start_epoch == 0:
            self.freeze_backbone()
        elif start_epoch >= FREEZE_BACKBONE_EPOCHS:
            self.unfreeze_backbone()
            for g in self.optimizer.param_groups:
                g['lr'] = LR * 0.1
            print(f"\n=== Backbone unfrozen on resume (epoch {start_epoch}) ===")
        else:
            self.freeze_backbone()

        for epoch in range(start_epoch, EPOCHS):
            if epoch == FREEZE_BACKBONE_EPOCHS:
                print("\n=== Unfreezing backbone ===")
                self.unfreeze_backbone()
                for g in self.optimizer.param_groups:
                    g['lr'] = LR * 0.1

            train_loss, train_acc = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "Val")

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            self.print_metrics(val_metrics, "Val")

            ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self._save_checkpoint(epoch, val_metrics['acc'], val_metrics['f1_macro'], ckpt_path)
                print(f"✓ New best model saved (acc={val_metrics['acc']:.4f})")

            self._save_checkpoint(epoch, val_metrics['acc'], val_metrics['f1_macro'],
                                  os.path.join(self.checkpoint_dir, "last_model.pt"))

        print("\n=== Final Test Evaluation ===")
        ckpt = torch.load(
            os.path.join(self.checkpoint_dir, "best_model.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        test_metrics = self.evaluate(self.test_loader, "Test")
        self.print_metrics(test_metrics, "Test")
        return test_metrics
