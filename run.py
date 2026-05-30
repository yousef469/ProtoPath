import torch
import numpy as np
import random
import os

from config import BATCH_SIZE, NUM_WORKERS, SEED, DEVICE, EPOCHS, LR, WEIGHT_DECAY, FREEZE_BACKBONE_EPOCHS
from v2lightning_model import CancerV2Model
from dataset import TCGADataModule
from train import Trainer


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    set_seed()

    print("\nLoading TCGA Uniform Tumor dataset (31 cancer types, 1.6M patches)...")
    data_module = TCGADataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train samples: {len(data_module.train_dataset):,}")
    print(f"Val samples:   {len(data_module.val_dataset):,}")
    print(f"Test samples:  {len(data_module.test_dataset):,}")

    model = CancerV2Model(model_size="pro")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: adapted from sign language architecture")
    print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")

    trainer = Trainer(model, data_module, device=DEVICE)

    checkpoint_path = os.path.join("checkpoints", "last_model.pt")
    start_epoch = trainer.resume_from_checkpoint(checkpoint_path)

    print(f"\n{'='*50}")
    print(f"Training Config")
    print(f"{'='*50}")
    print(f"  Epochs:              {EPOCHS}")
    print(f"  Batch size:          {BATCH_SIZE}")
    print(f"  Learning rate:       {LR}")
    print(f"  Weight decay:        {WEIGHT_DECAY}")
    print(f"  Freeze backbone:     first {FREEZE_BACKBONE_EPOCHS} epochs")
    num_sub = model.pdm.num_subspaces
    sub_dim = model.pdm.subspace_dim
    print(f"  Prototype subspaces: {num_sub} × {sub_dim} = {num_sub*sub_dim}-dim")
    print(f"  Orthogonality loss:  yes")
    print(f"  Starting epoch:      {start_epoch + 1}")
    print(f"{'='*50}")

    test_metrics = trainer.fit(start_epoch=start_epoch)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Test accuracy:           {test_metrics['acc']:.4f}")
    print(f"Test F1-macro:          {test_metrics['f1_macro']:.4f}")
    print(f"Checkpoint: checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
