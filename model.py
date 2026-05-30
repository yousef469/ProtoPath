import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import BACKBONE_NAME, EMBED_DIM, NUM_CANCERS, PROTOTYPES_PER_TYPE, TEMPERATURE


class PrototypeBank(nn.Module):
    def __init__(self, num_cancers=NUM_CANCERS, num_prototypes=PROTOTYPES_PER_TYPE, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_cancers = num_cancers
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim

        self.prototypes = nn.Parameter(
            torch.randn(num_cancers, num_prototypes, embed_dim)
        )
        self._init_weights()

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(TEMPERATURE)))

    def _init_weights(self):
        for c in range(self.num_cancers):
            nn.init.normal_(self.prototypes[c], mean=0.0, std=0.01)
            self.prototypes.data[c] = F.normalize(self.prototypes.data[c], dim=-1)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bd,cpd->bcp", x, prototypes)
        logits, _ = sim.max(dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        return logits / temperature


class CancerDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
        )

        in_features = self.backbone.num_features
        self.proj = nn.Linear(in_features, EMBED_DIM)
        self.norm = nn.LayerNorm(EMBED_DIM)

        self.prototype_bank = PrototypeBank()

    def forward(self, x):
        features = self.backbone(x)
        features = self.proj(features)
        features = self.norm(features)
        logits = self.prototype_bank(features)
        return logits

    def extract_features(self, x):
        features = self.backbone(x)
        features = self.proj(features)
        features = self.norm(features)
        return F.normalize(features, dim=-1)

    def compute_prototype_means(self, dataloader, device):
        self.eval()
        feature_buffers = {i: [] for i in range(self.prototype_bank.num_cancers)}
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                features = self.extract_features(images)
                for feat, label in zip(features, labels):
                    feature_buffers[label.item()].append(feat.cpu())
        for c in range(self.prototype_bank.num_cancers):
            if len(feature_buffers[c]) == 0:
                continue
            class_mean = torch.stack(feature_buffers[c]).mean(dim=0)
            class_mean = F.normalize(class_mean, dim=0)
            noise = torch.randn_like(self.prototype_bank.prototypes[c]) * 0.01
            self.prototype_bank.prototypes.data[c] = (
                class_mean.unsqueeze(0) + noise
            )
            self.prototype_bank.prototypes.data[c] = F.normalize(
                self.prototype_bank.prototypes.data[c], dim=-1
            )
        self.train()
