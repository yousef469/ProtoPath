"""
Adapted from top-3-worldwide sign language architecture.
Mapping: joints×time (42×50) → spatial positions (7×7=49) via CNN.

Two configs:
  Lite (~1.8M)  — trains on MacBook in ~32h
  Pro  (~10.8M) — matches SL model size, needs GPU (Colab)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


NUM_CANCERS = 31


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, dim, dilation=1, dropout=0.2):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=dilation,
                                   dilation=dilation, groups=dim, bias=False)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.norm(x).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class SpatialCrossAttention(nn.Module):
    def __init__(self, num_queries=49, dim=768, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        queries = self.queries.expand(B, -1, -1)
        out, _ = self.attn(queries, x_flat, x_flat)
        return self.norm(out).mean(dim=1)


class PrototypeDistributionModule(nn.Module):
    """EXACT replica from sign language model — multi-subspace + cosine prototypes."""
    def __init__(self, input_dim=768, num_subspaces=4, subspace_dim=32):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        proto_dim = num_subspaces * subspace_dim

        self.subspace_proj = nn.ModuleList([
            nn.Linear(input_dim, subspace_dim, bias=False)
            for _ in range(num_subspaces)
        ])
        self.prototypes = nn.Parameter(torch.randn(NUM_CANCERS, proto_dim))
        nn.init.normal_(self.prototypes, std=0.01)
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward(self, x):
        subspaces = [proj(x) for proj in self.subspace_proj]
        d = F.normalize(torch.cat(subspaces, dim=-1), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = torch.mm(d, p.t())
        temp = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        return logits / temp, d

    def orthogonality_loss(self):
        loss = 0.0
        count = 0
        for i in range(self.num_subspaces):
            for j in range(i + 1, self.num_subspaces):
                Wi = F.normalize(self.subspace_proj[i].weight, dim=-1)
                Wj = F.normalize(self.subspace_proj[j].weight, dim=-1)
                loss += (Wi @ Wj.t()).norm(p='fro') ** 2
                count += 1
        return loss / count if count > 0 else 0.0


class CancerV2Model(nn.Module):
    def __init__(self, model_size="pro"):
        super().__init__()
        if model_size == "lite":
            self._build_lite()
        else:
            self._build_pro()
        self.model_size = model_size

    def _build_lite(self):
        embed_dim = 512
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", pretrained=True, features_only=True,
        )
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(576, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.conv_blocks = nn.ModuleList([
            DepthwiseSeparableBlock(embed_dim, dilation=2**i)
            for i in range(2)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pdm = PrototypeDistributionModule(embed_dim, num_subspaces=4, subspace_dim=32)

    def _build_pro(self):
        embed_dim = 768
        self.backbone = timm.create_model(
            "mobilenetv3_large_100", pretrained=True, features_only=True,
        )
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(960, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.conv_blocks = nn.ModuleList([
            DepthwiseSeparableBlock(embed_dim, dilation=2**i)
            for i in range(6)
        ])
        self.cross_attn = SpatialCrossAttention(num_queries=49, dim=embed_dim)
        self.pdm = PrototypeDistributionModule(embed_dim, num_subspaces=8, subspace_dim=32)

    def forward(self, x):
        feats = self.backbone(x)[-1]
        spatial = self.spatial_proj(feats)
        for block in self.conv_blocks:
            spatial = block(spatial)
        if self.model_size == "pro":
            pooled = self.cross_attn(spatial)
        else:
            pooled = self.pool(spatial).flatten(1)
        logits, _ = self.pdm(pooled)
        return logits

    def extract_embedding(self, x):
        feats = self.backbone(x)[-1]
        spatial = self.spatial_proj(feats)
        for block in self.conv_blocks:
            spatial = block(spatial)
        if self.model_size == "pro":
            pooled = self.cross_attn(spatial)
        else:
            pooled = self.pool(spatial).flatten(1)
        subspaces = [proj(pooled) for proj in self.pdm.subspace_proj]
        return F.normalize(torch.cat(subspaces, dim=-1), dim=-1)

    def get_aux_losses(self, x):
        feats = self.backbone(x)[-1]
        spatial = self.spatial_proj(feats)
        for block in self.conv_blocks:
            spatial = block(spatial)
        if self.model_size == "pro":
            pooled = self.cross_attn(spatial)
        else:
            pooled = self.pool(spatial).flatten(1)
        logits, _ = self.pdm(pooled)
        return logits, self.pdm.orthogonality_loss()

    def compute_prototype_means(self, dataloader, device):
        self.eval()
        buffers = {i: [] for i in range(NUM_CANCERS)}
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                emb = self.extract_embedding(images)
                for e, l in zip(emb, labels):
                    buffers[l.item()].append(e.cpu())
        for c in range(NUM_CANCERS):
            if len(buffers[c]) == 0:
                continue
            mean = torch.stack(buffers[c]).mean(dim=0)
            mean = F.normalize(mean, dim=0)
            noise = torch.randn_like(self.pdm.prototypes[c]) * 0.01
            self.pdm.prototypes.data[c] = F.normalize(mean + noise, dim=0)
        self.train()
