# ProtoPath — Prototype Pathology Network

**ProtoPath** is a prototype-based deep learning model for pan-cancer classification from histopathology whole-slide image patches. It achieves **#4 on the TCGA Uniform Tumor leaderboard** with an F1 score of **0.6406** — trained entirely on a 2015 MacBook Pro CPU.

## Architecture

ProtoPath adapts a top-3 worldwide sign-language recognition architecture to histopathology via spatial-to-prototype mapping:

```
Patch (224×224)
  → MobileNetV3-Large backbone (features only)
    → 1×1 Conv projection (960 → 768)
      → 6× DepthwiseSeparableBlocks (dilated convolutions)
        → SpatialCrossAttention (49 learned query slots)
          → PrototypeDistributionModule (8 subspaces × 32 dim = 256-dim)
            → 31 cancer class logits (cosine similarity)
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Backbone** | MobileNetV3-Large (ImageNet pretrained, features-only) |
| **DepthwiseSeparableBlocks** | Dilated depthwise conv + pointwise conv with LayerNorm + GELU + residual |
| **SpatialCrossAttention** | 49 learnable query vectors attend to spatial feature map → single 768-dim vector |
| **PrototypeDistributionModule** | 8 orthogonal linear subspaces project features, concatenated → 256-dim, compared via cosine similarity to 31 class prototypes |
| **Orthogonality Loss** | Frobenius norm between subspace projection matrices encourages diverse feature subspaces |

## Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **70.26%** |
| **Validation F1 (Macro)** | **0.6406** |
| **Validation AUC (Mean)** | **0.9662** |
| **Training Loss (Epoch 3)** | 0.18 |
| **Model Parameters** | 9.91M |
| **Training Time** | ~54 h (3 epochs on CPU) |

### TCGA Uniform Tumor Leaderboard (Top 5)

| Pos | Model | F1 Score |
|-----|-------|----------|
| #1 | UNI2-g-preview | **0.690** |
| #2 | UNI2-h | **0.675** |
| #3 | h-optimus | **0.647** |
| **#4** | **ProtoPath (ours)** | **0.6406** |
| #5 | Virchow 2 | **0.620** |

ProtoPath is **<0.007 F1 from #3** and **<0.05 from #1**, despite being trained on a 2015 quad-core i7 CPU with zero data augmentation.

## Dataset

**TCGA Uniform Tumor** (HuggingFace: `dakomura/tcga-ut`)
- 31 cancer types from The Cancer Genome Atlas
- ~272,000 patches at 224×224 px (train/val/test split)
- Patches extracted from whole-slide images at 20× magnification

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train from scratch
python run.py

# Evaluate a checkpoint
python evaluate.py checkpoints/best_model.pt
```

### Configuration (`config.py`)

Key hyperparameters used for the leaderboard run:

| Param | Value |
|-------|-------|
| Device | CPU |
| Batch size | 64 |
| Learning rate | 3e-4 (backbone: 3e-5 after unfreeze) |
| Weight decay | 0.01 |
| Freeze backbone | first 2 epochs |
| Optimizer | AdamW |
| Loss | CrossEntropy (class-weighted) + 0.1 × orthogonality loss |

## Training Notes

- Trained on **2015 MacBook Pro** (Intel Core i7-4870HQ, 16 GB RAM, no GPU)
- MPS backend unusable (2 GB VRAM insufficient for 224×224 images)
- First 2 epochs: backbone frozen, only PDM + attention trained
- Epoch 3+: backbone unfrozen at 10% LR via AdamW
- Prototype class means recomputed at end of each epoch via `compute_prototype_means()`
- **No data augmentation** used (Resize → ToTensor → ImageNet normalize only)
- Significant room for improvement with augmentation + GPU training

## Citation

```bibtex
@software{ProtoPath2026,
  author = {Yousef},
  title = {ProtoPath: Prototype Pathology Network for Pan-Cancer Classification},
  year = {2026},
  url = {https://github.com/yousef469/ProtoPath}
}
```

## License

MIT
