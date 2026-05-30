import torch

from config import NUM_CANCERS


def compute_class_weights(dataset, num_classes=NUM_CANCERS):
    from config import CANCER_TO_IDX
    from collections import Counter
    label_names = [s["label"] for s in dataset.dataset["json"]]
    counts = Counter(label_names)
    label_counts = torch.zeros(num_classes)
    for name, count in counts.items():
        label_counts[CANCER_TO_IDX[name]] = count
    total = label_counts.sum()
    weights = total / (label_counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return weights
