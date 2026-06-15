import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from config import DEVICE, CANCER_TYPES
from model import CancerDetectionModel
from dataset import TCGADataset


@torch.no_grad()
def evaluate_model(model_path, device=DEVICE):
    model = CancerDetectionModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    dataset = TCGADataset(split="test")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=4
    )

    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=CANCER_TYPES,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for i, cancer in enumerate(CANCER_TYPES):
        try:
            auc = roc_auc_score(
                (np.array(all_labels) == i).astype(int),
                all_probs[:, i],
            )
        except ValueError:
            auc = 0.0
        rows.append({
            "cancer": cancer,
            "precision": report[cancer]["precision"],
            "recall": report[cancer]["recall"],
            "f1": report[cancer]["f1-score"],
            "support": report[cancer]["support"],
            "auc": auc,
        })

    df = pd.DataFrame(rows)
    df.loc["MEAN"] = df.select_dtypes(include=[np.number]).mean()
    df.loc["MEAN", "cancer"] = "MEAN"
    df.to_csv("evaluation_results.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    evaluate_model(path)
