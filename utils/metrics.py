import os
import numpy as np


def compute_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Compute OA, per-class P/R/F1/IoU."""
    P = np.zeros(num_classes)
    R = np.zeros(num_classes)
    F1 = np.zeros(num_classes)
    IoU = np.zeros(num_classes)

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        P[c] = p
        R[c] = r
        F1[c] = 2 * p * r / (p + r + 1e-9)
        IoU[c] = tp / (tp + fp + fn + 1e-9)

    return {
        "OA": float(np.mean(preds == labels)),
        "mP": float(P.mean()),
        "mR": float(R.mean()),
        "mF1": float(F1.mean()),
        "mIoU": float(IoU.mean()),
        "per_class_P": P,
        "per_class_R": R,
        "per_class_F1": F1,
        "per_class_IoU": IoU,
    }


def print_results(metrics: dict, class_names: list = None, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"\n{'='*55}")
    print(f"{tag}RESULTS")
    print(f"{'='*55}")
    print(f"  OA={100*metrics['OA']:.2f}%  mP={100*metrics['mP']:.2f}%  mR={100*metrics['mR']:.2f}%  mF1={100*metrics['mF1']:.2f}%  mIoU={100*metrics['mIoU']:.2f}%")
    print(f"{'='*55}")


def save_results(metrics: dict, class_names: list, path: str, header: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(f"{header}\n")
        f.write(f"OA: {100*metrics['OA']:.2f}%\n")
        f.write(f"mP: {100*metrics['mP']:.2f}%\n")
        f.write(f"mR: {100*metrics['mR']:.2f}%\n")
        f.write(f"mF1: {100*metrics['mF1']:.2f}%\n")
        f.write(f"mIoU: {100*metrics['mIoU']:.2f}%\n")
        f.write("-" * 45 + "\n")
        for i, name in enumerate(class_names):
            f.write(
                f"  {name:<20} | F1={100*metrics['per_class_F1'][i]:5.2f}%  "
                f"IoU={100*metrics['per_class_IoU'][i]:5.2f}%\n"
            )
