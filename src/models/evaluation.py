"""Evaluation metrics and reports for genre classification."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list[str]] = None,
) -> dict:
    """Return a metrics dict suitable for printing or JSON dumping."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def print_metrics(metrics: dict) -> None:
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"macro F1    : {metrics['macro_f1']:.4f}")
    print(f"weighted F1 : {metrics['weighted_f1']:.4f}")
    print()
    print(metrics["report"])


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap. Requires matplotlib."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
