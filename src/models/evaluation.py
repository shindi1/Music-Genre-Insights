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
    recall_score,
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
        "per_class_recall": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def print_metrics(metrics: dict, labels: Optional[list[str]] = None) -> None:
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"macro F1    : {metrics['macro_f1']:.4f}")
    print(f"weighted F1 : {metrics['weighted_f1']:.4f}")
    if labels and "per_class_recall" in metrics:
        print("\ngenre prediction rate:")
        for genre, recall in sorted(
            zip(labels, metrics["per_class_recall"]), key=lambda x: -x[1]
        ):
            bar = "█" * int(recall * 20)
            print(f"  {genre:<12}: {recall:5.1%}  {bar}")
    print()
    print(metrics["report"])


def save_genre_recall_plot(
    labels: list[str],
    recall_scores: list[float],
    out_path: Path,
    title: str = "Genre Prediction Rate",
) -> None:
    """Horizontal bar chart of per-genre recall with a random-baseline reference line."""
    import matplotlib.pyplot as plt

    pairs = sorted(zip(labels, recall_scores), key=lambda x: x[1])
    genres, scores = zip(*pairs)
    baseline = 1 / len(labels)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(genres, scores, color="steelblue")
    ax.axvline(baseline, color="red", linestyle="--", linewidth=1, label=f"random ({baseline:.0%})")
    ax.bar_label(bars, fmt=lambda v: f"{v:.1%}", padding=4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall (prediction success rate)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_feature_importance_plot(
    clf,
    feature_names: list[str],
    labels: list[str],
    out_path: Path,
    title: str = "Feature Importance",
    top_n: int = 15,
) -> None:
    """Save a feature importance plot.

    - XGBoost: built-in feature_importances_ (gain), top_n features
    - Logistic: mean |coef| across classes for audio; top words per genre for lyrics
    - Other models: skipped silently
    """
    import matplotlib.pyplot as plt

    model_name = type(clf).__name__

    if model_name == "XGBClassifier":
        importances = clf.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        names = [feature_names[i] for i in indices]
        scores = importances[indices]

        fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
        ax.barh(names, scores, color="steelblue")
        ax.set_xlabel("Feature importance (gain)")
        ax.set_title(title)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    elif model_name == "LogisticRegression":
        coef = clf.coef_  # (n_classes, n_features)
        tfidf_mask = [f.startswith("tfidf::") for f in feature_names]
        is_lyrics = any(tfidf_mask)

        if is_lyrics:
            # Top words per genre by coefficient value
            tfidf_idx = [i for i, m in enumerate(tfidf_mask) if m]
            words = [feature_names[i].replace("tfidf::", "") for i in tfidf_idx]
            n_top = 10
            n_classes = len(labels)
            fig, axes = plt.subplots(1, n_classes, figsize=(3 * n_classes, 5), sharey=False)
            for i, (label, ax) in enumerate(zip(labels, axes)):
                class_coef = coef[i, tfidf_idx]
                top_idx = np.argsort(class_coef)[-n_top:]
                ax.barh([words[j] for j in top_idx], class_coef[top_idx], color="steelblue")
                ax.set_title(label, fontsize=10)
                ax.tick_params(labelsize=8)
            fig.suptitle(title)
            plt.tight_layout()
        else:
            # Audio: mean absolute coefficient across classes
            mean_abs = np.abs(coef).mean(axis=0)
            indices = np.argsort(mean_abs)
            names = [feature_names[i] for i in indices]
            scores = mean_abs[indices]

            fig, ax = plt.subplots(figsize=(7, max(4, len(names) * 0.4)))
            ax.barh(names, scores, color="steelblue")
            ax.set_xlabel("Mean |coefficient| across genres")
            ax.set_title(title)
            plt.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


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
