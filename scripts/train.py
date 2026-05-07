"""Train and evaluate a classifier.

Usage:
    # Audio mode (default — uses Spotify audio features)
    python scripts/train.py --model xgboost

    # Lyrics mode (uses TF-IDF on Genius lyrics)
    python scripts/train.py --model logistic --mode lyrics

    # Override feature blocks manually
    python scripts/train.py --model xgboost --no-audio --use-tfidf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.models.classifiers import MODEL_FACTORIES
from src.models.evaluation import (
    evaluate, print_metrics,
    save_confusion_matrix_plot, save_genre_recall_plot,
    save_feature_importance_plot,
)
from src.models.features import build_features

_DATA_ROOT = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_AUDIO_DIR  = _DATA_ROOT / "audio"
PROCESSED_LYRICS_DIR = _DATA_ROOT / "lyrics"
REPORTS_DIR   = Path(__file__).parent.parent / "reports"


@click.command(context_settings=dict(show_default=True))
@click.option("--model",   type=click.Choice(list(MODEL_FACTORIES)), default="xgboost")
@click.option("--mode",    type=click.Choice(["audio", "lyrics", "both"]), default="audio",
              help="Feature mode: audio features, TF-IDF lyrics, or both.")
@click.option("--data-dir", type=click.Path(path_type=Path), default=None,
              help="Data directory (default: data/processed/audio or data/processed/lyrics)")
@click.option("--max-features", type=int, default=10_000, help="TF-IDF vocab size (lyrics mode)")
@click.option("--seed", type=int, default=42)
def main(model, mode, data_dir, max_features, seed):
    if data_dir is None:
        data_dir = PROCESSED_AUDIO_DIR if mode == "audio" else PROCESSED_LYRICS_DIR

    train = pd.read_parquet(data_dir / "train.parquet")
    val   = pd.read_parquet(data_dir / "val.parquet")
    test  = pd.read_parquet(data_dir / "test.parquet")
    click.echo(f"loaded splits: train={len(train)} val={len(val)} test={len(test)}")

    use_audio         = mode in ("audio", "both")
    use_tfidf         = mode in ("lyrics", "both")
    use_lyric_numerics = mode in ("lyrics", "both")

    train_b, val_b, test_b = build_features(
        train, val, test,
        max_features=max_features,
        use_audio=use_audio,
        use_tfidf=use_tfidf,
        use_lyric_numerics=use_lyric_numerics,
    )
    click.echo(f"feature matrix: {train_b.X.shape}  mode={mode}")

    le = LabelEncoder().fit(train_b.y)
    y_train = le.transform(train_b.y)
    y_val   = le.transform(val_b.y)
    y_test  = le.transform(test_b.y)
    labels  = list(le.classes_)

    if model == "knn":
        clf = MODEL_FACTORIES[model](mode=mode)
    else:
        clf = MODEL_FACTORIES[model](seed=seed, mode=mode)
    click.echo(f"training {model}...")
    if model == "xgboost":
        clf.fit(train_b.X, y_train,
                sample_weight=compute_sample_weight("balanced", y_train))
    else:
        clf.fit(train_b.X, y_train)

    click.echo("\n=== validation ===")
    val_pred  = le.inverse_transform(clf.predict(val_b.X))
    val_metrics = evaluate(le.inverse_transform(y_val), val_pred, labels=labels)
    print_metrics(val_metrics, labels=labels)

    click.echo("\n=== test ===")
    test_pred = le.inverse_transform(clf.predict(test_b.X))
    test_metrics = evaluate(le.inverse_transform(y_test), test_pred, labels=labels)
    print_metrics(test_metrics, labels=labels)

    out_dir = REPORTS_DIR / mode / model
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps({
        "model": model,
        "mode": mode,
        "validation": {k: v for k, v in val_metrics.items() if k != "report"},
        "test":       {k: v for k, v in test_metrics.items() if k != "report"},
    }, indent=2))

    save_confusion_matrix_plot(
        le.inverse_transform(y_test), test_pred, labels,
        out_dir / "confusion_matrix_test.png",
        title=f"{model} ({mode}) — test confusion matrix",
    )
    save_genre_recall_plot(
        labels, test_metrics["per_class_recall"],
        out_dir / "genre_recall_test.png",
        title=f"{model} ({mode}) — genre prediction rate (test)",
    )
    save_feature_importance_plot(
        clf, train_b.feature_names, labels,
        out_dir / "feature_importance.png",
        title=f"{model} ({mode}) — feature importance",
    )
    click.echo(f"\nartifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
