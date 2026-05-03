"""Train and evaluate a single classifier on the processed splits.

Usage:
    python scripts/train.py --model logistic
    python scripts/train.py --model knn
    python scripts/train.py --model xgboost
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models.classifiers import MODEL_FACTORIES
from src.models.evaluation import evaluate, print_metrics, save_confusion_matrix_plot
from src.models.features import build_features

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


@click.command()
@click.option("--model", type=click.Choice(list(MODEL_FACTORIES)), default="logistic")
@click.option("--data-dir", type=click.Path(path_type=Path), default=PROCESSED_DIR)
@click.option("--no-audio", is_flag=True, help="Skip Spotify audio features")
@click.option("--no-lyric-numerics", is_flag=True, help="Skip lyric stats features")
@click.option("--max-features", type=int, default=10_000, help="TF-IDF vocab size")
@click.option("--seed", type=int, default=42)
def main(model, data_dir, no_audio, no_lyric_numerics, max_features, seed):
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    click.echo(f"loaded splits: train={len(train)} val={len(val)} test={len(test)}")

    train_b, val_b, test_b = build_features(
        train, val, test,
        max_features=max_features,
        use_audio=not no_audio,
        use_lyric_numerics=not no_lyric_numerics,
    )
    click.echo(f"feature matrix: {train_b.X.shape}")

    le = LabelEncoder().fit(train_b.y)
    y_train = le.transform(train_b.y)
    y_val = le.transform(val_b.y)
    y_test = le.transform(test_b.y)
    labels = list(le.classes_)

    clf = MODEL_FACTORIES[model](seed=seed) if model != "knn" else MODEL_FACTORIES[model]()
    click.echo(f"training {model}...")
    clf.fit(train_b.X, y_train)

    click.echo("\n=== validation ===")
    val_pred = le.inverse_transform(clf.predict(val_b.X))
    val_metrics = evaluate(le.inverse_transform(y_val), val_pred, labels=labels)
    print_metrics(val_metrics)

    click.echo("\n=== test ===")
    test_pred = le.inverse_transform(clf.predict(test_b.X))
    test_metrics = evaluate(le.inverse_transform(y_test), test_pred, labels=labels)
    print_metrics(test_metrics)

    out_dir = REPORTS_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "model": model,
        "validation": {k: v for k, v in val_metrics.items() if k != "report"},
        "test": {k: v for k, v in test_metrics.items() if k != "report"},
    }, indent=2))
    save_confusion_matrix_plot(
        le.inverse_transform(y_test), test_pred, labels,
        out_dir / "confusion_matrix_test.png",
        title=f"{model} — test confusion matrix",
    )
    click.echo(f"\nartifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
