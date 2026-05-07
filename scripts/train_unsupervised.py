"""Train and evaluate a single unsupervised model on processed Spotify/Genius data.

Usage:
    # Audio mode (default — uses Spotify z-scaled audio features)
    python scripts/train_unsupervised.py --model kmeans
    python scripts/train_unsupervised.py --model pca --n-components 10
    python scripts/train_unsupervised.py --model gmm

    # Lyrics mode (uses lyric numeric stats)
    python scripts/train_unsupervised.py --model kmeans --mode lyrics

    # Both feature sets combined
    python scripts/train_unsupervised.py --model kmeans --mode both

    python scripts/train_unsupervised.py --model kmeans --elbow
    python scripts/train_unsupervised.py --model kmeans --n-clusters 8
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.unsupervised import UNSUPERVISED_FACTORIES, UNSUPERVISED_MODEL_CONFIGS
from src.models.unsupervised_evaluation import (
    evaluate_clustering,
    evaluate_gmm_fit,
    evaluate_pca,
    print_clustering_metrics,
    print_pca_metrics,
    save_bic_aic_plot,
    save_cluster_composition_plot,
    save_cluster_scatter_plot,
    save_elbow_plot,
    save_explained_variance_plot,
    save_genre_cluster_heatmap,
    save_pca_loadings_plot,
    save_pca_scatter_plot,
)
from config import TARGET_GENRES

_DATA_ROOT = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_AUDIO_DIR  = _DATA_ROOT / "audio"
PROCESSED_LYRICS_DIR = _DATA_ROOT / "lyrics"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

_AUDIO_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
_LYR_COLS = [
    "lyr_word_count", "lyr_unique_word_count", "lyr_vocab_diversity",
    "lyr_line_count", "lyr_avg_word_length", "lyr_avg_line_length",
    "lyr_exclaim_density", "lyr_question_density",
    "lyr_repetition_ratio", "lyr_uppercase_ratio",
]


def _build_feature_matrix(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names) for the requested feature set.

    Both audio and lyric features are raw in the parquets and are
    z-scaled here so all features are on the same scale.
    """
    use_audio  = mode in ("audio", "both")
    use_lyrics = mode in ("lyrics", "both")

    audio_cols = [c for c in _AUDIO_COLS if c in df.columns] if use_audio else []
    lyr_cols   = [c for c in _LYR_COLS if c in df.columns] if use_lyrics else []

    blocks: list[np.ndarray] = []
    if audio_cols:
        X_audio = df[audio_cols].fillna(0).to_numpy(dtype=np.float64)
        blocks.append(StandardScaler().fit_transform(X_audio))
    if lyr_cols:
        X_lyr = df[lyr_cols].fillna(0).to_numpy(dtype=np.float64)
        blocks.append(StandardScaler().fit_transform(X_lyr))

    if not blocks:
        raise ValueError(f"No features available for mode='{mode}'. Check that the parquet contains the expected columns.")

    X = np.hstack(blocks)
    return X, audio_cols + lyr_cols


@click.command(context_settings=dict(show_default=True))
@click.option("--model", type=click.Choice(list(UNSUPERVISED_FACTORIES)), default="kmeans")
@click.option("--mode", type=click.Choice(["audio", "lyrics", "both"]), default="audio",
              help="Feature mode: z-scaled audio features, lyric numeric stats, or both.")
@click.option("--data-dir", type=click.Path(path_type=Path), default=None,
              help="Data directory (default: data/processed/audio or data/processed/lyrics)")
@click.option("--data-split", type=click.Choice(["full", "train", "test"]), default="full",
              help="Parquet to load. 'full' is recommended for unsupervised (no leakage concern).")
@click.option("--n-clusters", type=int, default=6,
              help="Number of clusters for kmeans/gmm. Default=6 matches the genre taxonomy.")
@click.option("--n-components", type=int, default=None,
              help="Number of PCA components. Default=2 for a scatter plot, use 10+ for scree analysis.")
@click.option("--elbow", is_flag=True, help="Run elbow analysis over k=2..12 before fitting (kmeans only).")
@click.option("--bic-sweep", "bic_sweep", is_flag=True,
              help="Sweep n_components=2..12 across all covariance types and plot BIC/AIC (gmm only).")
@click.option("--covariance-type", "covariance_type",
              type=click.Choice(["full", "tied", "diag", "spherical"]), default="full",
              help="GMM covariance structure. full=most flexible, spherical=soft k-means.")
@click.option("--seed", type=int, default=42)
def main(model, mode, data_dir, data_split, n_clusters, n_components, elbow, bic_sweep, covariance_type, seed):
    if data_dir is None:
        data_dir = PROCESSED_AUDIO_DIR if mode == "audio" else PROCESSED_LYRICS_DIR

    parquet = "full.parquet" if data_split == "full" else f"{data_split}.parquet"
    df = pd.read_parquet(data_dir / parquet)
    click.echo(f"loaded {parquet}: {len(df)} rows")

    X, feature_names = _build_feature_matrix(df, mode)
    y_true = df["genre"].to_numpy(dtype=str)
    genre_labels = list(TARGET_GENRES)
    click.echo(f"feature matrix: {X.shape}  ({len(feature_names)} features)  mode={mode}")

    out_dir = REPORTS_DIR / mode / model
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = UNSUPERVISED_MODEL_CONFIGS[model]
    if cfg.task == "reduction":
        _run_pca(model, X, y_true, feature_names, n_components or 2,
                 mode, data_split, len(df), out_dir, seed)
    else:
        if elbow and model == "kmeans":
            click.echo("running elbow analysis k=2..12 ...")
            save_elbow_plot(X, range(2, 13), out_dir / "elbow.png", seed=seed)
            click.echo(f"saved {out_dir}/elbow.png — pick a k, then rerun with --n-clusters <k>")
            return
        if bic_sweep and model == "gmm":
            click.echo("running BIC/AIC sweep (n_components=2..12, all covariance types) ...")
            save_bic_aic_plot(X, range(2, 13), out_dir / "bic_aic_sweep.png", seed=seed)
            click.echo(f"saved {out_dir}/bic_aic_sweep.png — pick the k and covariance_type at the BIC elbow")
            return
        _run_clustering(model, X, y_true, genre_labels, n_clusters,
                        mode, data_split, len(df), out_dir, seed,
                        covariance_type=covariance_type)


def _run_pca(model, X, y_true, feature_names, n_components, mode, data_split, n_samples, out_dir, seed):
    from src.models.unsupervised import make_pca

    pca = make_pca(n_components=n_components, seed=seed)
    click.echo(f"fitting PCA (n_components={n_components}) ...")
    X_reduced = pca.fit_transform(X)

    metrics = evaluate_pca(pca, X, X_reduced)
    metrics.update({
        "model":        model,
        "mode":         mode,
        "data_split":   data_split,
        "n_samples":    n_samples,
    })

    click.echo("\n=== PCA metrics ===")
    print_pca_metrics(metrics)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    save_pca_scatter_plot(
        X_reduced[:, :2], y_true,
        metrics["explained_variance_ratio"][:2],
        out_dir / "pca_scatter.png",
        title="PCA — PC1 vs PC2 by genre",
    )
    save_explained_variance_plot(
        metrics["explained_variance_ratio"],
        out_dir / "explained_variance.png",
        title="PCA — Explained Variance per Component",
    )
    save_pca_loadings_plot(
        pca.components_,
        feature_names,
        out_dir / "pca_loadings.png",
        title="PCA — Feature Loadings (PC1 & PC2)",
        max_components=2,
    )
    click.echo(f"\nartifacts saved to {out_dir}/")


def _run_clustering(model, X, y_true, genre_labels, n_clusters, mode, data_split, n_samples, out_dir, seed,
                    covariance_type="full"):
    if model == "kmeans":
        clf = UNSUPERVISED_FACTORIES["kmeans"](n_clusters=n_clusters, seed=seed)
    else:  # gmm
        clf = UNSUPERVISED_FACTORIES["gmm"](
            n_components=n_clusters, covariance_type=covariance_type, seed=seed
        )

    click.echo(f"fitting {model} (n_clusters={n_clusters}"
               + (f", covariance_type={covariance_type}" if model == "gmm" else "") + ") ...")
    clf.fit(X)
    cluster_labels = np.array(clf.predict(X))

    metrics = evaluate_clustering(X, cluster_labels, y_true, seed=seed)
    if model == "gmm":
        metrics.update(evaluate_gmm_fit(clf, X))
    metrics.update({
        "model":      model,
        "mode":       mode,
        "n_clusters": int(n_clusters),
        "data_split": data_split,
        "n_samples":  int(n_samples),
    })

    click.echo(f"\n=== {model} metrics ===")
    print_clustering_metrics(metrics)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Project to 2D with PCA for visualization (independent of the clustering itself)
    pca_2d = PCA(n_components=2, random_state=seed)
    X_2d = pca_2d.fit_transform(X)
    explained_2d = pca_2d.explained_variance_ratio_.tolist()
    click.echo(f"  2D PCA for scatter captures {sum(explained_2d):.1%} of variance")

    save_cluster_scatter_plot(
        X_2d, cluster_labels, y_true,
        out_dir / "cluster_scatter.png",
        title=f"{model} cluster assignments",
        explained=explained_2d,
    )
    save_genre_cluster_heatmap(
        cluster_labels, y_true, genre_labels,
        out_dir / "genre_cluster_heatmap.png",
        title=f"{model} — Genre × Cluster Distribution",
    )
    save_cluster_composition_plot(
        cluster_labels, y_true, genre_labels,
        out_dir / "cluster_composition.png",
        title=f"{model} — Cluster Genre Composition",
    )
    click.echo(f"\nartifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
