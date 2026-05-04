"""Metrics and plots for unsupervised genre analysis.

Mirrors evaluation.py but with unsupervised-appropriate metrics:
  - Clustering: silhouette, Davies-Bouldin, ARI, purity
  - Reduction (PCA): explained variance, reconstruction MSE, loadings
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

# Fixed palette keyed to the 6 canonical genres for cross-plot consistency
_GENRE_PALETTE: dict[str, str] = {
    "pop":        "#4C72B0",
    "hip-hop":    "#DD8452",
    "rock":       "#55A868",
    "r&b":        "#C44E52",
    "country":    "#8172B2",
    "electronic": "#937860",
}


def _genre_color(genre: str) -> str:
    return _GENRE_PALETTE.get(genre, "#999999")


# ── Clustering metrics ────────────────────────────────────────────────────────

def cluster_purity(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """(1/N) * sum_k max_j |cluster_k ∩ class_j|."""
    correct = 0
    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        if mask.sum():
            correct += Counter(y_true[mask]).most_common(1)[0][1]
    return correct / len(y_true)


def evaluate_clustering(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    sample_silhouette: int = 3000,
    seed: int = 42,
) -> dict:
    """Return a metrics dict suitable for printing or JSON dumping.

    y_true is used only post-hoc (ARI, purity, dominant-genre mapping);
    it is never used to guide the clustering itself.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > sample_silhouette:
        idx = rng.choice(n, size=sample_silhouette, replace=False)
        sil = float(silhouette_score(X[idx], cluster_labels[idx]))
    else:
        sil = float(silhouette_score(X, cluster_labels))

    cluster_genre_map: dict[str, str] = {}
    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        cluster_genre_map[str(int(cid))] = Counter(y_true[mask]).most_common(1)[0][0]

    return {
        "silhouette_score":    sil,
        "davies_bouldin_index": float(davies_bouldin_score(X, cluster_labels)),
        "adjusted_rand_index":  float(adjusted_rand_score(y_true, cluster_labels)),
        "purity":               float(cluster_purity(y_true, cluster_labels)),
        "n_clusters":           int(len(np.unique(cluster_labels))),
        "cluster_genre_map":    cluster_genre_map,
        "cluster_sizes":        {str(int(k)): int(v)
                                 for k, v in Counter(cluster_labels.tolist()).items()},
    }


def evaluate_pca(pca, X_original: np.ndarray, X_reduced: np.ndarray) -> dict:
    """PCA-specific metrics — no clustering metrics apply."""
    evr = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components_for_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    X_reconstructed = pca.inverse_transform(X_reduced)
    reconstruction_mse = float(np.mean((X_original - X_reconstructed) ** 2))
    return {
        "explained_variance_ratio":      evr,
        "cumulative_explained_variance": cumulative.tolist(),
        "components_for_95pct_variance": components_for_95,
        "reconstruction_mse":            reconstruction_mse,
        "n_components_used":             len(evr),
        "n_features_original":           X_original.shape[1],
    }


def print_clustering_metrics(metrics: dict) -> None:
    print(f"silhouette       : {metrics['silhouette_score']:.4f}  (higher better, max 1.0)")
    print(f"davies-bouldin   : {metrics['davies_bouldin_index']:.4f}  (lower better)")
    print(f"adjusted rand    : {metrics['adjusted_rand_index']:.4f}  (1.0 = perfect)")
    print(f"purity           : {metrics['purity']:.4f}  (random baseline ≈ {1/6:.3f})")
    print(f"n_clusters       : {metrics['n_clusters']}")
    print("\ncluster → dominant genre:")
    for cid, genre in sorted(metrics["cluster_genre_map"].items()):
        size = metrics["cluster_sizes"][cid]
        print(f"  cluster {cid}: {genre:<12} ({size} songs)")


def print_pca_metrics(metrics: dict) -> None:
    n = metrics["n_components_used"]
    total_var = sum(metrics["explained_variance_ratio"])
    print(f"components used    : {n}")
    print(f"variance captured  : {total_var:.1%}")
    print(f"components for 95% : {metrics['components_for_95pct_variance']}")
    print(f"reconstruction MSE : {metrics['reconstruction_mse']:.6f}")
    print("\nexplained variance per component:")
    for i, v in enumerate(metrics["explained_variance_ratio"]):
        bar = "█" * int(v * 40)
        print(f"  PC{i + 1:<3}: {v:.1%}  {bar}")


# ── Plot functions ────────────────────────────────────────────────────────────

def save_cluster_scatter_plot(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    out_path: Path,
    title: str = "Cluster Scatter",
    explained: Optional[list[float]] = None,
) -> None:
    """Two side-by-side subplots: left=cluster coloring, right=true genre coloring."""
    genres = sorted(set(y_true))
    n_clusters = len(np.unique(cluster_labels))
    cluster_cmap = plt.cm.get_cmap("tab10", n_clusters)

    xlab = f"PC1 ({explained[0]:.1%} var)" if explained else "PC1"
    ylab = f"PC2 ({explained[1]:.1%} var)" if explained else "PC2"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    ax = axes[0]
    for cid in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.5,
                   color=cluster_cmap(cid), label=f"cluster {cid}")
    ax.set_title(f"{title} — by cluster")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(markerscale=2, fontsize=8, loc="best")

    ax = axes[1]
    for genre in genres:
        mask = y_true == genre
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.5,
                   color=_genre_color(genre), label=genre)
    ax.set_title(f"{title} — by true genre")
    ax.set_xlabel(xlab)
    ax.legend(markerscale=2, fontsize=8, loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_genre_cluster_heatmap(
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    genre_labels: list[str],
    out_path: Path,
    title: str = "Genre × Cluster Distribution",
) -> None:
    """Heatmap: rows=genres, cols=clusters, values=fraction of cluster that is each genre.

    Analogous to confusion_matrix_test.png for supervised models.
    """
    clusters = sorted(np.unique(cluster_labels))
    n_genres = len(genre_labels)
    n_clusters = len(clusters)

    mat = np.zeros((n_genres, n_clusters))
    for j, cid in enumerate(clusters):
        mask = cluster_labels == cid
        counts = Counter(y_true[mask])
        total = int(mask.sum())
        for i, genre in enumerate(genre_labels):
            mat[i, j] = counts.get(genre, 0) / total if total else 0

    fig, ax = plt.subplots(figsize=(max(7, n_clusters + 2), 5))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="fraction of cluster")
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"cluster {c}" for c in clusters], rotation=45, ha="right")
    ax.set_yticks(range(n_genres))
    ax.set_yticklabels(genre_labels)
    for i in range(n_genres):
        for j in range(n_clusters):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if mat[i, j] > 0.5 else "black")
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("True Genre")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_cluster_composition_plot(
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    genre_labels: list[str],
    out_path: Path,
    title: str = "Cluster Genre Composition",
) -> None:
    """Stacked horizontal bar: each cluster is a bar, color segments show genre fraction."""
    clusters = sorted(np.unique(cluster_labels))
    fractions = []
    for cid in clusters:
        mask = cluster_labels == cid
        counts = Counter(y_true[mask])
        total = int(mask.sum())
        fractions.append([counts.get(g, 0) / total for g in genre_labels])

    fractions = np.array(fractions)  # (n_clusters, n_genres)
    yticks = [f"cluster {c}" for c in clusters]

    fig, ax = plt.subplots(figsize=(9, max(4, len(clusters) * 0.8 + 1)))
    left = np.zeros(len(clusters))
    for i, genre in enumerate(genre_labels):
        ax.barh(yticks, fractions[:, i], left=left,
                color=_genre_color(genre), label=genre)
        left += fractions[:, i]

    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_elbow_plot(
    X: np.ndarray,
    k_range: range,
    out_path: Path,
    seed: int = 42,
    title: str = "K-Means Elbow",
) -> None:
    """Inertia vs k — helps select the number of clusters."""
    from sklearn.cluster import KMeans

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, "o-", color="steelblue")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title(title)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_pca_scatter_plot(
    X_2d: np.ndarray,
    y_true: np.ndarray,
    explained: list[float],
    out_path: Path,
    title: str = "PCA — PC1 vs PC2",
) -> None:
    """Scatter of PC1 vs PC2, colored by true genre."""
    genres = sorted(set(y_true))
    fig, ax = plt.subplots(figsize=(8, 6))
    for genre in genres:
        mask = y_true == genre
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.5,
                   color=_genre_color(genre), label=genre)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance explained)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance explained)")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_explained_variance_plot(
    explained_variance_ratio: list[float],
    out_path: Path,
    title: str = "PCA — Explained Variance",
) -> None:
    """Bar chart + cumulative line (scree plot)."""
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)
    x = list(range(1, n + 1))

    fig, ax1 = plt.subplots(figsize=(max(7, n * 0.6 + 2), 4))
    ax1.bar(x, explained_variance_ratio, color="steelblue", alpha=0.8, label="per-component")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, "o-", color="darkorange", linewidth=2, label="cumulative")
    ax2.axhline(0.95, color="red", linestyle="--", linewidth=1, label="95% threshold")
    ax2.set_ylabel("Cumulative Explained Variance", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_pca_loadings_plot(
    components: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    title: str = "PCA Feature Loadings",
    max_components: int = 2,
) -> None:
    """Horizontal bar chart of PC loadings (blue=positive, red=negative).

    The most interpretable PCA output: shows which audio/lyric features
    drive each principal component.
    """
    n = min(max_components, components.shape[0])
    fig, axes = plt.subplots(
        1, n,
        figsize=(7 * n, max(4, len(feature_names) * 0.4 + 1)),
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        loadings = components[i]
        colors = ["steelblue" if v >= 0 else "tomato" for v in loadings]
        ax.barh(feature_names, loadings, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"PC{i + 1} loadings")
        ax.set_xlabel("Loading")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
