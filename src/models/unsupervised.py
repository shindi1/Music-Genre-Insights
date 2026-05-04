"""Unsupervised model factories for clustering and dimensionality reduction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ModelConfig:
    task: Literal["clustering", "reduction"]
    feature_set: Literal["audio+lyric"]
    use_dense: bool


def make_kmeans(n_clusters: int = 6, seed: int = 42) -> KMeans:
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=seed,
    )


def make_pca(n_components: int = 2, seed: int = 42) -> PCA:
    return PCA(n_components=n_components, random_state=seed)


def make_gmm(n_components: int = 6, seed: int = 42) -> GaussianMixture:
    """GMM with full covariance — handles overlapping, non-spherical genre clusters."""
    return GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=5,
        random_state=seed,
    )


UNSUPERVISED_MODEL_CONFIGS: dict[str, ModelConfig] = {
    "kmeans": ModelConfig(task="clustering", feature_set="audio+lyric", use_dense=True),
    "pca":    ModelConfig(task="reduction",  feature_set="audio+lyric", use_dense=True),
    "gmm":    ModelConfig(task="clustering", feature_set="audio+lyric", use_dense=True),
}

UNSUPERVISED_FACTORIES: dict[str, object] = {
    "kmeans": make_kmeans,
    "pca":    make_pca,
    "gmm":    make_gmm,
}
