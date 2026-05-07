"""Classifier factories for the three models in the proposal."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def make_logistic(seed: int = 42, mode: str = "audio") -> LogisticRegression:
    if mode == "lyrics":
        # liblinear converges reliably on sparse TF-IDF; OvR multi-class
        return LogisticRegression(
            max_iter=50000,
            solver="lbfgs",
            C=2.0,
            class_weight="balanced",
            random_state=seed,
        )
    # audio: 9 dense features, lbfgs + L2 is fast and accurate
    return LogisticRegression(
        max_iter=10000,
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def make_knn(mode: str = "audio") -> KNeighborsClassifier:
    if mode == "lyrics":
        # cosine distance is far better than euclidean on sparse TF-IDF
        return KNeighborsClassifier(
            n_neighbors=11,
            metric="cosine",
            weights="distance",
            n_jobs=-1,
        )
    # audio: euclidean on 9 z-scaled features, larger k for smoother boundaries
    return KNeighborsClassifier(
        n_neighbors=15,
        metric="minkowski",
        weights="distance",
        n_jobs=-1,
    )


def make_xgboost(seed: int = 42, mode: str = "audio"):
    """XGBoost gradient-boosted trees. Requires `pip install xgboost`."""
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost not installed. Run: pip install xgboost"
        ) from e
    if mode == "lyrics":
        # tuned via randomized search (tune.py)
        return XGBClassifier(
            n_estimators=700,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.3,
            min_child_weight=5,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=seed,
        )
    # audio: tuned via randomized search (tune.py)
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=seed,
    )


MODEL_FACTORIES = {
    "logistic": make_logistic,
    "knn": make_knn,
    "xgboost": make_xgboost,
}
