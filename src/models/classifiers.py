"""Classifier factories for the three models in the proposal."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def make_logistic(seed: int = 42) -> LogisticRegression:
    """Multinomial logistic regression baseline."""
    return LogisticRegression(
        max_iter=10000,
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def make_knn(n_neighbors: int = 15) -> KNeighborsClassifier:
    """KNN classifier — sensitive to feature scaling, so use the z-scaled audio cols."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        n_jobs=-1,
    )


def make_xgboost(seed: int = 42):
    """XGBoost gradient-boosted trees. Requires `pip install xgboost`."""
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost not installed. Run: pip install xgboost"
        ) from e
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
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
