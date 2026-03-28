"""Iris dataset loading and SVM pipeline training."""

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_PATH = "/tmp/model.joblib"
TARGET_NAMES: List[str] = ["setosa", "versicolor", "virginica"]


@dataclass
class TrainResult:
    accuracy: float
    train_size: int
    test_size: int


def train_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    kernel: str = "rbf",
) -> Tuple[Pipeline, TrainResult]:
    """Load Iris, split, fit StandardScaler + SVC pipeline, persist to disk."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=C, kernel=kernel)),
    ])
    pipe.fit(X_train, y_train)

    accuracy = float(pipe.score(X_test, y_test))
    joblib.dump(pipe, MODEL_PATH)

    return pipe, TrainResult(
        accuracy=accuracy,
        train_size=len(X_train),
        test_size=len(X_test),
    )


def load_pipeline() -> Pipeline:
    """Load persisted pipeline from disk."""
    return joblib.load(MODEL_PATH)


def predict(pipe: Pipeline, features: List[float]) -> Tuple[int, str]:
    """Return (species_id, species_name) for a single sample."""
    arr = np.array(features).reshape(1, -1)
    pred_id: int = int(pipe.predict(arr)[0])
    return pred_id, TARGET_NAMES[pred_id]
