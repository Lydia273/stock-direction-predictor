"""Model training and honest time-series evaluation.

Models
------
* ``logistic``          - StandardScaler -> L2 logistic regression (linear).
* ``gradient_boosting`` - HistGradientBoostingClassifier (captures non-linear
  feature interactions; no scaling needed).

Evaluation protocols
--------------------
* ``chronological_holdout`` - a single 80 / 20 split in time order. Quick to read,
  but the score depends on one particular period.
* ``walk_forward`` - an expanding-window backtest that retrains every N trading
  days and always predicts the *next* block. Closer to real use and more
  trustworthy.

Every result is compared against a majority-class baseline (always predict the
most common class in the training window).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_LABELS = {
    "logistic": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}

_MODEL_FACTORIES: dict[str, Callable[[], object]] = {
    "logistic": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    ),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=0,
    ),
}


def make_model(kind: str = "logistic"):
    """Return a fresh, unfitted estimator for ``kind``."""
    try:
        return _MODEL_FACTORIES[kind]()
    except KeyError:
        raise ValueError(
            f"unknown model {kind!r}; choose from {list(_MODEL_FACTORIES)}"
        ) from None


@dataclass
class EvalResult:
    model: str
    protocol: str
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    baseline_accuracy: float
    baseline_balanced_accuracy: float
    n_predictions: int
    predictions: pd.DataFrame = field(repr=False)

    @property
    def edge_over_baseline(self) -> float:
        return self.accuracy - self.baseline_accuracy


def _summarise(model: str, protocol: str, frame: pd.DataFrame) -> EvalResult:
    actual = frame["actual"]
    return EvalResult(
        model=model,
        protocol=protocol,
        accuracy=accuracy_score(actual, frame["predicted"]),
        balanced_accuracy=balanced_accuracy_score(actual, frame["predicted"]),
        roc_auc=roc_auc_score(actual, frame["probability_up"]),
        baseline_accuracy=accuracy_score(actual, frame["baseline"]),
        baseline_balanced_accuracy=balanced_accuracy_score(actual, frame["baseline"]),
        n_predictions=len(frame),
        predictions=frame,
    )


def chronological_holdout(
    X: pd.DataFrame, y: pd.Series, kind: str = "logistic", train_fraction: float = 0.80
) -> EvalResult:
    split = int(len(X) * train_fraction)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_model(kind).fit(X_train, y_train)
    majority_class = int(y_train.mode()[0])

    frame = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": model.predict(X_test),
            "probability_up": model.predict_proba(X_test)[:, 1],
            "baseline": majority_class,
        },
        index=y_test.index,
    )
    return _summarise(MODEL_LABELS[kind], "Chronological hold-out (last 20%)", frame)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str = "logistic",
    initial_train: int = 1000,
    step: int = 21,
) -> EvalResult:
    """Expanding-window backtest.

    Start with ``initial_train`` days, predict the next ``step`` days, add them to
    the training set, repeat. ``step=21`` is roughly one trading month between
    retrains.
    """
    rows: list[pd.DataFrame] = []
    for start in range(initial_train, len(X), step):
        end = min(start + step, len(X))
        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_test, y_test = X.iloc[start:end], y.iloc[start:end]
        if y_train.nunique() < 2 or len(X_test) == 0:
            continue

        model = make_model(kind).fit(X_train, y_train)
        rows.append(
            pd.DataFrame(
                {
                    "actual": y_test.to_numpy(),
                    "predicted": model.predict(X_test),
                    "probability_up": model.predict_proba(X_test)[:, 1],
                    "baseline": int(y_train.mode()[0]),
                },
                index=y_test.index,
            )
        )

    frame = pd.concat(rows)
    return _summarise(
        MODEL_LABELS[kind],
        "Walk-forward backtest (expanding window, monthly retrain)",
        frame,
    )


def coefficient_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Logistic-regression coefficients on standardised features (full sample)."""
    model = make_model("logistic").fit(X, y)
    coefs = model.named_steps["clf"].coef_[0]
    return (
        pd.DataFrame({"feature": X.columns, "coefficient": coefs})
        .sort_values("coefficient", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_table(
    X: pd.DataFrame, y: pd.Series, kind: str = "gradient_boosting"
) -> pd.DataFrame:
    """Permutation importance on a chronological hold-out (drop in accuracy)."""
    split = int(len(X) * 0.80)
    model = make_model(kind).fit(X.iloc[:split], y.iloc[:split])
    result = permutation_importance(
        model, X.iloc[split:], y.iloc[split:], n_repeats=20, random_state=0, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
