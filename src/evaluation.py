"""Model training and honest time-series evaluation.

Two evaluation protocols are provided:

* ``chronological_holdout`` - a single 80 / 20 split in time order. Quick to run
  and easy to read, but the test score depends on one particular period.
* ``walk_forward`` - an expanding-window backtest that retrains every N trading
  days and always predicts the *next* block. This is closer to how the model
  would actually be used and gives a more trustworthy estimate.

Both compare the model against a majority-class baseline (always predict the
most common class in the training data).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model() -> Pipeline:
    """Standardise features, then fit L2-regularised logistic regression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(max_iter=1000)),
        ]
    )


@dataclass
class EvalResult:
    name: str
    accuracy: float
    balanced_accuracy: float
    baseline_accuracy: float
    baseline_balanced_accuracy: float
    n_predictions: int
    predictions: pd.DataFrame = field(repr=False)

    @property
    def edge_over_baseline(self) -> float:
        return self.accuracy - self.baseline_accuracy


def chronological_holdout(
    X: pd.DataFrame, y: pd.Series, train_fraction: float = 0.80
) -> EvalResult:
    split = int(len(X) * train_fraction)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_model().fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    majority_class = int(y_train.mode()[0])
    baseline = np.full(len(y_test), majority_class)

    frame = pd.DataFrame(
        {"actual": y_test.to_numpy(), "predicted": preds, "probability_up": proba},
        index=y_test.index,
    )
    return EvalResult(
        name="Chronological hold-out (last 20%)",
        accuracy=accuracy_score(y_test, preds),
        balanced_accuracy=balanced_accuracy_score(y_test, preds),
        baseline_accuracy=accuracy_score(y_test, baseline),
        baseline_balanced_accuracy=balanced_accuracy_score(y_test, baseline),
        n_predictions=len(y_test),
        predictions=frame,
    )


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train: int = 1000,
    step: int = 21,
) -> EvalResult:
    """Expanding-window backtest.

    Start with ``initial_train`` days, predict the next ``step`` days, then add
    those days to the training set and repeat. ``step=21`` is roughly one
    trading month between retrains.
    """
    rows: list[pd.DataFrame] = []
    for start in range(initial_train, len(X), step):
        end = min(start + step, len(X))
        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_test, y_test = X.iloc[start:end], y.iloc[start:end]
        if y_train.nunique() < 2 or len(X_test) == 0:
            continue

        model = make_model().fit(X_train, y_train)
        majority_class = int(y_train.mode()[0])
        rows.append(
            pd.DataFrame(
                {
                    "actual": y_test.to_numpy(),
                    "predicted": model.predict(X_test),
                    "probability_up": model.predict_proba(X_test)[:, 1],
                    "baseline": majority_class,
                },
                index=y_test.index,
            )
        )

    frame = pd.concat(rows)
    return EvalResult(
        name="Walk-forward backtest (expanding window, monthly retrain)",
        accuracy=accuracy_score(frame["actual"], frame["predicted"]),
        balanced_accuracy=balanced_accuracy_score(frame["actual"], frame["predicted"]),
        baseline_accuracy=accuracy_score(frame["actual"], frame["baseline"]),
        baseline_balanced_accuracy=balanced_accuracy_score(
            frame["actual"], frame["baseline"]
        ),
        n_predictions=len(frame),
        predictions=frame,
    )


def coefficient_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Logistic-regression coefficients on standardised features (full sample).

    Useful for reading which features push the prediction toward "up".
    """
    model = make_model().fit(X, y)
    coefs = model.named_steps["logistic_regression"].coef_[0]
    return (
        pd.DataFrame({"feature": X.columns, "coefficient": coefs})
        .sort_values("coefficient", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )
