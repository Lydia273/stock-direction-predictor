"""Feature engineering for next-day direction prediction.

Every feature uses only information available by the close of the current
trading day. The target is the *next* trading day's direction, so there is no
look-ahead leakage between the features and the label.
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "distance_from_ma_5d",
    "distance_from_ma_20d",
    "volatility_5d",
    "volume_change_1d",
    "intraday_return",
]

TARGET_COLUMN = "target"


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready frame with engineered features and the target.

    The target is ``1`` when the next trading day closes above its open and
    ``0`` otherwise (down or flat).
    """
    df = prices.sort_index().copy()

    # Intraday move for the current day (close vs. open).
    df["intraday_return"] = df["Close"] / df["Open"] - 1.0

    # Trailing close-to-close returns over several horizons.
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Position relative to short and medium moving averages.
    ma_5 = df["Close"].rolling(window=5).mean()
    ma_20 = df["Close"].rolling(window=20).mean()
    df["distance_from_ma_5d"] = df["Close"] / ma_5 - 1.0
    df["distance_from_ma_20d"] = df["Close"] / ma_20 - 1.0

    # Recent realised volatility and change in traded volume.
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()
    df["volume_change_1d"] = df["Volume"].pct_change(1)

    # Label: direction of the following trading day.
    next_day_intraday = df["intraday_return"].shift(-1)
    df["target"] = (next_day_intraday > 0).astype("Int64")

    model_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    model_df[TARGET_COLUMN] = model_df[TARGET_COLUMN].astype(int)
    return model_df


def split_features_target(model_df: pd.DataFrame):
    """Split a model-ready frame into an ``X`` matrix and ``y`` vector."""
    return model_df[FEATURE_COLUMNS], model_df[TARGET_COLUMN]
