"""Download and cache daily price data.

The raw data is one row per trading day (Open / High / Low / Close / Volume),
adjusted for splits and dividends. It is cached locally as a CSV file so the
rest of the pipeline is reproducible without hitting the network every run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TICKER = "SPY"
DEFAULT_START = "2010-01-01"


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker.lower()}_daily.csv"


def load_prices(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of daily OHLCV data indexed by date.

    Parameters
    ----------
    ticker:
        Market symbol to download (default ``"SPY"``, the SPDR S&P 500 ETF).
    start, end:
        Date range. ``end=None`` downloads through the latest available day.
    refresh:
        If ``True``, ignore any local cache and re-download.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(ticker)

    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col="Date", parse_dates=["Date"])

    frame = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    frame = frame.dropna().copy()
    frame.index.name = "Date"
    frame.to_csv(cache)
    return frame


if __name__ == "__main__":
    prices = load_prices(refresh=True)
    print(f"{len(prices):,} trading days "
          f"({prices.index.min().date()} to {prices.index.max().date()})")
