# stock-direction-predictor

Predicting the **next trading day's direction** for the S&P 500 ETF (SPY) from
daily price and volume data — with time-series validation that does not cheat.

The headline result is deliberately unglamorous: a carefully built logistic
regression **does not beat a majority-class baseline** once it is evaluated with a
walk-forward backtest. The value of this repo is the *methodology* — leak-free
features, chronological splits, an expanding-window backtest, and an honest
comparison against the right baseline.

## Problem setup

| | |
|---|---|
| **Asset** | SPY (SPDR S&P 500 ETF), daily bars, adjusted for splits/dividends |
| **Sample** | 2010 → present, ~4,200 usable trading days after feature warm-up |
| **Target** | `1` if the next trading day closes above its open, else `0` |
| **Baseline** | Always predict the most common class in the training data (~54%) |
| **Model** | `StandardScaler` → `LogisticRegression` (L2) |

Features (all computed from information available by today's close, so there is no
look-ahead leakage into the label):

- trailing close-to-close returns over 1 / 5 / 10 days
- distance of the close from its 5- and 20-day moving averages
- 5-day realised volatility of daily returns
- 1-day change in traded volume
- today's intraday return (close / open − 1)

## Results

Data: SPY daily bars, 2010-02-01 → 2026-09-02 (4,173 usable trading days).

| Evaluation | Predictions | Model accuracy | Majority baseline | Edge | Model balanced acc. |
|---|---:|---:|---:|---:|---:|
| Chronological hold-out (last 20%) | 835 | 54.13% | 54.01% | +0.12% | 50.15% |
| Walk-forward backtest (expanding window, monthly retrain) | 3,173 | 53.86% | 53.83% | +0.03% | 50.24% |

**Reading:** the model tracks the baseline to within a fraction of a percent, and
its balanced accuracy sits at ~50%. There is no exploitable next-day signal in
this feature set — which is the expected outcome for a liquid index at daily
frequency, and exactly why the evaluation protocol matters.

<p align="center">
  <img src="reports/figures/price_history.png" width="80%"><br>
  <img src="reports/figures/target_balance.png" width="45%">
  <img src="reports/figures/feature_coefficients.png" width="52%"><br>
  <img src="reports/figures/walk_forward_equity.png" width="80%">
</p>

The equity chart holds SPY only on days the model predicts "up" (no transaction
costs). It underperforms buy-and-hold — the classifier is not identifying good
days to be in the market.

Full numbers, including the feature-coefficient table, are regenerated into
[`reports/metrics.md`](reports/metrics.md) on every run.

## Repository layout

```
stock-direction-predictor/
├── src/
│   ├── data.py         # download + local CSV cache of daily bars (yfinance)
│   ├── features.py     # leak-free feature engineering and target construction
│   └── evaluation.py   # chronological hold-out + walk-forward backtest, metrics
├── scripts/
│   └── run_experiment.py   # end-to-end run → figures + reports/metrics.md
├── notebooks/
│   └── 01_data_exploration.ipynb   # narrated EDA → first baseline model
├── reports/
│   ├── figures/        # PNGs used above
│   └── metrics.md      # generated results table
└── requirements.txt
```

## Setup & run

```bash
git clone https://github.com/Lydia273/stock-direction-predictor.git
cd stock-direction-predictor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# reproduce every figure and the metrics table
python scripts/run_experiment.py
```

The first run downloads ~15 years of SPY data and caches it under `data/`
(git-ignored); later runs are offline. To open the walkthrough notebook:

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

## Next steps

- Richer features: market breadth, term structure, options-implied volatility,
  simple macro series.
- Models that capture interactions (gradient-boosted trees) under the **same**
  walk-forward protocol.
- Move from direction to a small expected-return threshold and add realistic
  costs before drawing any strategy conclusions.

## License

MIT — see [LICENSE](LICENSE).
