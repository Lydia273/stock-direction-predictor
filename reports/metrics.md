# Results

Data: SPY daily bars, 2010-02-01 to 2026-09-02 (4,173 usable trading days after feature warm-up).

| Evaluation | Predictions | Model accuracy | Majority baseline | Edge | Model balanced acc. |
|---|---:|---:|---:|---:|---:|
| Chronological hold-out (last 20%) | 835 | 54.13% | 54.01% | +0.12% | 50.15% |
| Walk-forward backtest (expanding window, monthly retrain) | 3,173 | 53.86% | 53.83% | +0.03% | 50.24% |

## Feature coefficients (full sample, standardised)

| Feature | Coefficient |
|---|---:|
| distance_from_ma_5d | +0.1696 |
| return_5d | -0.1029 |
| intraday_return | -0.0823 |
| volume_change_1d | +0.0182 |
| volatility_5d | +0.0173 |
| distance_from_ma_20d | -0.0150 |
| return_1d | -0.0132 |
| return_10d | +0.0111 |

_A positive coefficient means a higher value of that feature raises the estimated probability that the next trading day closes up._
