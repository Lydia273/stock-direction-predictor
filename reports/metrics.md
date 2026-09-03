# Results

Data: SPY daily bars, 2010-02-01 to 2026-09-02 (4,173 usable trading days after feature warm-up).

| Model | Evaluation | Predictions | Accuracy | Majority baseline | Edge | Balanced acc. | ROC AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | Chronological hold-out (last 20%) | 835 | 54.13% | 54.01% | +0.12% | 50.15% | 0.518 |
| Logistic regression | Walk-forward backtest (expanding window, monthly retrain) | 3,173 | 53.86% | 53.83% | +0.03% | 50.24% | 0.495 |
| Gradient boosting | Chronological hold-out (last 20%) | 835 | 54.13% | 54.01% | +0.12% | 50.19% | 0.513 |
| Gradient boosting | Walk-forward backtest (expanding window, monthly retrain) | 3,173 | 53.39% | 53.83% | -0.44% | 49.82% | 0.495 |

## Logistic regression — coefficients (full sample, standardised)

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

## Gradient boosting — permutation importance (hold-out)

| Feature | Importance | Std |
|---|---:|---:|
| volatility_5d | +0.0041 | 0.0035 |
| return_10d | +0.0026 | 0.0044 |
| volume_change_1d | +0.0013 | 0.0005 |
| intraday_return | +0.0010 | 0.0009 |
| return_1d | -0.0002 | 0.0006 |
| distance_from_ma_5d | -0.0004 | 0.0014 |
| return_5d | -0.0007 | 0.0023 |
| distance_from_ma_20d | -0.0015 | 0.0012 |

_Positive logistic coefficient: a higher feature value raises the estimated probability the next day closes up. Permutation importance: how much hold-out accuracy falls when that feature is shuffled._
