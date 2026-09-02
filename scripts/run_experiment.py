"""End-to-end run: download data, build features, evaluate, save figures.

Outputs
-------
reports/figures/*.png   plots used in the README
reports/metrics.md      summary table of every model / protocol
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_prices  # noqa: E402
from src.evaluation import (  # noqa: E402
    chronological_holdout,
    coefficient_table,
    walk_forward,
)
from src.features import FEATURE_COLUMNS, build_dataset, split_features_target  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def fig_price_history(prices: pd.DataFrame) -> None:
    ax = prices["Close"].plot(figsize=(11, 4.5), color="#1f77b4")
    ax.set_title("SPY adjusted closing price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted price ($)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "price_history.png")
    plt.close()


def fig_target_balance(model_df: pd.DataFrame) -> None:
    counts = model_df["target"].value_counts(normalize=True).sort_index()
    ax = counts.plot(kind="bar", figsize=(5, 4), color=["#d62728", "#2ca02c"])
    ax.set_xticklabels(["Down / flat (0)", "Up (1)"], rotation=0)
    ax.set_ylabel("Share of trading days")
    ax.set_title("Next-day direction: class balance")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_balance.png")
    plt.close()


def fig_coefficients(coef_df: pd.DataFrame) -> None:
    ordered = coef_df.iloc[::-1]
    colors = ["#2ca02c" if c > 0 else "#d62728" for c in ordered["coefficient"]]
    ax = ordered.plot.barh(
        x="feature", y="coefficient", figsize=(7, 4.5), legend=False, color=colors
    )
    ax.set_xlabel("Standardised logistic-regression coefficient")
    ax.set_ylabel("")
    ax.set_title("Feature influence on P(next day up)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_coefficients.png")
    plt.close()


def fig_walk_forward_equity(wf_predictions: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Compare buy-and-hold vs. a long/flat strategy driven by the model.

    The strategy holds SPY for the next day only when the model predicts "up".
    Transaction costs are ignored, so this is an illustration of signal quality,
    not a trading result.
    """
    daily = prices["Close"].pct_change().reindex(wf_predictions.index).fillna(0.0)
    strat = daily.where(wf_predictions["predicted"] == 1, 0.0)

    equity = pd.DataFrame(
        {
            "Buy & hold": (1 + daily).cumprod(),
            "Model long/flat": (1 + strat).cumprod(),
        }
    )
    ax = equity.plot(figsize=(11, 4.5), color=["#7f7f7f", "#1f77b4"])
    ax.set_title("Growth of $1 over the walk-forward test window (no costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of $1")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "walk_forward_equity.png")
    plt.close()


def write_metrics(results, coef_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    lines = ["# Results", ""]
    lines.append(
        f"Data: SPY daily bars, {model_df.index.min().date()} to "
        f"{model_df.index.max().date()} "
        f"({len(model_df):,} usable trading days after feature warm-up).\n"
    )
    lines.append(
        "| Evaluation | Predictions | Model accuracy | Majority baseline | "
        "Edge | Model balanced acc. |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.n_predictions:,} | {r.accuracy:.2%} | "
            f"{r.baseline_accuracy:.2%} | {r.edge_over_baseline:+.2%} | "
            f"{r.balanced_accuracy:.2%} |"
        )
    lines += ["", "## Feature coefficients (full sample, standardised)", ""]
    lines.append("| Feature | Coefficient |")
    lines.append("|---|---:|")
    for _, row in coef_df.iterrows():
        lines.append(f"| {row['feature']} | {row['coefficient']:+.4f} |")
    lines += [
        "",
        "_A positive coefficient means a higher value of that feature raises the "
        "estimated probability that the next trading day closes up._",
        "",
    ]
    (ROOT / "reports" / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prices = load_prices()
    model_df = build_dataset(prices)
    X, y = split_features_target(model_df)

    print(f"Usable observations: {len(model_df):,}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")

    holdout = chronological_holdout(X, y)
    wf = walk_forward(X, y)
    coef_df = coefficient_table(X, y)

    for r in (holdout, wf):
        print(
            f"\n{r.name}\n  accuracy {r.accuracy:.2%} vs baseline "
            f"{r.baseline_accuracy:.2%} ({r.edge_over_baseline:+.2%}), "
            f"balanced {r.balanced_accuracy:.2%}, n={r.n_predictions:,}"
        )

    fig_price_history(prices)
    fig_target_balance(model_df)
    fig_coefficients(coef_df)
    fig_walk_forward_equity(wf.predictions, prices)
    write_metrics([holdout, wf], coef_df, model_df)
    print(f"\nFigures written to {FIG_DIR.relative_to(ROOT)}/")
    print(f"Metrics written to reports/metrics.md")


if __name__ == "__main__":
    main()
