"""End-to-end run: download data, build features, evaluate models, save figures.

Compares a majority-class baseline, logistic regression and gradient boosting
under both a chronological hold-out and an expanding-window walk-forward backtest.

Outputs
-------
reports/figures/*.png   plots used in the README
reports/metrics.md      full results table
"""

from __future__ import annotations

import os
import warnings

# Silence a benign joblib/loky message on Windows when `wmic` is not on PATH.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_prices  # noqa: E402
from src.evaluation import (  # noqa: E402
    chronological_holdout,
    coefficient_table,
    permutation_importance_table,
    walk_forward,
)
from src.features import FEATURE_COLUMNS, build_dataset, split_features_target  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
MODELS = ["logistic", "gradient_boosting"]


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
    ax.set_title("Logistic regression: feature influence on P(up)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_coefficients.png")
    plt.close()


def fig_permutation_importance(imp_df: pd.DataFrame) -> None:
    ordered = imp_df.iloc[::-1]
    ax = ordered.plot.barh(
        x="feature",
        y="importance",
        xerr="std",
        figsize=(7, 4.5),
        legend=False,
        color="#1f77b4",
    )
    ax.set_xlabel("Mean accuracy drop when the feature is shuffled")
    ax.set_ylabel("")
    ax.set_title("Gradient boosting: permutation importance (hold-out)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance.png")
    plt.close()


def fig_accuracy_comparison(results: list) -> None:
    wf = [r for r in results if r.protocol.startswith("Walk-forward")]
    labels = [r.model for r in wf]
    acc = [r.accuracy for r in wf]
    base = wf[0].baseline_accuracy

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, acc, color=["#1f77b4", "#ff7f0e"], width=0.5)
    ax.axhline(base, color="#7f7f7f", linestyle="--", label=f"majority baseline ({base:.1%})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.50, max(acc + [base]) + 0.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Walk-forward accuracy vs. baseline")
    for xi, a in zip(x, acc):
        ax.text(xi, a + 0.002, f"{a:.2%}", ha="center")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "accuracy_comparison.png")
    plt.close()


def fig_walk_forward_equity(wf_predictions: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Buy-and-hold vs. a long/flat strategy driven by the model (no costs)."""
    daily = prices["Close"].pct_change().reindex(wf_predictions.index).fillna(0.0)
    strat = daily.where(wf_predictions["predicted"] == 1, 0.0)
    equity = pd.DataFrame(
        {
            "Buy & hold": (1 + daily).cumprod(),
            "Model long/flat": (1 + strat).cumprod(),
        }
    )
    ax = equity.plot(figsize=(11, 4.5), color=["#7f7f7f", "#1f77b4"])
    ax.set_title("Growth of $1 over the walk-forward window — best model (no costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of $1")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "walk_forward_equity.png")
    plt.close()


def write_metrics(results, coef_df, imp_df, model_df) -> None:
    lines = ["# Results", ""]
    lines.append(
        f"Data: SPY daily bars, {model_df.index.min().date()} to "
        f"{model_df.index.max().date()} "
        f"({len(model_df):,} usable trading days after feature warm-up).\n"
    )
    lines.append(
        "| Model | Evaluation | Predictions | Accuracy | Majority baseline | "
        "Edge | Balanced acc. | ROC AUC |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.model} | {r.protocol} | {r.n_predictions:,} | {r.accuracy:.2%} | "
            f"{r.baseline_accuracy:.2%} | {r.edge_over_baseline:+.2%} | "
            f"{r.balanced_accuracy:.2%} | {r.roc_auc:.3f} |"
        )
    lines += ["", "## Logistic regression — coefficients (full sample, standardised)", ""]
    lines += ["| Feature | Coefficient |", "|---|---:|"]
    for _, row in coef_df.iterrows():
        lines.append(f"| {row['feature']} | {row['coefficient']:+.4f} |")
    lines += ["", "## Gradient boosting — permutation importance (hold-out)", ""]
    lines += ["| Feature | Importance | Std |", "|---|---:|---:|"]
    for _, row in imp_df.iterrows():
        lines.append(
            f"| {row['feature']} | {row['importance']:+.4f} | {row['std']:.4f} |"
        )
    lines += [
        "",
        "_Positive logistic coefficient: a higher feature value raises the "
        "estimated probability the next day closes up. Permutation importance: "
        "how much hold-out accuracy falls when that feature is shuffled._",
        "",
    ]
    (ROOT / "reports" / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prices = load_prices()
    model_df = build_dataset(prices)
    X, y = split_features_target(model_df)

    print(f"Usable observations: {len(model_df):,}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}\n")

    results = []
    for kind in MODELS:
        for evaluator in (chronological_holdout, walk_forward):
            r = evaluator(X, y, kind=kind)
            results.append(r)
            print(
                f"{r.model:<20} {r.protocol:<52} "
                f"acc {r.accuracy:.2%} (base {r.baseline_accuracy:.2%}, "
                f"{r.edge_over_baseline:+.2%})  bal {r.balanced_accuracy:.2%}  "
                f"AUC {r.roc_auc:.3f}  n={r.n_predictions:,}"
            )

    coef_df = coefficient_table(X, y)
    imp_df = permutation_importance_table(X, y)

    best_wf = max(
        (r for r in results if r.protocol.startswith("Walk-forward")),
        key=lambda r: r.accuracy,
    )
    print(f"\nBest walk-forward model: {best_wf.model} ({best_wf.accuracy:.2%})")

    fig_price_history(prices)
    fig_target_balance(model_df)
    fig_coefficients(coef_df)
    fig_permutation_importance(imp_df)
    fig_accuracy_comparison(results)
    fig_walk_forward_equity(best_wf.predictions, prices)
    write_metrics(results, coef_df, imp_df, model_df)
    print(f"\nFigures written to {FIG_DIR.relative_to(ROOT)}/")
    print("Metrics written to reports/metrics.md")


if __name__ == "__main__":
    main()
