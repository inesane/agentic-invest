"""
Visualization for backtest results.

Generates:
1. Equity curve comparison (strategy vs Nifty 50)
2. Drawdown chart
3. Rolling Sharpe ratio
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from data_fetcher import load_all_data
from main import run_backtest, BACKTEST_START, BACKTEST_END, DATA_FETCH_START, INITIAL_CAPITAL

OUTPUT_DIR = Path(__file__).parent


def plot_results(results: dict, save: bool = True):
    """Generate all plots from backtest results."""
    equity = results["equity_curve"]
    metrics = results["metrics"]
    bench_metrics = results["benchmark_metrics"]
    nifty50_close = results.get("benchmark_equity")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        "Indian Stock Market Strategy — Backtest Results",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # ── 1. Equity Curve ──────────────────────────────────────────────────
    ax1 = axes[0]

    # Normalize both to start at same value
    strat_norm = equity / equity.iloc[0] * 100
    ax1.plot(strat_norm.index, strat_norm.values, label="Strategy", color="#2563eb", linewidth=1.5)

    if nifty50_close is not None:
        bench_period = nifty50_close.loc[
            (nifty50_close.index >= equity.index[0])
            & (nifty50_close.index <= equity.index[-1])
        ]
        if len(bench_period) > 0:
            bench_norm = bench_period / bench_period.iloc[0] * 100
            ax1.plot(bench_norm.index, bench_norm.values, label="Nifty 50", color="#9ca3af", linewidth=1.2, linestyle="--")

    ax1.set_ylabel("Value (₹100 start)", fontsize=11)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f"Sharpe: {metrics.get('sharpe', 'N/A')} | "
        f"CAGR: {metrics.get('cagr_pct', 'N/A')}% | "
        f"Max DD: {metrics.get('max_drawdown_pct', 'N/A')}% | "
        f"Nifty Sharpe: {bench_metrics.get('sharpe', 'N/A')}",
        fontsize=10,
        color="#666",
    )

    # ── 2. Drawdown Chart ────────────────────────────────────────────────
    ax2 = axes[1]
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#ef4444", alpha=0.4)
    ax2.plot(drawdown.index, drawdown.values, color="#ef4444", linewidth=0.8)
    ax2.set_ylabel("Drawdown %", fontsize=11)
    ax2.set_ylim(drawdown.min() * 1.1, 2)
    ax2.grid(True, alpha=0.3)

    # ── 3. Rolling 6-month Sharpe ────────────────────────────────────────
    ax3 = axes[2]
    returns = equity.pct_change().dropna()
    rolling_sharpe = (
        returns.rolling(126).mean() / returns.rolling(126).std()
    ) * np.sqrt(252)
    ax3.plot(rolling_sharpe.index, rolling_sharpe.values, color="#8b5cf6", linewidth=0.8)
    ax3.axhline(y=0, color="#666", linewidth=0.5, linestyle="--")
    ax3.axhline(y=1, color="#22c55e", linewidth=0.5, linestyle="--", alpha=0.5)
    ax3.set_ylabel("Rolling Sharpe (6m)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = OUTPUT_DIR / "backtest_equity.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved equity curve to {path}")
    else:
        plt.show()

    plt.close()


def plot_progress(results_file: str = "results.tsv"):
    """Plot experiment progress from results.tsv."""
    path = Path(results_file)
    if not path.exists():
        print(f"No results file found at {path}")
        return

    df = pd.read_csv(path, sep="\t")
    if len(df) == 0:
        print("No results to plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Experiment Progress", fontsize=14, fontweight="bold")

    # Sharpe over time
    ax1 = axes[0]
    kept = df[df["status"] == "kept"]
    discarded = df[df["status"] == "discarded"]

    ax1.scatter(range(len(discarded)), discarded["sharpe"], color="#ef4444", alpha=0.4, s=15, label="Discarded")
    ax1.scatter(range(len(kept)), kept["sharpe"], color="#22c55e", alpha=0.8, s=25, label="Kept")

    # Running best
    if len(kept) > 0:
        running_best = kept["sharpe"].cummax()
        ax1.plot(range(len(kept)), running_best, color="#2563eb", linewidth=1.5, label="Best Sharpe")

    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CAGR over time
    ax2 = axes[1]
    ax2.scatter(range(len(discarded)), discarded["cagr_pct"], color="#ef4444", alpha=0.4, s=15, label="Discarded")
    ax2.scatter(range(len(kept)), kept["cagr_pct"], color="#22c55e", alpha=0.8, s=25, label="Kept")
    ax2.set_ylabel("CAGR %")
    ax2.set_xlabel("Experiment #")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "progress.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved progress chart to {path}")
    plt.close()


if __name__ == "__main__":
    if "--progress" in sys.argv:
        plot_progress()
    else:
        print("Loading data...")
        data = load_all_data(start=DATA_FETCH_START, end=BACKTEST_END)

        print("Running backtest...")
        results = run_backtest(
            close_prices=data["close_prices"],
            volume=data["volume"],
            benchmarks=data["benchmarks"],
            sectors=data["sectors"],
        )

        if results:
            plot_results(results)
        else:
            print("Backtest failed, nothing to plot.")
