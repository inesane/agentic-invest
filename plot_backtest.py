"""
Visualization for backtest results.

Generates:
1. backtest_equity.png  — equity curve vs benchmarks with shaded max-drawdown regions
2. progress.png         — experiment progress scatter (Sharpe over time)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from pathlib import Path

from data_fetcher import load_all_data
from main import run_backtest, BACKTEST_START, BACKTEST_END, DATA_FETCH_START, INITIAL_CAPITAL, compute_metrics

OUTPUT_DIR = Path(__file__).parent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _max_drawdown_window(equity: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) dates of the maximum drawdown period."""
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    trough_idx = drawdown.idxmin()
    peak_idx = cummax.loc[:trough_idx].idxmax()
    return peak_idx, trough_idx


def _drawdown_series(equity: pd.Series) -> pd.Series:
    cummax = equity.cummax()
    return (equity - cummax) / cummax * 100


# ── Equity Curve Plot ─────────────────────────────────────────────────────────

def plot_results(results: dict, save: bool = True):
    """
    Generate equity curve comparison chart.
    Mirrors the style of greenfish8090/claude-investment-strategy:
    - Single equity panel with three curves
    - Shaded region shows each curve's maximum drawdown window
    - Subtitle with key metrics per curve
    """
    equity = results["equity_curve"]
    metrics = results["metrics"]
    bench_metrics = results["benchmark_metrics"]
    nifty50_raw = results.get("benchmark_equity")
    benchmarks_dict = results.get("benchmarks_raw", {})

    # Align all series to backtest window
    start, end = equity.index[0], equity.index[-1]

    # Build curves dict: name -> (series_normalized, color, linestyle, metrics_dict)
    COLORS = {
        "Strategy": "#2563eb",
        "Nifty 50": "#6b7280",
        "Nifty 500": "#d97706",
    }

    curves = {}

    # Strategy
    strat_norm = equity / equity.iloc[0] * 100
    curves["Strategy"] = (strat_norm, metrics)

    # Nifty 50
    if nifty50_raw is not None:
        n50 = nifty50_raw.loc[(nifty50_raw.index >= start) & (nifty50_raw.index <= end)].dropna()
        if len(n50) > 20:
            n50_norm = n50 / n50.iloc[0] * 100
            n50_metrics = compute_metrics(n50 / n50.iloc[0] * INITIAL_CAPITAL)
            curves["Nifty 50"] = (n50_norm, n50_metrics)

    # Nifty 500
    if "NIFTY500" in benchmarks_dict:
        n500_raw = benchmarks_dict["NIFTY500"]
        col = "Adj Close" if "Adj Close" in n500_raw.columns else "Close"
        n500_s = n500_raw[col]
        if isinstance(n500_s, pd.DataFrame):
            n500_s = n500_s.iloc[:, 0]
        n500 = n500_s.loc[(n500_s.index >= start) & (n500_s.index <= end)].dropna()
        if len(n500) > 20:
            n500_norm = n500 / n500.iloc[0] * 100
            n500_metrics = compute_metrics(n500 / n500.iloc[0] * INITIAL_CAPITAL)
            curves["Nifty 500"] = (n500_norm, n500_metrics)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 1, hspace=0.08)
    ax_eq = fig.add_subplot(gs[:3, 0])   # equity (3/4 height)
    ax_dd = fig.add_subplot(gs[3, 0], sharex=ax_eq)  # drawdown (1/4)

    # ── Subtitle string ───────────────────────────────────────────────────
    subtitle_parts = []
    for name, (series, m) in curves.items():
        sharpe = m.get("sharpe", "—")
        cagr = m.get("cagr_pct", "—")
        maxdd = m.get("max_drawdown_pct", "—")
        subtitle_parts.append(f"{name}: Sharpe {sharpe}, CAGR {cagr}%, Max DD {maxdd}%")
    subtitle = "   |   ".join(subtitle_parts)

    fig.suptitle(
        "Backtest: Indian Stock Market Strategy vs Benchmarks, 2016–2026",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    ax_eq.set_title(subtitle, fontsize=8.5, color="#555", pad=6)

    # ── Plot curves + shade max drawdown windows ──────────────────────────
    ls_map = {"Strategy": "-", "Nifty 50": "--", "Nifty 500": ":"}
    lw_map = {"Strategy": 2.0, "Nifty 50": 1.3, "Nifty 500": 1.3}

    for name, (series, m) in curves.items():
        color = COLORS.get(name, "#888")
        ls = ls_map.get(name, "-")
        lw = lw_map.get(name, 1.5)

        ax_eq.plot(series.index, series.values, label=name, color=color,
                   linewidth=lw, linestyle=ls, zorder=3)

        # Shade max drawdown window
        try:
            peak_dt, trough_dt = _max_drawdown_window(series)
            ax_eq.axvspan(peak_dt, trough_dt, alpha=0.10, color=color, zorder=1)
        except Exception:
            pass

    ax_eq.set_ylabel("Value (₹100 start)", fontsize=11)
    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax_eq.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax_eq.grid(True, alpha=0.25, zorder=0)
    ax_eq.tick_params(labelbottom=False)

    # ── Drawdown panel ────────────────────────────────────────────────────
    strat_dd = _drawdown_series(strat_norm)
    ax_dd.fill_between(strat_dd.index, strat_dd.values, 0,
                       color=COLORS["Strategy"], alpha=0.35, zorder=2)
    ax_dd.plot(strat_dd.index, strat_dd.values,
               color=COLORS["Strategy"], linewidth=0.8, zorder=3)

    if "Nifty 50" in curves:
        n50_dd = _drawdown_series(curves["Nifty 50"][0])
        ax_dd.plot(n50_dd.index, n50_dd.values,
                   color=COLORS["Nifty 50"], linewidth=0.8, linestyle="--",
                   alpha=0.7, zorder=2)

    ax_dd.set_ylabel("Drawdown", fontsize=10)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.set_ylim(min(strat_dd.min() * 1.15, -5), 2)
    ax_dd.grid(True, alpha=0.25)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.tick_params(axis="x", labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        path = OUTPUT_DIR / "backtest_equity.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")
    else:
        plt.show()
    plt.close()


# ── Progress Plot ─────────────────────────────────────────────────────────────

def plot_progress(results_file: str | None = None, save: bool = True):
    """
    Scatter plot of Sharpe ratio over experiments — mirrors progress.png
    from the original repo.
    Gray dots = discarded, green dots = kept, blue step line = running best.
    """
    path = Path(results_file) if results_file else OUTPUT_DIR / "results.tsv"
    if not path.exists():
        print(f"No results file at {path}")
        return

    df = pd.read_csv(path, sep="\t")
    if len(df) == 0:
        print("No results to plot")
        return

    # Deduplicate exact same timestamps (re-runs of same experiment)
    df = df.drop_duplicates(subset=["timestamp", "sharpe", "description"])
    df = df.reset_index(drop=True)
    df["exp_num"] = range(len(df))

    kept = df[df["status"] == "kept"]
    discarded = df[df["status"] == "discarded"]

    n_total = len(df)
    n_kept = len(kept)
    best_sharpe = kept["sharpe"].max() if len(kept) > 0 else 0

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        f"Strategy Research Progress: {n_total} Experiments, {n_kept} Kept Improvements",
        fontsize=14,
        fontweight="bold",
    )

    # ── Top panel: Sharpe scatter ─────────────────────────────────────────
    ax1 = axes[0]

    if len(discarded) > 0:
        ax1.scatter(discarded["exp_num"], discarded["sharpe"],
                    color="#9ca3af", alpha=0.5, s=20, zorder=2, label="Discarded")
    if len(kept) > 0:
        ax1.scatter(kept["exp_num"], kept["sharpe"],
                    color="#22c55e", alpha=0.9, s=40, zorder=3, label="Kept")
        # Running best as step line
        running_best = kept["sharpe"].cummax()
        ax1.step(kept["exp_num"], running_best,
                 color="#2563eb", linewidth=2, where="post",
                 zorder=4, label=f"Best Sharpe ({best_sharpe:.4f})")

    ax1.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelbottom=False)

    # ── Bottom panel: CAGR + Max DD scatter ──────────────────────────────
    ax2 = axes[1]

    if len(discarded) > 0:
        ax2.scatter(discarded["exp_num"], discarded["cagr_pct"],
                    color="#9ca3af", alpha=0.3, s=18, zorder=1)
        ax2.scatter(discarded["exp_num"], discarded["max_dd_pct"],
                    color="#9ca3af", alpha=0.3, s=18, zorder=1)
    if len(kept) > 0:
        ax2.scatter(kept["exp_num"], kept["cagr_pct"],
                    color="#22c55e", alpha=0.9, s=35, zorder=3, label="CAGR %")
        ax2.scatter(kept["exp_num"], kept["max_dd_pct"],
                    color="#ef4444", alpha=0.7, s=35, zorder=3, label="Max DD %")
        # Connect kept points with lines
        ax2.plot(kept["exp_num"], kept["cagr_pct"],
                 color="#22c55e", alpha=0.4, linewidth=1, zorder=2)
        ax2.plot(kept["exp_num"], kept["max_dd_pct"],
                 color="#ef4444", alpha=0.4, linewidth=1, zorder=2)

    ax2.axhline(0, color="#666", linewidth=0.6, linestyle="--")
    ax2.set_ylabel("CAGR % / Max DD %", fontsize=10)
    ax2.set_xlabel("Experiment #", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out = OUTPUT_DIR / "progress.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--progress" in sys.argv:
        plot_progress()
        sys.exit(0)

    print("Loading data...")
    data = load_all_data(start=DATA_FETCH_START, end=BACKTEST_END)

    print("Running backtest...")
    results = run_backtest(
        close_prices=data["close_prices"],
        volume=data["volume"],
        benchmarks=data["benchmarks"],
        sectors=data["sectors"],
        verbose=not ("--quiet" in sys.argv),
    )

    if results:
        # Attach raw benchmarks for Nifty 500
        results["benchmarks_raw"] = data["benchmarks"]
        plot_results(results)
        plot_progress()
    else:
        print("Backtest failed, nothing to plot.")
