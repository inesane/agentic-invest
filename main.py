"""
Backtesting engine for Indian stock market strategy.

This file is LOCKED — the agent must NOT modify it.
It simulates portfolio management with:
- Weekly rebalancing on Mondays
- Configurable commission (default 0.1% per side — India brokerage)
- Slippage model (0.05%)
- Performance metrics: Sharpe, Sortino, Calmar, CAGR, max drawdown
- Benchmark comparison against Nifty 50

Usage:
    python main.py
"""

import sys
import os
import datetime
import importlib
import traceback
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from data_fetcher import load_all_data, build_close_price_matrix, build_volume_matrix

# ── Configuration ────────────────────────────────────────────────────────────

BACKTEST_START = "2016-01-01"
BACKTEST_END = "2026-01-01"
DATA_FETCH_START = "2014-01-01"  # Extra history for lookback calculations

INITIAL_CAPITAL = 10_000_000  # 1 crore INR
COMMISSION_BPS = 10  # 0.1% per side (typical Indian discount broker)
SLIPPAGE_BPS = 5  # 0.05% slippage
MAX_LEVERAGE = 1.0  # No leverage for Indian retail
REBALANCE_DAY = 0  # Monday = 0
REBALANCE_FREQ_WEEKS = 1  # Rebalance every N weeks

RESULTS_FILE = Path(__file__).parent / "results.tsv"

# ── Broker Simulation ────────────────────────────────────────────────────────

class VirtualBroker:
    """Simulates trade execution with commission and slippage."""

    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.positions: dict[str, float] = {}  # ticker -> num_shares
        self.trade_log: list[dict] = []

    def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value = cash + sum(shares * price)."""
        holdings_value = sum(
            shares * prices.get(ticker, 0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + holdings_value

    def get_holdings_value(self, prices: dict[str, float]) -> dict[str, float]:
        """Return dict of ticker -> current market value."""
        return {
            ticker: shares * prices.get(ticker, 0)
            for ticker, shares in self.positions.items()
            if shares > 0
        }

    def execute_rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        date: datetime.date,
    ) -> list[dict]:
        """
        Rebalance portfolio to target weights.
        Returns list of executed trades.
        """
        portfolio_value = self.get_portfolio_value(prices)
        trades = []

        # Calculate target positions
        target_values = {
            ticker: weight * portfolio_value
            for ticker, weight in target_weights.items()
            if weight > 0 and ticker in prices and prices[ticker] > 0
        }

        # Current position values
        current_values = self.get_holdings_value(prices)

        # All tickers involved
        all_tickers = set(list(target_values.keys()) + list(current_values.keys()))

        # Calculate trades needed
        for ticker in all_tickers:
            target_val = target_values.get(ticker, 0)
            current_val = current_values.get(ticker, 0)
            delta_val = target_val - current_val
            price = prices.get(ticker, 0)

            if price <= 0 or abs(delta_val) < 100:  # Skip tiny trades
                continue

            shares_delta = delta_val / price
            # Apply slippage
            if shares_delta > 0:  # Buying
                effective_price = price * (1 + SLIPPAGE_BPS / 10000)
            else:  # Selling
                effective_price = price * (1 - SLIPPAGE_BPS / 10000)

            trade_value = abs(shares_delta * effective_price)
            commission = trade_value * COMMISSION_BPS / 10000

            # Execute
            self.positions[ticker] = self.positions.get(ticker, 0) + shares_delta
            self.cash -= shares_delta * effective_price + commission

            # Clean up zero/negative positions
            if self.positions[ticker] <= 0.001:
                self.positions.pop(ticker, None)

            trade = {
                "date": date,
                "ticker": ticker,
                "shares": shares_delta,
                "price": price,
                "effective_price": effective_price,
                "value": trade_value,
                "commission": commission,
                "side": "BUY" if shares_delta > 0 else "SELL",
            }
            trades.append(trade)
            self.trade_log.append(trade)

        return trades


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.06) -> dict:
    """
    Compute performance metrics from an equity curve.
    Uses 6% risk-free rate (approximate Indian T-bill rate).
    """
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 20:
        return {}

    # Annualization factor (252 trading days)
    ann_factor = 252

    # CAGR
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = (total_return ** (365.25 / total_days)) - 1 if total_days > 0 else 0

    # Volatility
    annual_vol = returns.std() * np.sqrt(ann_factor)

    # Sharpe Ratio
    daily_rf = (1 + risk_free_rate) ** (1 / ann_factor) - 1
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(ann_factor) if excess_returns.std() > 0 else 0

    # Sortino Ratio
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 1
    sortino = (excess_returns.mean() * ann_factor) / downside_std if downside_std > 0 else 0

    # Max Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    win_rate = (returns > 0).mean()

    # Average positions (from equity curve alone we can't tell, but we'll add this from outside)

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "cagr_pct": round(cagr * 100, 2),
        "annual_vol_pct": round(annual_vol * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round((total_return - 1) * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
    }


# ── Backtest Runner ──────────────────────────────────────────────────────────

def run_backtest(
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    benchmarks: dict[str, pd.DataFrame],
    sectors: dict[str, dict],
    start_date: str = BACKTEST_START,
    end_date: str = BACKTEST_END,
    verbose: bool = True,
) -> dict:
    """
    Run the backtest using the strategy defined in strategy.py.

    The strategy's compute_rebalance() function is called on each rebalance date
    and must return a dict of {ticker: weight} representing target portfolio weights.
    Weights should sum to <= 1.0 (remainder stays in cash).
    """
    # Import strategy (allows hot-reload)
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    import strategy

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    # Filter to backtest period (but keep earlier data for lookback)
    all_dates = close_prices.index
    backtest_dates = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]

    if len(backtest_dates) == 0:
        print("ERROR: No dates in backtest range")
        return {}

    broker = VirtualBroker(INITIAL_CAPITAL)
    equity_curve = {}
    position_counts = []
    rebalance_count = 0
    last_rebalance_week = None
    state = {}  # Persistent state dict passed to strategy

    # Get Nifty 50 close for benchmark
    nifty50_close = None
    if "NIFTY50" in benchmarks:
        ndf = benchmarks["NIFTY50"]
        col = "Adj Close" if "Adj Close" in ndf.columns else "Close"
        nifty50_close = ndf[col]
        if isinstance(nifty50_close, pd.DataFrame):
            nifty50_close = nifty50_close.iloc[:, 0]

    # India VIX series
    vix_series = None
    if "INDIAVIX" in benchmarks:
        vdf = benchmarks["INDIAVIX"]
        col = "Close"
        vix_series = vdf[col]
        if isinstance(vix_series, pd.DataFrame):
            vix_series = vix_series.iloc[:, 0]

    if verbose:
        print(f"Backtesting from {start_date} to {end_date}")
        print(f"Universe: {len(close_prices.columns)} stocks")
        print(f"Initial capital: ₹{INITIAL_CAPITAL:,.0f}")
        print()

    for date in backtest_dates:
        # Get prices for today
        today_prices = close_prices.loc[date].dropna().to_dict()

        # Record equity
        pv = broker.get_portfolio_value(today_prices)
        equity_curve[date] = pv

        # Check if rebalance day
        week_num = date.isocalendar()[1]
        is_rebalance = (
            date.weekday() == REBALANCE_DAY
            and (last_rebalance_week is None or week_num != last_rebalance_week)
        )

        if not is_rebalance:
            position_counts.append(len(broker.positions))
            continue

        last_rebalance_week = week_num

        # Prepare data for strategy
        # Historical prices up to yesterday (T-1 to avoid lookahead)
        hist_close = close_prices.loc[:date].iloc[:-1]  # Exclude today
        hist_volume = volume.loc[:date].iloc[:-1]

        # Available tickers: those with a valid price today
        available_tickers = list(today_prices.keys())

        # Current holdings
        current_holdings = broker.get_holdings_value(today_prices)

        # VIX value
        vix_value = None
        if vix_series is not None:
            vix_before = vix_series.loc[:date]
            if len(vix_before) > 0:
                vix_value = float(vix_before.iloc[-1])

        # Nifty 50 series up to yesterday
        nifty50_hist = None
        if nifty50_close is not None:
            nifty50_hist = nifty50_close.loc[:date].iloc[:-1]

        # Call strategy
        try:
            target_weights = strategy.compute_rebalance(
                date=date,
                hist_close=hist_close,
                hist_volume=hist_volume,
                available_tickers=available_tickers,
                sectors=sectors,
                current_holdings=current_holdings,
                portfolio_value=pv,
                vix=vix_value,
                nifty50=nifty50_hist,
                state=state,
            )
        except Exception as e:
            print(f"Strategy error on {date}: {e}")
            traceback.print_exc()
            target_weights = {}

        # Validate weights
        if target_weights:
            # Clamp weights
            target_weights = {
                t: max(0, min(1, w))
                for t, w in target_weights.items()
                if w > 0.001 and t in today_prices
            }

            total_weight = sum(target_weights.values())
            if total_weight > MAX_LEVERAGE:
                # Scale down proportionally
                scale = MAX_LEVERAGE / total_weight
                target_weights = {t: w * scale for t, w in target_weights.items()}

        # Execute rebalance
        trades = broker.execute_rebalance(target_weights, today_prices, date)
        rebalance_count += 1
        position_counts.append(len(broker.positions))

        if verbose and rebalance_count % 26 == 0:  # Print every ~6 months
            print(
                f"  {date.strftime('%Y-%m-%d')} | "
                f"₹{pv:>14,.0f} | "
                f"{len(broker.positions):>3} positions | "
                f"{len(trades):>3} trades"
            )

    # Build equity curve series
    equity_series = pd.Series(equity_curve, name="Portfolio")
    equity_series.index = pd.to_datetime(equity_series.index)

    # Compute strategy metrics
    metrics = compute_metrics(equity_series)
    metrics["avg_positions"] = round(np.mean(position_counts), 1) if position_counts else 0
    metrics["total_rebalances"] = rebalance_count
    metrics["total_trades"] = len(broker.trade_log)

    # Compute benchmark metrics
    bench_metrics = {}
    if nifty50_close is not None:
        bench_eq = nifty50_close.loc[
            (nifty50_close.index >= start_dt) & (nifty50_close.index <= end_dt)
        ]
        if len(bench_eq) > 20:
            # Normalize to same starting capital for comparison
            bench_eq_norm = bench_eq / bench_eq.iloc[0] * INITIAL_CAPITAL
            bench_metrics = compute_metrics(bench_eq_norm)

    if verbose:
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ₹{INITIAL_CAPITAL:,.0f}")
        print(f"Final Value:     ₹{equity_series.iloc[-1]:,.0f}")
        print()
        print(f"{'Metric':<25} {'Strategy':>12} {'Nifty 50':>12}")
        print("-" * 50)
        for key in ["cagr_pct", "sharpe", "sortino", "calmar", "max_drawdown_pct", "annual_vol_pct"]:
            strat_val = metrics.get(key, "N/A")
            bench_val = bench_metrics.get(key, "N/A")
            label = key.replace("_pct", " %").replace("_", " ").title()
            print(f"  {label:<23} {strat_val:>12} {bench_val:>12}")
        print()
        print(f"  Avg Positions:          {metrics.get('avg_positions', 'N/A'):>12}")
        print(f"  Total Rebalances:       {metrics.get('total_rebalances', 'N/A'):>12}")
        print(f"  Total Trades:           {metrics.get('total_trades', 'N/A'):>12}")
        print("=" * 70)

    return {
        "equity_curve": equity_series,
        "metrics": metrics,
        "benchmark_metrics": bench_metrics,
        "benchmark_equity": nifty50_close,
        "broker": broker,
    }


def log_result(metrics: dict, description: str, status: str = "kept"):
    """Append a result row to results.tsv."""
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write(
                "timestamp\tsharpe\tsortino\tcalmar\tcagr_pct\t"
                "max_dd_pct\tavg_positions\tstatus\tdescription\n"
            )

    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{datetime.datetime.now().isoformat()}\t"
            f"{metrics.get('sharpe', 0)}\t"
            f"{metrics.get('sortino', 0)}\t"
            f"{metrics.get('calmar', 0)}\t"
            f"{metrics.get('cagr_pct', 0)}\t"
            f"{metrics.get('max_drawdown_pct', 0)}\t"
            f"{metrics.get('avg_positions', 0)}\t"
            f"{status}\t"
            f"{description}\n"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest Indian stock market strategy")
    parser.add_argument("--start", default=BACKTEST_START, help="Backtest start date")
    parser.add_argument("--end", default=BACKTEST_END, help="Backtest end date")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    parser.add_argument("--description", default="", help="Description for results log")
    args = parser.parse_args()

    # Load data
    print("Loading market data...")
    data = load_all_data(start=DATA_FETCH_START, end=args.end)

    # Run backtest
    results = run_backtest(
        close_prices=data["close_prices"],
        volume=data["volume"],
        benchmarks=data["benchmarks"],
        sectors=data["sectors"],
        start_date=args.start,
        end_date=args.end,
        verbose=not args.quiet,
    )

    if not results:
        print("Backtest failed!")
        sys.exit(1)

    metrics = results["metrics"]

    # Log result
    if args.description:
        log_result(metrics, args.description)

    if args.json:
        print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    main()
