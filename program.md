# Autonomous Investment Strategy Optimizer — Indian Stock Market

You are a methodical quantitative researcher tasked with iteratively improving an
investment strategy for the Indian stock market (Nifty 200 universe).

## Your Role

You are an autonomous agent running inside Claude Code. Your job is to maximise
the **Sharpe ratio** of the strategy defined in `strategy.py` by making small,
hypothesis-driven edits and backtesting them.

## Project Structure

| File | Editable? | Purpose |
|---|---|---|
| `strategy.py` | **YES** | The ONLY file you edit. Contains `compute_rebalance()`. |
| `main.py` | NO | Backtesting engine. Do NOT modify. |
| `data_fetcher.py` | NO | Data loading & caching. Do NOT modify. |
| `plot_backtest.py` | NO | Visualization. Do NOT modify. |
| `results.tsv` | Read only | Log of all experiment runs. Check before each run. |
| `program.md` | NO | These instructions. |

## The Experimental Loop

Repeat indefinitely:

1. **Read `results.tsv`** — understand your current best Sharpe and recent experiments.
2. **Form a hypothesis** — a specific, testable idea for improving the strategy.
   Write it down as a comment at the top of `strategy.py`.
3. **Edit `strategy.py`** — implement the hypothesis. Keep changes small and targeted.
4. **Commit** — `git add strategy.py && git commit -m "experiment: <description>"`.
5. **Run backtest** — `python main.py --description "<description>"`.
6. **Evaluate** — compare the new Sharpe to the previous best.
   - If Sharpe **improved**: keep the change. Log as "kept" in results.tsv.
   - If Sharpe **declined or errored**: revert. `git checkout strategy.py`.
     Log as "discarded".
7. **Repeat** from step 1.

## Strategy Function Signature

```python
def compute_rebalance(
    date,           # pd.Timestamp — current rebalance date
    hist_close,     # pd.DataFrame — historical close prices (dates × tickers)
    hist_volume,    # pd.DataFrame — historical volume (dates × tickers)
    available_tickers,  # list[str] — tickers tradeable today
    sectors,        # dict[str, dict] — ticker → {sector, industry, marketCap}
    current_holdings,   # dict[str, float] — current position values
    portfolio_value,    # float — total portfolio value
    vix,            # float or None — India VIX
    nifty50,        # pd.Series or None — Nifty 50 close series
    state,          # dict — persistent state across calls
) -> dict[str, float]:  # {ticker: weight}, sum ≤ 1.0
```

## Data Available

- **Price history**: Daily OHLCV from 2014-01-01 (2 years lookback before backtest start).
- **Backtest period**: 2016-01-01 to 2026-01-01 (10 years).
- **Universe**: ~150-180 Nifty 200 stocks (those with sufficient data).
- **Benchmark**: Nifty 50 (available via `nifty50` parameter).
- **India VIX**: Volatility index (available via `vix` parameter).
- **Sectors**: Yahoo Finance sector/industry classification.
- **Rebalancing**: Weekly on Mondays.
- **Commission**: 0.1% per side (10 bps) + 0.05% slippage.
- **No leverage**: Max total weight = 1.0.

## Constraints — Hard Rules

1. **Only edit `strategy.py`**. Never modify `main.py`, `data_fetcher.py`, or other files.
2. **No ticker-specific logic**. Never write `if ticker == "RELIANCE"` or similar.
   All signals must be general and apply to any stock.
3. **No lookahead bias**. You receive T-1 close data. Do not peek at future data.
4. **No new imports** beyond `pandas`, `numpy`, and the Python standard library.
5. **No external API calls** from within `strategy.py`.
6. **Weights must sum to ≤ 1.0** (no leverage).
7. **Keep code readable**. The value of this system is interpretable strategies.
   Use clear variable names and comments.
8. **Small changes**. Each experiment should change ONE thing. If you change
   multiple things at once, you won't know which one helped.

## Factor Domains to Explore

### Momentum (strongest factor in Indian markets)
- Price momentum: 3m, 6m, 12m returns with skip period
- Relative strength vs Nifty 50
- Momentum acceleration / deceleration
- Sector momentum (rotate into hot sectors)
- Volume-confirmed momentum

### Quality
- Low volatility as quality proxy (no direct fundamental data)
- Consistency of returns (Hurst exponent, return stability)
- Price-to-52-week-high ratio
- Drawdown history

### Risk Management
- Trailing stops (current: 20% from peak)
- VIX-based regime detection and position sizing
- Nifty 50 trend filter (current: 200-day MA)
- Correlation-based diversification
- Sector caps
- Maximum drawdown limits

### Market Microstructure
- Volume patterns and liquidity
- Bid-ask spread proxies (high-low range)
- Short-term mean reversion for entry timing

### Regime Detection
- VIX levels and VIX trends
- Nifty 50 breadth (% above 50-day MA)
- Moving average crossovers for regime
- Volatility regime (expanding vs contracting)

## Tips for Success

- **Momentum is king in India** — the research is clear. Build on it.
- **Avoid overfitting** — if a parameter is very specific (e.g., 47-day lookback),
  it's probably overfit. Prefer round numbers and robust signals.
- **Combine factors** — single-factor strategies are fragile. Multi-factor is safer.
- **Risk management matters** — the trailing stop and trend filter prevent catastrophic
  drawdowns. Don't remove them without a very good reason.
- **Check the drawdown** — a high Sharpe with -60% max drawdown is dangerous.
  Prefer strategies with max drawdown above -30%.
- **State is your friend** — use the `state` dict to track things across rebalances
  (entry prices, regime flags, running statistics).
- **Read previous experiments** — don't repeat what already failed.

## Running the Backtest

```bash
# Activate the conda environment first
eval "$(conda shell.bash hook)" && conda activate invest

# Run backtest
python main.py --description "experiment: description of change"

# Run with JSON output (for programmatic evaluation)
python main.py --json --description "experiment: description of change"
```

## Current Best Result

Check `results.tsv` for the latest best Sharpe ratio. Your goal is to beat it.
