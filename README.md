# agentic-invest

An autonomous experiment loop where Claude iterates on a stock selection strategy for the Indian market. Claude proposes a hypothesis, edits one file, runs a backtest, and decides whether to keep or revert the change — repeating until interrupted.

![backtest equity curve](backtest_equity.png)
*Strategy vs Nifty 50 and Nifty 500, 2016–2026*

![experiment progress](progress.png)
*69 experiments. Green = kept, grey = discarded. Step line = running best Sharpe.*

---

## Results

| | Strategy | Nifty 50 | Nifty 500 |
|---|---|---|---|
| Sharpe | **1.25** | 0.49 | 0.52 |
| CAGR | **22.5%** | 12.9% | 13.5% |
| Max Drawdown | **-10.9%** | -38.4% | -38.3% |
| Annual Volatility | **12.5%** | 16.2% | 16.0% |
| Sortino | **1.83** | 0.59 | — |

Backtest: Jan 2016 – Jan 2026. Universe: ~187 Nifty 200 stocks. Initial capital: ₹1 crore.

---

## How it works

Claude is given:
- A backtesting harness (`main.py`) it cannot modify
- A strategy file (`strategy.py`) it can freely edit
- A scoring rule: maximize Sharpe ratio

Each experiment follows the same loop:
1. Propose a hypothesis (e.g. "adding a short-term reversal screen should reduce drawdowns")
2. Edit `strategy.py`
3. Run `python main.py --json` and read the metrics
4. If Sharpe improved: `git commit`. If not: `git checkout strategy.py`
5. Repeat

The agent cannot cheat — `main.py` is the single source of truth and uses only T-1 data. There is no way to accidentally look ahead.

---

## What the strategy actually does

After 69 experiments, the strategy that emerged:

**Scoring each stock:**
- Composite momentum (12m, 6m, 3m returns — all with a 1-month skip to avoid short-term reversal)
- Momentum acceleration: is the 3-month trend stronger than the 6-month? Stocks where momentum is building rank higher
- Relative strength vs Nifty 50 over 6 months
- Inverse 60-day realized volatility (low-vol stocks score higher)
- Proximity to 52-week high
- Return consistency: what fraction of the past 12 months were positive
- Volume trend: is recent volume higher than the 60-day average

**Hard filters before scoring:**
- Positive composite momentum
- Outperforming Nifty 50 over 6 months
- Accelerating momentum (3m return > 6m return)
- Not in the top 7.5% of stocks by raw 1-month return — this guards against chasing stocks that already had a big move

**Market regime:**
- If Nifty 50 is above its 200-day MA: full exposure
- Between 50-day and 200-day MA: 75% exposure
- Below 50-day MA: 20% exposure (stay mostly in cash during selloffs)

**Portfolio construction:**
- Up to 20 stocks, capped at 30% per sector and 10% per position
- Weights are a blend of inverse-volatility (60%) and composite-score (40%)
- Rebalanced monthly — not weekly, to avoid excessive churn

---

## What the experiments revealed

A few findings that were non-obvious:

**Monthly rebalancing matters a lot.** Switching from weekly to monthly cadence was the single biggest early improvement. Transaction costs from weekly churn were destroying returns.

**VIX-based position sizing backfired.** Cutting positions when VIX was elevated made the strategy sit in cash during some of the best recovery rallies. Removing it improved results.

**The 50-day MA is a better bear signal than the 100-day.** When Nifty 50 crosses below its 50-day MA it is a more timely warning than the 100-day. Using it as the bear threshold (with 20% exposure) significantly improved the Sharpe and reduced drawdown.

**Excluding stocks with very recent large moves helps.** A 1-month raw return screen that strips out the top 7.5% of recent movers cut the max drawdown from ~20% to ~11%. The intuition: stocks that just ran hard are prone to short-term pullbacks, and you don't want to be buying them at the top.

**Score-weighting the portfolio improves risk-adjusted returns.** Pure inverse-vol weighting (risk parity) ignores signal quality. Blending it with composite-score weighting means the stocks the model is most confident in get slightly more capital.

---

## Setup

```bash
conda create -n invest python=3.11 -y
conda activate invest
pip install yfinance matplotlib pandas numpy requests tqdm

# First run fetches all data (~2 min) and caches it
python main.py
```

After the first run, everything works offline from `data/`.

---

## Running the agent

Open Claude Code in this directory:

```
Hi, have a look at program.md and let's kick off a session!
```

Claude will read `results.tsv` for the current best Sharpe, pick a hypothesis, edit `strategy.py`, run the backtest, and decide whether to keep the change. It will keep going until you interrupt it.

---

## Files

```
strategy.py          — the only file the agent edits
main.py              — backtest engine (fixed, do not modify)
data_fetcher.py      — downloads and caches Nifty 200 data via yfinance
plot_backtest.py     — generates the charts above
program.md           — instructions given to the agent
results.tsv          — log of every experiment
data/                — cached price, volume, sector, and benchmark data
```

---

## Data

All data via [yfinance](https://github.com/ranaroussi/yfinance), no API key needed:

- NSE tickers (e.g. `RELIANCE.NS`) for ~187 Nifty 200 stocks
- Benchmarks: Nifty 50 (`^NSEI`), Nifty 500 (`^CRSLDX`)
- India VIX (`^INDIAVIX`)
- Yahoo Finance sector/industry classification

---

## Caveats

**Survivorship bias.** The universe is current Nifty 200 constituents. Stocks that were removed or delisted between 2016 and 2026 are not represented, which inflates performance.

**Transaction costs.** The backtest charges 0.1% per side. Real-world costs also include STT, exchange fees, and GST — likely another 0.3–0.5% per year off CAGR.

**Short backtest window.** 2016–2026 was mostly a bull market for India. The strategy hasn't been tested against anything like 2008.

---

## License

MIT
