"""
Investment strategy for Indian stock market.

THIS IS THE ONLY FILE THE AGENT MAY EDIT.

The agent iteratively improves compute_rebalance() to maximize the Sharpe ratio
on a backtest over Nifty 200 stocks from 2016-2026.

Strategy: Quality-Momentum Multi-Factor
- Momentum: 6-month and 12-month price returns (strongest factor in India)
- Quality: proxy via low volatility and consistent returns
- Trend filter: Nifty 50 above 200-day MA to avoid bear markets
- VIX regime: reduce exposure in high-volatility regimes
- Risk parity: weight by inverse volatility
- Sector diversification: cap any single sector at 30%
"""

import pandas as pd
import numpy as np


def compute_rebalance(
    date: pd.Timestamp,
    hist_close: pd.DataFrame,
    hist_volume: pd.DataFrame,
    available_tickers: list[str],
    sectors: dict[str, dict],
    current_holdings: dict[str, float],
    portfolio_value: float,
    vix: float | None,
    nifty50: pd.Series | None,
    state: dict,
) -> dict[str, float]:
    """
    Compute target portfolio weights for the given rebalance date.

    Parameters
    ----------
    date : pd.Timestamp
        Current rebalance date.
    hist_close : pd.DataFrame
        Historical adjusted close prices (dates x tickers), up to T-1.
        Contains full history back to 2014 for lookback calculations.
    hist_volume : pd.DataFrame
        Historical daily volume (dates x tickers), up to T-1.
    available_tickers : list[str]
        Tickers with valid prices today (tradeable universe).
    sectors : dict[str, dict]
        Mapping of ticker -> {sector, industry, marketCap}.
    current_holdings : dict[str, float]
        Current position values {ticker: market_value_inr}.
    portfolio_value : float
        Total portfolio value (cash + holdings).
    vix : float or None
        Latest India VIX reading. None if unavailable.
    nifty50 : pd.Series or None
        Nifty 50 close price series up to T-1. None if unavailable.
    state : dict
        Persistent state dictionary — survives across rebalance calls.
        Use this to track entry prices, regime flags, etc.

    Returns
    -------
    dict[str, float]
        Target portfolio weights {ticker: weight}.
        Weights should sum to <= 1.0 (remainder stays in cash).
        Return {} to go fully to cash.
    """
    # ── Minimum data requirement ─────────────────────────────────────────
    if len(hist_close) < 252:  # Need at least 1 year of history
        return {}

    # ── Trend filter: graded exposure based on Nifty vs MAs ─────────────
    # Instead of binary cash/invested, grade exposure by trend strength.
    # Nifty above 200-day MA: full exposure
    # Nifty between 100-day and 200-day MA: 50% exposure
    # Nifty below 100-day MA: 25% exposure (defensive, not cash)
    trend_multiplier = 1.0
    if nifty50 is not None and len(nifty50) >= 200:
        nifty_ma200 = nifty50.rolling(200).mean().iloc[-1]
        nifty_ma50 = nifty50.rolling(50).mean().iloc[-1]
        nifty_current = nifty50.iloc[-1]
        if nifty_current < nifty_ma50:
            trend_multiplier = 0.20
            state["regime"] = "bear"
        elif nifty_current < nifty_ma200:
            trend_multiplier = 0.75
            state["regime"] = "caution"
        else:
            state["regime"] = "bull"

    # Fixed position sizing — VIX gating removed (was cutting positions during recoveries)
    max_positions = 20
    max_total_exposure = 1.0

    # ── Filter universe ──────────────────────────────────────────────────
    # Only consider tickers with sufficient history
    tickers = [
        t for t in available_tickers
        if t in hist_close.columns
        and hist_close[t].dropna().shape[0] >= 252
    ]

    if len(tickers) < 10:
        return {}

    # ── Monthly rebalancing cadence (strategy-level) ─────────────────────
    # Rebalancing weekly creates excessive churn. We only rebalance monthly.
    last_rebalance = state.get("last_rebalance_date")
    if last_rebalance is not None:
        days_since = (date - last_rebalance).days
        if days_since < 25:  # Monthly cadence (~4 weeks)
            # Hold current positions unchanged
            total_val = sum(current_holdings.values())
            if total_val > 0 and current_holdings:
                return {t: v / portfolio_value for t, v in current_holdings.items()}
            return {}

    state["last_rebalance_date"] = date

    # ── Compute signals ──────────────────────────────────────────────────
    closes = hist_close[tickers]

    # Proper 12m-1m momentum: return from T-252 to T-21 (skip last month)
    # This is the standard implementation from academic literature
    p_now = closes.iloc[-21]       # price 1 month ago
    p_12m = closes.iloc[-252]      # price 12 months ago
    p_6m = closes.iloc[-126]       # price 6 months ago
    p_3m = closes.iloc[-63]        # price 3 months ago

    ret_12m = (p_now - p_12m) / p_12m
    ret_6m = (p_now - p_6m) / p_6m
    ret_3m = (p_now - p_3m) / p_3m

    # Composite momentum: blend of 12m, 6m, 3m with 1-month skip
    momentum = 0.4 * ret_12m + 0.35 * ret_6m + 0.25 * ret_3m

    # Volatility (annualized, 60-day window)
    daily_returns = closes.pct_change()
    vol_60d = daily_returns.iloc[-60:].std() * np.sqrt(252)

    # Volume confirmation: 20-day average volume vs 60-day (liquidity + interest)
    if hist_volume is not None and len(hist_volume) >= 60:
        vol_tickers = [t for t in tickers if t in hist_volume.columns]
        vol_20 = hist_volume[vol_tickers].iloc[-20:].mean()
        vol_60 = hist_volume[vol_tickers].iloc[-60:].mean()
        volume_ratio = vol_20 / vol_60.replace(0, np.nan)
    else:
        volume_ratio = pd.Series(1.0, index=tickers)

    # Relative strength vs Nifty 50 (6-month)
    rs_vs_nifty = pd.Series(np.nan, index=tickers)
    if nifty50 is not None and len(nifty50) >= 126:
        nifty_ret_6m = (nifty50.iloc[-21] - nifty50.iloc[-126]) / nifty50.iloc[-126]
        rs_vs_nifty = ret_6m - nifty_ret_6m  # Positive = outperforming Nifty

    # 52-week high proximity: ratio of current price to 52-week high
    # Stocks near their 52-week high tend to continue outperforming (George & Hwang 2004)
    high_52w = closes.iloc[-252:].max()
    price_to_52w_high = closes.iloc[-1] / high_52w.replace(0, np.nan)

    # Return consistency: fraction of positive monthly returns in past 12 months
    # 12 non-overlapping monthly return windows, each 21 days
    monthly_rets = pd.DataFrame(index=tickers)
    for i in range(12):
        start_idx = -(252 - i * 21)
        end_idx = -(252 - (i + 1) * 21) if i < 11 else -21
        if end_idx == 0:
            end_idx = None
        if end_idx is not None:
            month_ret = (closes.iloc[end_idx] - closes.iloc[start_idx]) / closes.iloc[start_idx].replace(0, np.nan)
        else:
            month_ret = (closes.iloc[-1] - closes.iloc[start_idx]) / closes.iloc[start_idx].replace(0, np.nan)
        monthly_rets[f"m{i}"] = month_ret
    consistency = (monthly_rets > 0).sum(axis=1) / 12.0

    # ── Score and rank ───────────────────────────────────────────────────
    scores = pd.DataFrame(index=tickers)
    scores["momentum"] = momentum
    scores["inv_vol"] = 1.0 / vol_60d.replace(0, np.nan)  # Lower vol = better
    scores["vol_confirm"] = volume_ratio.reindex(tickers, fill_value=1.0)
    scores["rs_nifty"] = rs_vs_nifty
    scores["near_52w_high"] = price_to_52w_high
    scores["consistency"] = consistency

    # Drop tickers with NaN scores
    scores = scores.dropna()

    if len(scores) < 5:
        return {}

    # Rank each factor (higher rank = better)
    for col in scores.columns:
        scores[f"{col}_rank"] = scores[col].rank(pct=True)

    # Composite score: momentum-heavy, with relative strength
    # Momentum acceleration as ranked factor
    scores["mom_accel"] = ret_3m - ret_6m
    scores["mom_accel_rank"] = scores["mom_accel"].rank(pct=True)

    scores["composite"] = (
        0.28 * scores["momentum_rank"]
        + 0.15 * scores["inv_vol_rank"]
        + 0.07 * scores["vol_confirm_rank"]
        + 0.17 * scores["rs_nifty_rank"]
        + 0.13 * scores["mom_accel_rank"]
        + 0.10 * scores["near_52w_high_rank"]
        + 0.10 * scores["consistency_rank"]
    )

    # ── Select top stocks ────────────────────────────────────────────────
    # Filter: positive momentum AND outperforming Nifty AND accelerating momentum
    scores = scores[
        (scores["momentum"] > 0)
        & (scores["rs_nifty"] > 0)
        & (scores["mom_accel"] > 0)
    ]

    if len(scores) < 3:
        return {}

    # Sort by composite score
    scores = scores.sort_values("composite", ascending=False)

    # Select top N
    selected = scores.head(max_positions)

    # ── Sector diversification ───────────────────────────────────────────
    sector_cap = 0.30
    sector_counts: dict[str, int] = {}
    final_tickers = []

    for ticker in selected.index:
        sector = sectors.get(ticker, {}).get("sector", "Unknown")
        count = sector_counts.get(sector, 0)
        max_per_sector = max(2, int(max_positions * sector_cap))
        if count < max_per_sector:
            final_tickers.append(ticker)
            sector_counts[sector] = count + 1

    if len(final_tickers) < 3:
        return {}

    # ── Weight by inverse volatility (risk parity) ───────────────────────
    selected_vols = vol_60d[final_tickers].replace(0, np.nan).dropna()
    final_tickers = list(selected_vols.index)

    if len(final_tickers) < 3:
        return {}

    inv_vol = 1.0 / selected_vols
    inv_vol_weights = inv_vol / inv_vol.sum()
    equal_weights = pd.Series(1.0 / len(final_tickers), index=final_tickers)

    # Score-based weights: use composite score to upweight top picks
    score_weights_raw = scores.loc[final_tickers, "composite"].clip(lower=0)
    score_weights = score_weights_raw / score_weights_raw.sum() if score_weights_raw.sum() > 0 else equal_weights

    # Blend: 60% inverse-vol, 40% score-weighted (remove equal-weight)
    raw_weights = 0.60 * inv_vol_weights + 0.40 * score_weights

    # Cap individual position at 10%
    max_weight = 0.10
    capped = raw_weights.clip(upper=max_weight)
    # Redistribute excess
    capped = capped / capped.sum()

    # Scale by total exposure limit and trend multiplier
    weights = (capped * max_total_exposure * trend_multiplier).to_dict()

    return weights
