"""Betting backtest: model probabilities vs. Kalshi market prices.

Kalshi contracts settle at $1 if the event happens, $0 otherwise, so the price
IS the market's probability. If our model thinks a side is worth more than its
price (by more than `edge_threshold`), we buy that side.

Per contract bought at price p:
    win  -> +(1 - p)
    lose -> -p
    minus an entry fee (Kalshi's ~0.07 * p * (1-p), rounded up per order).

Honesty caveats (read before trusting any number here):
  * The model still uses full-season player rates and starter ERA -> it has
    partially seen the outcomes. Backtest returns are optimistic.
  * One price per game (the ~1 h pre-game mark). Real fills move the book,
    especially with size. No slippage modelled.
  * 826 games ~ half a season. A 3 % ROI edge still has wide error bars here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def kalshi_fee(price: float, contracts: float = 1.0, rate: float = 0.07) -> float:
    """Kalshi trading fee, rounded up to the cent per order."""
    return math.ceil(rate * contracts * price * (1.0 - price) * 100.0) / 100.0


def run_backtest(games: pd.DataFrame, model_prob, *, edge_threshold: float = 0.0,
                 stake: str = "flat", bankroll: float = 1000.0,
                 kelly_fraction: float = 0.25, kelly_cap: float = 0.05,
                 fee_rate: float = 0.07, apply_fee: bool = True,
                 exclude_extreme: tuple = (0.05, 0.95),
                 slippage: float = 0.0, max_contracts: float | None = None,
                 max_stake_frac: float | None = None) -> dict:
    """Simulate betting every game where |model - market| > edge_threshold.

    stake="flat"  -> 1 contract per bet, P&L summed (no compounding).
    stake="kelly" -> fractional-Kelly share of the *current* bankroll, compounded.

    Fill realism:
      slippage        cents added to the fill price against you (0.01 = 1c worse)
      max_contracts   absolute cap on contracts per bet
      max_stake_frac  cap contracts at this fraction of that game's `volume`
                      column, if present

    Returns a summary dict plus 'bets' (per-bet DataFrame) and, for kelly,
    'curve' (bankroll after each bet).
    """
    vol = (games["volume"].to_numpy(dtype=float)
           if "volume" in getattr(games, "columns", []) else None)
    q_home = np.asarray(model_prob, dtype=float)
    p_home = games["home_implied_win_prob"].to_numpy(dtype=float)
    y_home = games["home_win"].to_numpy(dtype=int)

    lo, hi = exclude_extreme
    tradable = (p_home >= lo) & (p_home <= hi)

    rows = []
    bank = bankroll
    curve = []
    for i in range(len(games)):
        if not tradable[i]:
            continue
        # edge on each side (model prob minus price)
        e_home = q_home[i] - p_home[i]
        e_away = (1.0 - q_home[i]) - (1.0 - p_home[i])   # == -e_home
        if max(e_home, e_away) <= edge_threshold:
            continue

        if e_home >= e_away:
            side, price, won, edge = "home", p_home[i], y_home[i] == 1, e_home
        else:
            side, price, won, edge = "away", 1.0 - p_home[i], y_home[i] == 0, e_away

        price = min(0.99, price + slippage)          # fill worse than the mark
        edge -= slippage

        if stake == "flat":
            contracts = 1.0
        elif stake == "kelly":
            f = kelly_fraction * (edge / (1.0 - price))      # Kelly share of bankroll
            f = max(0.0, min(f, kelly_cap))
            contracts = (f * bank) / price if price > 0 else 0.0
        else:
            raise ValueError(stake)
        if max_contracts is not None:
            contracts = min(contracts, max_contracts)
        if max_stake_frac is not None and vol is not None and vol[i] > 0:
            contracts = min(contracts, max_stake_frac * vol[i])
        if contracts <= 0:
            continue

        cost = contracts * price
        fee = kalshi_fee(price, contracts, fee_rate) if apply_fee else 0.0
        payoff = contracts * (1.0 if won else 0.0)
        pnl = payoff - cost - fee
        bank += pnl
        curve.append(bank)

        rows.append({"i": i, "side": side, "price": round(price, 3),
                     "model": round(q_home[i] if side == "home" else 1 - q_home[i], 3),
                     "edge": round(edge, 3), "contracts": round(contracts, 2),
                     "won": bool(won), "pnl": round(pnl, 3)})

    bets = pd.DataFrame(rows)
    if bets.empty:
        return {"n_bets": 0, "edge_threshold": edge_threshold, "stake": stake}

    staked = (bets["contracts"] * bets["price"]).sum()
    net = bets["pnl"].sum()
    out = {
        "stake": stake,
        "edge_threshold": edge_threshold,
        "n_bets": len(bets),
        "win_rate": round(bets["won"].mean(), 3),
        "total_staked": round(staked, 2),
        "net_pnl": round(net, 2),
        "roi": round(net / staked, 4) if staked else 0.0,
        "avg_edge": round(bets["edge"].mean(), 3),
        "bets": bets,
    }
    if stake == "kelly":
        curve = np.array(curve)
        peak = np.maximum.accumulate(curve)
        out["final_bankroll"] = round(float(curve[-1]), 2)
        out["return_pct"] = round(100.0 * (curve[-1] / bankroll - 1.0), 1)
        out["max_drawdown_pct"] = round(100.0 * float(((peak - curve) / peak).max()), 1)
        out["curve"] = curve
    return out


def threshold_sweep(games: pd.DataFrame, model_prob, thresholds, **kw) -> pd.DataFrame:
    """run_backtest at several edge thresholds; one row each."""
    keep = ["edge_threshold", "n_bets", "win_rate", "total_staked",
            "net_pnl", "roi", "avg_edge"]
    out = []
    for t in thresholds:
        r = run_backtest(games, model_prob, edge_threshold=t, **kw)
        out.append({k: r.get(k) for k in keep})
    return pd.DataFrame(out)
