"""Read forward_log.csv and print today's recommended bets.

    python picks.py                 # edge >= 0.03, 10 contracts/bet, today+tomorrow
    python picks.py --edge 0.04 --unit 20 --date 2026-08-30

Nothing is placed automatically -- this prints a slip; you enter it on Kalshi.
Stakes are FLAT (what the walk-forward tested). No Kelly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math

import pandas as pd

LOG = "forward_log.csv"
FEE_RATE = 0.07
PRICE_BAND = (0.05, 0.95)


def fee(price, n):
    return math.ceil(FEE_RATE * n * price * (1 - price) * 100) / 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.03, help="min |model - price|")
    ap.add_argument("--unit", type=int, default=10, help="contracts per bet (flat)")
    ap.add_argument("--date", default=None, help="only this game_date (default: all open)")
    ap.add_argument("--max-bets", type=int, default=12, help="cap number of bets shown")
    a = ap.parse_args()

    log = pd.read_csv(LOG)
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date().isoformat()
    df = log[(log.status == "open") & log.entry_p_home.notna()].copy()
    df["start"] = pd.to_datetime(df.get("start_utc"), utc=True, errors="coerce")
    # keep only games confirmed to start in the future; a today-or-earlier game with
    # no start time on file couldn't be refreshed -> it has already begun
    df = df[(df.start > now) | (df.start.isna() & (df.game_date > today))]
    if a.date:
        df = df[df.game_date == a.date]
    if df.empty:
        print("no upcoming open games with a price"); return

    rows = []
    for r in df.itertuples():
        edge_home = r.model_p_home - r.entry_p_home
        if abs(edge_home) < a.edge:
            continue
        if r.entry_p_home < PRICE_BAND[0] or r.entry_p_home > PRICE_BAND[1]:
            continue
        mins = int((r.start - now).total_seconds() // 60) if pd.notna(r.start) else None
        if edge_home > 0:
            side, team, price, model = "HOME", r.home_team, r.entry_p_home, r.model_p_home
        else:
            side, team, price, model = "AWAY", r.away_team, 1 - r.entry_p_home, 1 - r.model_p_home
        n = a.unit
        cost = n * price
        f = fee(price, n)
        ev = n * (model * 1.0 - price) - f          # expected profit, $
        rows.append({
            "starts": f"{mins//60}h{mins%60:02d}m" if mins is not None else "?",
            "game": f"{r.away_team} @ {r.home_team}",
            "bet": f"{side} {team}", "price": round(price, 2), "model": round(model, 3),
            "edge": round(abs(edge_home), 3), "n": n, "cost$": round(cost, 2),
            "fee$": round(f, 2), "EV$": round(ev, 2),
            "lineup": "Y" if r.lineup_used else "pre",
        })

    if not rows:
        print(f"no bets clear edge >= {a.edge:.2f}"); return
    out = pd.DataFrame(rows).sort_values("edge", ascending=False).head(a.max_bets)

    print(f"\nBET SLIP  ({dt.datetime.now():%Y-%m-%d %H:%M})   edge >= {a.edge:.2f}   {a.unit} contracts/bet\n")
    print(out.to_string(index=False))
    print(f"\n{len(out)} bets   staked ${out['cost$'].sum():.2f}   fees ${out['fee$'].sum():.2f}"
          f"   model EV ${out['EV$'].sum():+.2f}")
    if (out.lineup == "pre").any():
        print("note: 'pre' = lineup not posted yet; re-run after ~3pm local for those.")


if __name__ == "__main__":
    main()
