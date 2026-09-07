"""Generate walk-forward predictions for the blend/calibration step.

For each month, run the sim twice with inputs cut off at that month's start:
  p_std  = season-to-date rates
  p_proj = Marcel-projected rates (current window + 2 prior seasons, regressed)
Both use realism + platoon + a real bullpen matchup for innings 7-9.

Writes wf_predictions.csv: month, game_date, away_team, home_team, home_win,
market, p_std, p_proj  -- so blend.py / walkforward.py never re-sim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import montecarlo as mc

SEASON_START = "2026-03-01"
N = 40_000

ODDS_FILES = [
    "MLB_2026_First_Half_Odds_And_Results.csv",
    "MLB_2026_June_Odds_And_Results.csv",
    "MLB_2026_July_Odds_And_Results.csv",
    "MLB_2026_August_Odds_And_Results.csv",
]
COLS = ["game_date", "away_team", "home_team", "away_score", "home_score",
        "away_implied_win_prob", "home_implied_win_prob"]

MONTHS = {  # name: (cutoff, start, end)
    "May":  ("2026-05-01", "2026-05-01", "2026-05-31"),
    "June": ("2026-06-01", "2026-06-01", "2026-06-30"),
    "July": ("2026-07-01", "2026-07-01", "2026-07-31"),
    "Aug":  ("2026-08-01", "2026-08-01", "2026-08-27"),
}


def load_odds() -> pd.DataFrame:
    frames = []
    for f in ODDS_FILES:
        try:
            frames.append(pd.read_csv(f)[COLS])
        except FileNotFoundError:
            pass
    o = pd.concat(frames, ignore_index=True)
    o = o.drop_duplicates(subset=["game_date", "away_team", "home_team"], keep="last")
    o["home_win"] = (o["home_score"] > o["away_score"]).astype(int)
    return o


def main():
    odds = load_odds()
    rows = []
    for name, (cut, s, e) in MONTHS.items():
        test = odds[(odds.game_date >= s) & (odds.game_date <= e)].reset_index(drop=True)
        if test.empty:
            print(f"{name}: no games"); continue
        print(f"{name}: {len(test)} games, cutoff {cut}", flush=True)

        tab_s = mc.fetch_team_tables(2026, SEASON_START, cut)
        tab_p = mc.fetch_team_tables(2026, SEASON_START, cut, project=True)
        pr_s = mc.fetch_player_hit_rates(2026, shrink_pa=100, start_date=SEASON_START, end_date=cut)
        pr_p = mc.fetch_player_hit_rates(2026, start_date=SEASON_START, end_date=cut, project=True)
        bp_s = mc.fetch_team_bullpen(2026, SEASON_START, cut)
        bp_p = mc.fetch_team_bullpen(2026, SEASON_START, cut, project=True)
        sr_s = mc.fetch_starter_rates(s, e, 2026, rate_start=SEASON_START, rate_end=cut)
        sr_p = mc.fetch_starter_rates(s, e, 2026, rate_start=SEASON_START, rate_end=cut, project=True)
        plt = mc.fetch_team_platoon(2026, SEASON_START, cut)
        st = mc.fetch_probable_starter_eras(s, e, 2026, era_start=SEASON_START, era_end=cut, hands=True)
        lu = mc.fetch_lineups(s, e)

        common = dict(lineups=lu, starters=st, starter_strength=0.30,
                      platoon=plt, realism=True, n=N, seed=7)
        p_std = mc.predict_games(test, tab_s, player_rates=pr_s, bullpen_rates=bp_s,
                                 starter_rates=sr_s, **common)
        p_proj = mc.predict_games(test, tab_p, player_rates=pr_p, bullpen_rates=bp_p,
                                  starter_rates=sr_p, **common)

        for j in range(len(test)):
            r = test.iloc[j]
            rows.append({"month": name, "game_date": r.game_date,
                         "away_team": r.away_team, "home_team": r.home_team,
                         "home_win": int(r.home_win),
                         "market": float(r.home_implied_win_prob),
                         "p_std": float(p_std.iloc[j]), "p_proj": float(p_proj.iloc[j])})
        print(f"  done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("wf_predictions.csv", index=False)
    print(f"\nsaved {len(df)} rows -> wf_predictions.csv")
    for m in df.month.unique():
        d = df[df.month == m]
        b = lambda c: np.mean((d[c] - d.home_win) ** 2)
        print(f"  {m}: n={len(d)}  Brier  mkt {b('market'):.4f}  std {b('p_std'):.4f}  proj {b('p_proj'):.4f}")


if __name__ == "__main__":
    main()
