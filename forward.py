"""Forward test: snapshot upcoming games, settle finished ones, track CLV.

Run daily (cron). Each run:
  1. settle()   -- fill result + closing line + P&L + CLV for games now final
  2. snapshot() -- add today..+2d games with the model prob and the current
                   Kalshi price, status='open'

forward_log.csv is the append-only record. Because the model inputs are whatever
is known at snapshot time, this is a genuine forward test -- no backtest can
substitute for it once Kalshi's ~6-week candle history rolls off.
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

import montecarlo as mc
from build_odds import KALSHI_API_BASE, SERIES, TEAM_TICKERS, kalshi_headers, mlb_games

LOG = "forward_log.csv"
EDGE = 0.02
FEE_RATE = 0.07
SEASON = 2026
COLUMNS = ["snapshot_ts", "game_date", "start_utc", "away_team", "home_team",
           "away_starter", "home_starter", "lineup_used",
           "model_p_home", "entry_p_home", "status",
           "close_p_home", "home_win", "bet_side", "bet_edge", "pnl_1c", "clv"]


_FILL_COLS = ("close_p_home", "home_win", "bet_side", "bet_edge", "pnl_1c", "clv")


def _load_log() -> pd.DataFrame:
    if os.path.exists(LOG):
        df = pd.read_csv(LOG)
        for c in _FILL_COLS:                       # accept both numbers and blanks
            if c in df.columns:
                df[c] = df[c].astype("object")
        return df
    return pd.DataFrame(columns=COLUMNS)


def _kalshi_price(ticker: str, lo_ts: int, hi_ts: int):
    path = f"/trade-api/v2/series/{SERIES}/markets/{ticker}/candlesticks"
    r = requests.get(f"{KALSHI_API_BASE}{path}", headers=kalshi_headers("GET", path),
                     params={"period_interval": 60, "start_ts": lo_ts, "end_ts": hi_ts},
                     timeout=30)
    if r.status_code != 200:
        return None
    for c in reversed(r.json().get("candlesticks", [])):
        for k in ("close_dollars", "mean_dollars", "previous_dollars"):
            v = c.get("price", {}).get(k)
            if v is not None and float(v) > 0:
                return float(v)
    return None


def _ticker_for(game: dict, when: dt.datetime):
    d = when.strftime("%y%b%d").upper()
    aw = TEAM_TICKERS.get(game["away_team"]); hm = TEAM_TICKERS.get(game["home_team"])
    if not aw or not hm:
        return None
    idx = requests.get(f"{KALSHI_API_BASE}/trade-api/v2/markets?series_ticker={SERIES}"
                       f"&limit=1000", headers=kalshi_headers(
                           "GET", f"/trade-api/v2/markets"), timeout=30)
    if idx.status_code != 200:
        return None
    pre = f"KXMLBGAME-{d}"
    for m in idx.json().get("markets", []):
        t = m["ticker"]
        if t.startswith(pre) and aw in t and hm in t and t.endswith(f"-{aw}"):
            return t
    return None


def _model_prob(games_df: pd.DataFrame) -> pd.Series:
    """Season-to-date + projected sims, blended with blender.pkl if present."""
    tab_s = mc.fetch_team_tables(SEASON)
    tab_p = mc.fetch_team_tables(SEASON, project=True)
    pr_s = mc.fetch_player_hit_rates(SEASON, shrink_pa=100)
    pr_p = mc.fetch_player_hit_rates(SEASON, project=True)
    bp_s = mc.fetch_team_bullpen(SEASON)
    bp_p = mc.fetch_team_bullpen(SEASON, project=True)
    plt = mc.fetch_team_platoon(SEASON)
    d0, d1 = games_df.game_date.min(), games_df.game_date.max()
    st = mc.fetch_probable_starter_eras(d0, d1, SEASON, hands=True)
    sr_s = mc.fetch_starter_rates(d0, d1, SEASON)
    sr_p = mc.fetch_starter_rates(d0, d1, SEASON, project=True)
    lu = mc.fetch_lineups(d0, d1)

    common = dict(lineups=lu, starters=st, starter_strength=0.30, platoon=plt,
                  realism=True, n=40_000, seed=7)
    p_std = mc.predict_games(games_df, tab_s, player_rates=pr_s, bullpen_rates=bp_s,
                             starter_rates=sr_s, **common)
    p_proj = mc.predict_games(games_df, tab_p, player_rates=pr_p, bullpen_rates=bp_p,
                              starter_rates=sr_p, **common)
    if os.path.exists("blender.pkl"):
        with open("blender.pkl", "rb") as f:
            return pd.Series(pickle.load(f).predict(p_std, p_proj), index=games_df.index)
    return 0.5 * (p_std + p_proj)


def snapshot():
    today = dt.date.today()
    end = today + timedelta(days=2)
    games = [g for g in mlb_games(str(today), str(end))]  # mlb_games only keeps finals
    # mlb_games filters to finals; for upcoming we need the raw schedule:
    sc = requests.get("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
                      f"&startDate={today}&endDate={end}&hydrate=probablePitcher,lineups"
                      "&gameType=R", timeout=30).json()
    rows = []
    for day in sc.get("dates", []):
        for g in day["games"]:
            if g.get("status", {}).get("statusCode") == "F":
                continue
            a, h = g["teams"]["away"]["team"]["name"], g["teams"]["home"]["team"]["name"]
            if a not in TEAM_TICKERS or h not in TEAM_TICKERS:
                continue
            lu = g.get("lineups", {})
            rows.append({
                "game_date": g["gameDate"][:10], "away_team": a, "home_team": h,
                "start_utc": dt.datetime.strptime(g["gameDate"], "%Y-%m-%dT%H:%M:%SZ"),
                "away_starter": (g["teams"]["away"].get("probablePitcher") or {}).get("fullName"),
                "home_starter": (g["teams"]["home"].get("probablePitcher") or {}).get("fullName"),
                "lineup_used": len(lu.get("awayPlayers", [])) == 9 and len(lu.get("homePlayers", [])) == 9,
            })
    if not rows:
        print("snapshot: no upcoming games"); return
    gdf = pd.DataFrame(rows).drop_duplicates(
        subset=["game_date", "away_team", "home_team"], keep="first").reset_index(drop=True)

    log = _load_log()
    now = dt.datetime.now(dt.timezone.utc)

    # backfill start times onto any open row missing one (e.g. rows written before
    # the column existed), so picks.py can filter out games already underway
    start_by_key = {(r.game_date, r.away_team, r.home_team): r.start_utc.isoformat()
                    for _, r in gdf.iterrows()}
    if "start_utc" not in log.columns:
        log["start_utc"] = np.nan
    for i, lr in log.iterrows():
        if lr.status == "open" and pd.isna(lr.get("start_utc")):
            s = start_by_key.get((lr.game_date, lr.away_team, lr.home_team))
            if s:
                log.at[i, "start_utc"] = s

    # only settled/started rows are frozen; every other open row is refreshed each
    # run (price + prediction + lineup) so the last pre-game snapshot is the entry
    frozen = {(lr.game_date, lr.away_team, lr.home_team)
              for _, lr in log.iterrows() if lr.status != "open"}
    open_keys = {(lr.game_date, lr.away_team, lr.home_team)
                 for _, lr in log[log.status == "open"].iterrows()}

    todo = gdf[gdf.apply(lambda r: (r.game_date, r.away_team, r.home_team) not in frozen
                         and r.start_utc.replace(tzinfo=dt.timezone.utc) > now, axis=1)
               ].reset_index(drop=True)
    if todo.empty:
        print("snapshot: nothing to add or refresh"); return

    probs = _model_prob(todo)
    log_idx = {(lr.game_date, lr.away_team, lr.home_team): i for i, lr in log.iterrows()}
    added = refreshed = 0
    new_rows = []
    for (_, r), mp in zip(todo.iterrows(), probs):
        key = (r.game_date, r.away_team, r.home_team)
        tk = _ticker_for(r.to_dict(), r.start_utc)
        entry = _kalshi_price(tk, int(now.timestamp()) - 4 * 3600, int(now.timestamp())) if tk else None
        entry_home = round(1.0 - entry, 3) if entry is not None else np.nan
        fields = dict(snapshot_ts=now.isoformat(timespec="seconds"),
                      start_utc=r.start_utc.isoformat(),
                      away_starter=r.away_starter, home_starter=r.home_starter,
                      lineup_used=bool(r.lineup_used), model_p_home=round(float(mp), 4),
                      entry_p_home=entry_home)
        if key in open_keys:                       # refresh the existing pre-lineup row
            for k, v in fields.items():
                log.at[log_idx[key], k] = v
            refreshed += 1
        else:                                      # brand-new game
            new_rows.append({**fields, "game_date": r.game_date, "away_team": r.away_team,
                             "home_team": r.home_team, "status": "open",
                             "close_p_home": np.nan, "home_win": np.nan, "bet_side": "",
                             "bet_edge": np.nan, "pnl_1c": np.nan, "clv": np.nan})
            added += 1
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    log.to_csv(LOG, index=False)
    print(f"snapshot: {added} added, {refreshed} refreshed")


def settle():
    log = _load_log()
    if log.empty:
        return
    openrows = log[log.status == "open"]
    if openrows.empty:
        print("settle: nothing open"); return

    dmin, dmax = openrows.game_date.min(), openrows.game_date.max()
    finals = {(g["game_date"], g["away_team"], g["home_team"]): g
              for g in mlb_games(dmin, dmax)}

    n = 0
    for i, r in openrows.iterrows():
        key = (r.game_date, r.away_team, r.home_team)
        g = finals.get(key)
        if not g:
            continue
        home_win = int(g["home_score"] > g["away_score"])
        start = g["start_time_utc"]
        tk = _ticker_for(r.to_dict(), start)
        close = None
        if tk:
            close = _kalshi_price(tk, int(start.timestamp()) - 3600, int(start.timestamp()) + 60)
        close_home = (1.0 - close) if close is not None else np.nan

        entry_home = r.entry_p_home
        bet_side, bet_edge, pnl, clv = "", np.nan, np.nan, np.nan
        if not np.isnan(entry_home):
            e_home = r.model_p_home - entry_home
            if abs(e_home) > EDGE:
                if e_home > 0:
                    bet_side, px, won = "home", entry_home, home_win == 1
                    close_px = close_home
                else:
                    bet_side, px, won = "away", 1 - entry_home, home_win == 0
                    close_px = (1 - close_home) if not np.isnan(close_home) else np.nan
                bet_edge = abs(e_home)
                fee = np.ceil(FEE_RATE * px * (1 - px) * 100) / 100
                pnl = round((1.0 - px if won else -px) - fee, 3)
                clv = round(float(close_px - px), 3) if not np.isnan(close_px) else np.nan

        log.at[i, "status"] = "settled"
        log.at[i, "home_win"] = home_win
        log.at[i, "bet_side"] = bet_side
        log.at[i, "close_p_home"] = round(close_home, 3) if not np.isnan(close_home) else ""
        log.at[i, "bet_edge"] = round(bet_edge, 3) if not np.isnan(bet_edge) else ""
        log.at[i, "pnl_1c"] = pnl if not np.isnan(pnl) else ""
        log.at[i, "clv"] = clv if not np.isnan(clv) else ""
        n += 1
    log.to_csv(LOG, index=False)
    print(f"settle: settled {n} games")

    s = log[(log.status == "settled") & (log.bet_side.astype(str) != "")]
    if len(s):
        roi = s.pnl_1c.sum() / s.apply(
            lambda r: r.entry_p_home if r.bet_side == "home" else 1 - r.entry_p_home,
            axis=1).sum()
        clv = s.clv.dropna()
        print(f"  bets settled: {len(s)}  win {s.pnl_1c.gt(0).mean():.3f}  "
              f"net ${s.pnl_1c.sum():.2f}  ROI {roi:+.3f}")
        if len(clv):
            print(f"  CLV: mean {clv.mean():+.3f}  beat-close rate {clv.gt(0).mean():.3f}  (n={len(clv)})")


if __name__ == "__main__":
    settle()
    snapshot()
