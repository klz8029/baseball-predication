"""Fetch the model's rate tables once and pickle them for the serving container.

The full fetch (team + player + bullpen + platoon + projections) takes ~40 s and
hits statsapi ~20 times. The API service loads this cache at startup instead.
Run nightly (the Prefect flow does) so the served model stays current.
"""
from __future__ import annotations

import datetime as dt
import os
import pickle

import montecarlo as mc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "feature_cache.pkl")
SEASON = 2026


def build(season: int = SEASON) -> dict:
    print("fetching rate tables ...")
    cache = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "tables_std": mc.fetch_team_tables(season),
        "tables_proj": mc.fetch_team_tables(season, project=True),
        "player_std": mc.fetch_player_hit_rates(season, shrink_pa=100),
        "player_proj": mc.fetch_player_hit_rates(season, project=True),
        "bullpen_std": mc.fetch_team_bullpen(season),
        "bullpen_proj": mc.fetch_team_bullpen(season, project=True),
        "platoon": mc.fetch_team_platoon(season),
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"wrote {CACHE}  ({os.path.getsize(CACHE) // 1024} KB, built_at {cache['built_at']})")
    return cache


if __name__ == "__main__":
    build()
