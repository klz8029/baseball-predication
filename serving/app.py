"""FastAPI wrapper around predict_games().

    uvicorn serving.app:app --reload
    curl -s localhost:8000/predict -H 'content-type: application/json' \
         -d '{"away_team":"Chicago Cubs","home_team":"Milwaukee Brewers"}'

Loads the pickled rate cache at startup (serving/build_cache.py builds it). Given
away/home teams it runs the 0.40/0.60 blended Monte-Carlo sim and returns
P(home win). Optional overrides: today's lineups, and each starter's ERA.
"""
from __future__ import annotations

import os
import pickle
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import montecarlo as mc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "feature_cache.pkl")
MODEL_VERSION = "mc-blend-0.40/0.60+SPcomponents"
STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not os.path.exists(CACHE):
        from serving.build_cache import build
        build()
    with open(CACHE, "rb") as f:
        STATE["cache"] = pickle.load(f)
    yield
    STATE.clear()


app = FastAPI(title="Baseball win-probability model", version="0.2.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    away_team: str
    home_team: str
    away_lineup: list[int] | None = Field(None, description="9 MLBAM hitter ids")
    home_lineup: list[int] | None = None
    away_starter_era: float | None = None
    home_starter_era: float | None = None
    n: int = Field(40_000, ge=2_000, le=200_000)


class PredictResponse(BaseModel):
    p_home_win: float
    p_away_win: float
    model_version: str
    cache_built_at: str


@app.get("/health")
def health():
    c = STATE.get("cache")
    return {"status": "ok" if c else "loading",
            "cache_built_at": c["built_at"] if c else None, "model_version": MODEL_VERSION}


@app.get("/model")
def model_card():
    return {
        "version": MODEL_VERSION,
        "sim": "Log5 matchup -> 3-phase vectorised inning simulator (starter TTO1/TTO2, "
               "real bullpen), probabilistic base-running, ghost-runner extras",
        "combiner": "0.40 * season-to-date sim + 0.60 * Marcel-projection sim",
        "adjustments": ["30-park HR/hit factors", "home field x1.024/0.976",
                        "team platoon vs starter hand", "reached-on-error"],
        "walk_forward_roi": "+12.6% (Jul-Aug 2026, 418 OOS bets, t~2.7)",
        "forward_test": "breakeven; closing-line-value beat rate ~40% -> no live edge",
    }


def _one_game(req: PredictRequest) -> float:
    c = STATE["cache"]
    games = pd.DataFrame([{"game_date": "2026-01-01",
                           "away_team": req.away_team, "home_team": req.home_team}])
    lineups = starters = None
    key = ("2026-01-01", req.away_team, req.home_team)
    if req.away_lineup and req.home_lineup:
        lineups = {key: {"away": req.away_lineup, "home": req.home_lineup}}
    if req.away_starter_era or req.home_starter_era:
        starters = {key: {"away_era": req.away_starter_era, "home_era": req.home_starter_era,
                          "away_hand": None, "home_hand": None}}

    def run(tables, player, bullpen):
        return float(mc.predict_games(
            games, tables, lineups=lineups,
            player_rates=player if lineups else None,
            starters=starters, bullpen_rates=bullpen, platoon=c["platoon"],
            realism=True, n=req.n, seed=7).iloc[0])

    p_std = run(c["tables_std"], c["player_std"], c["bullpen_std"])
    p_proj = run(c["tables_proj"], c["player_proj"], c["bullpen_proj"])
    return 0.40 * p_std + 0.60 * p_proj


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    c = STATE.get("cache")
    if not c:
        raise HTTPException(503, "model cache not loaded")
    for t in (req.away_team, req.home_team):
        if t not in c["tables_std"]["bat"]:
            raise HTTPException(422, f"unknown team: {t!r}")
    p = _one_game(req)
    return PredictResponse(p_home_win=round(p, 4), p_away_win=round(1 - p, 4),
                           model_version=MODEL_VERSION, cache_built_at=c["built_at"])
