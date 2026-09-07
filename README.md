# MLB win-probability model vs. the Kalshi market

A Monte-Carlo simulator that predicts **P(home team wins)** for MLB games, compares
it to Kalshi's prediction-market implied probability, and bets the games where the
two disagree.

## Status: no demonstrated edge

| | ROI | win rate | closing-line-value beat rate |
|---|---|---|---|
| **Walk-forward backtest** (Jul–Aug 2026, 418 OOS bets) | +12.6% | 58% | — |
| **Live forward test** (~110 bets, Sep 2026) | ~0% | 33% | ~40% |

The walk-forward number had a bootstrap 90% CI of `[+4.7%, +20.6%]` (t ≈ 2.7), but
it was two test months with the model config chosen on that same data. The Brier
score always showed the model only *matching* the market, never beating it — the
"edge" was the model's disagreements landing right for eight weeks. The live
forward test erased it in ten days: breakeven P&L, sub-33% win rate, and a
closing-line-value beat rate stuck around 40%, meaning the market consistently
moves *against* the model's picks after it bets.

Kept as a complete, honest record of a well-built system that answered its
question in the negative.

## The model (`montecarlo.py`)

Per game, run the simulator twice — once on **season-to-date** rates, once on
**Marcel projections** (5/4/3-year blend + regression) — and average the two
P(home win) numbers 40/60.

Each run:

1. **Offense** — the 9 hitters in today's lineup, each hitter's own per-PA rates
   (1B/2B/3B/HR/BB + K/GO/AO mix), regressed toward league, slot-weighted.
2. **Pitching** — the probable starter's own component rates for innings 1–6
   (falls back to team pitching + an ERA scale for < 40 BF); the opposing bullpen's
   own `rp`-split rates for innings 7–9.
3. **Log5 matchup** — `rate = batter × pitcher / league` per event.
4. **Adjustments** — 30-park HR/hit factors, home field ×1.024/0.976, team platoon
   (vs-LHP/RHP × starter hand), reached-on-error.
5. **3 phases** — starter 1st time through (inn 1–3), starter later (inn 4–6,
   offense ×1.03), bullpen (inn 7–9).
6. **Inning sim** — vectorised PA random walk with probabilistic base-running,
   GIDP, sac flies, steals (league constants), ghost-runner extra innings.
7. **P(home win)** — convolve phase run distributions, then
   `P(home > away) + P(tie)·P(home wins the extra frame)`.

Tried and measured neutral-to-negative out of sample, not used: starter days-rest
penalty, schedule fatigue (travel / long stretch / doubleheader), a learned
logistic/isotonic combiner.

## Files

| file | role |
|---|---|
| `montecarlo.py` | the model — data fetchers + Log5 + simulator + `predict_games()` |
| `blend.py` | combine the two sims (default: 0.40·season + 0.60·projection) |
| `metrics.py` | Brier, log loss, accuracy, calibration error, `compare()` |
| `backtest.py` | betting sim vs Kalshi prices — edge threshold, flat/Kelly stake, fees, fill model |
| `build_odds.py` | pull MLB results + Kalshi implied win probs into the odds CSVs (any date range) |
| `gen_predictions.py` | run both sims per month with the correct as-of cutoff → `wf_predictions.csv` |
| `walkforward.py` | expanding-window evaluation; writes `blender.pkl` |
| `forward.py` | live loop — `settle()` finished games, `snapshot()` upcoming ones → `forward_log.csv` |
| `picks.py` | read `forward_log.csv`, print the day's bet slip |
| `run_forward.bat` | wrapper for the Windows scheduled task (edit the Python path) |
| `base_line.ipynb` | the original exploration notebook (XGBoost attempt, Markov sim, odds builder) |

## Platform layer

The model is the same; this wraps it in the shape a data/ML team would run.

| path | role |
|---|---|
| `warehouse/` | **DuckDB** warehouse. `db.py` lands the CSVs; `transforms/*.sql` build `stg_games`, `stg_predictions`, then the `mart_*` tables (bet simulation, decile calibration, Brier/log-loss/ECE, rolling P&L / CLV — all in SQL with CTEs + window functions). |
| `flows/pipeline.py` | **Prefect** flow that replaces the scheduled task: `settle_and_snapshot` + `ingest_recent_odds` → `build_warehouse` → `refresh_dashboard_data`, plus a weekly `track_experiments`. Deployable on a cron schedule. |
| `experiments/track.py` | **MLflow** run log. Every combiner tried and rejected (season-only, projection-only, 50/50, the deployed 40/60, the over-fit logistic/isotonic) plus milestone model-input configs, each a run with params + walk-forward metrics. SQLite backing store. |
| `serving/app.py` | **FastAPI** service around `predict_games()`. Loads a pickled rate cache at startup; `POST /predict` with two team names (optional lineup / starter-ERA overrides) returns P(home win). `GET /model` is the model card. |
| `serving/build_cache.py` | fetch the rate tables once → `data/feature_cache.pkl` for the container. |
| `dashboard/app.py` | **Streamlit + Plotly** dashboard on the warehouse marts: calibration curve (model vs market), live P&L with 30-bet rolling closing-line-value, the scoreboard. |
| `docker/` | `Dockerfile` (API), `Dockerfile.dashboard`, `docker-compose.yml` (API + MLflow UI + dashboard). |
| `pyproject.toml` | editable install (`pip install -e .`), `[project.optional-dependencies] platform`. |

```bash
pip install -e ".[platform]"

python warehouse/db.py               # land CSVs + run SQL transforms
python experiments/track.py          # log the config sweep to MLflow
python serving/build_cache.py        # fetch the rate cache
uvicorn serving.app:app              # http://localhost:8000/docs
streamlit run dashboard/app.py       # http://localhost:8501
python flows/pipeline.py             # one orchestrated run of the above
mlflow ui --backend-store-uri sqlite:///mlflow.db

# or the whole stack:
docker compose -f docker/docker-compose.yml up --build
```

## Data files

| file | contents |
|---|---|
| `MLB_2026_*_Odds_And_Results.csv` | game results + Kalshi implied win prob ~1 h pre-game, by month |
| `wf_predictions.csv` | per-game `p_std` / `p_proj` for the walk-forward months |
| `forward_log.csv` | live forward-test log: model prob, entry price, result, closing line, P&L, CLV |
| `blender.pkl` | the fitted combiner used by `forward.py` |

## Running it

```bash
pip install -r requirements.txt
```

Model + backtest run on the free MLB statsapi alone. The odds builder and the
forward test need a **Kalshi API key**:

```bash
setx KALSHI_KEY_ID "your-key-id"
# private key file at %USERPROFILE%\.kalshi\kalshi_private_key.pem
# (or set KALSHI_PRIVATE_KEY_PATH)
```

Then:

```bash
python gen_predictions.py     # ~20 min: two sims/month → wf_predictions.csv
python walkforward.py         # evaluate, write blender.pkl
python forward.py             # log today's games + settle finished ones
python picks.py --date 2026-09-05 --edge 0.04 --unit 10
```

## Data source

MLB Stats API (`statsapi.mlb.com`) for schedules, results, rosters, lineups,
player and team stats. Kalshi (`external-api.kalshi.com`) for market prices.
