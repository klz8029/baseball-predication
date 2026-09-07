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
