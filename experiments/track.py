"""Log the model-config history to MLflow.

Every combiner we tried and rejected becomes a run: params, walk-forward metrics
(Brier / log loss / accuracy / ECE on the held-out July + August games) and the
flat-stake betting ROI. This is the experiment log the project kept informally in
a conversation -- made queryable.

    python experiments/track.py                                   # log the sweep
    mlflow ui --backend-store-uri sqlite:///mlflow.db             # browse it
"""
from __future__ import annotations

import os

import mlflow
import numpy as np
import pandas as pd

from blend import Blender, logit
from metrics import brier_score, log_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, "wf_predictions.csv")
MONTH_ORDER = ["May", "June", "July", "Aug"]
TEST_MONTHS = ["July", "Aug"]
EDGE, FEE = 0.03, 0.07


def _ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    num = den = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            num += m.sum() * abs(p[m].mean() - y[m].mean()); den += m.sum()
    return num / den if den else float("nan")


def _flat_roi(y, p, price):
    """Bet the favoured side if |p - price| > EDGE; flat 1 unit; Kalshi fee."""
    y, p, price = np.asarray(y), np.asarray(p), np.asarray(price)
    edge = p - price
    take = (np.abs(edge) > EDGE) & (price > 0.05) & (price < 0.95)
    side_home = edge > 0
    px = np.where(side_home, price, 1 - price)[take]
    won = np.where(side_home, y == 1, y == 0)[take]
    fee = np.ceil(FEE * px * (1 - px) * 100) / 100
    pnl = np.where(won, 1 - px, -px) - fee
    return (pnl.sum() / px.sum() if px.sum() else 0.0, int(take.sum()),
            float(won.mean()) if take.sum() else float("nan"))


def evaluate(df, combiner_factory):
    """Walk-forward: fit the combiner on earlier months, score July then August."""
    ys, ps, prices = [], [], []
    for m in TEST_MONTHS:
        tr = df[df.month.isin(MONTH_ORDER[:MONTH_ORDER.index(m)])]
        te = df[df.month == m]
        bl = combiner_factory().fit(tr.p_std, tr.p_proj, tr.home_win)
        ys.append(te.home_win.to_numpy())
        ps.append(np.asarray(bl.predict(te.p_std, te.p_proj)))
        prices.append(te.market.to_numpy())
    y, p, price = np.concatenate(ys), np.concatenate(ps), np.concatenate(prices)
    roi, n_bets, win = _flat_roi(y, p, price)
    return {"brier": brier_score(y, p), "log_loss": log_loss(y, p),
            "accuracy": float(np.mean((p >= 0.5) == (y == 1))), "ece": _ece(y, p),
            "n_test_games": len(y), "n_bets": n_bets, "win_rate": win, "roi": roi}


# combiner factories -- each is one MLflow run
COMBINERS = {
    "season_only":        (dict(strategy="average", w_std=1.0),  lambda: Blender("average", w_std=1.0)),
    "projection_only":    (dict(strategy="average", w_std=0.0),  lambda: Blender("average", w_std=0.0)),
    "avg_50_50":          (dict(strategy="average", w_std=0.5),  lambda: Blender("average", w_std=0.5)),
    "avg_40_60_deployed": (dict(strategy="average", w_std=0.4),  lambda: Blender("average", w_std=0.4)),
    "avg_30_70":          (dict(strategy="average", w_std=0.3),  lambda: Blender("average", w_std=0.3)),
    "logistic_unreg":     (dict(strategy="logistic", C=1e6),     lambda: Blender("logistic", C=1e6)),
    "logistic_C0_35":     (dict(strategy="logistic", C=0.35),    lambda: Blender("logistic", C=0.35)),
    "platt_on_avg":       (dict(strategy="platt", C=0.35),       lambda: Blender("platt", C=0.35)),
}

# milestone model-input configs, metrics recorded from the walk-forward runs in
# the project history (not recomputed here -- the sims are expensive)
MILESTONES = [
    ("m0_team_only_season",  dict(rates="season", starter="none", bullpen="flat"),
     dict(brier=0.2536, accuracy=0.534, roi=0.019, n_bets=None)),
    ("m1_starter_ERA_scalar", dict(rates="season", starter="era_scalar", bullpen="real"),
     dict(brier=0.2406, accuracy=0.567, roi=0.104, n_bets=None)),
    ("m2_starter_components", dict(rates="season+proj", starter="components", bullpen="real"),
     dict(brier=0.2378, accuracy=0.581, roi=0.126, n_bets=None)),
    ("x_schedule_fatigue",   dict(feature="travel+stretch+DHnightcap", verdict="rejected"),
     dict(brier=0.2472, accuracy=0.545, roi=0.055, n_bets=None)),
]


def main():
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(ROOT, "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("baseball-winprob")
    df = pd.read_csv(WF)

    # market baseline
    with mlflow.start_run(run_name="baseline_market"):
        y = df[df.month.isin(TEST_MONTHS)]
        mlflow.set_tag("kind", "baseline")
        mlflow.log_metrics({"brier": brier_score(y.home_win, y.market),
                            "log_loss": log_loss(y.home_win, y.market),
                            "accuracy": float(np.mean((y.market >= 0.5) == (y.home_win == 1)))})

    for name, (params, factory) in COMBINERS.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("kind", "combiner")
            mlflow.log_params(params)
            mlflow.log_metrics(evaluate(df, factory))

    for name, params, metrics in MILESTONES:
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("kind", "model_input")
            mlflow.set_tag("source", "recorded_from_walkforward_history")
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

    runs = mlflow.search_runs(order_by=["metrics.brier ASC"])
    cols = [c for c in ["tags.mlflow.runName", "metrics.brier", "metrics.accuracy",
                        "metrics.roi", "metrics.n_bets"] if c in runs.columns]
    print("\n=== leaderboard (by Brier) ===")
    print(runs[cols].to_string(index=False))
    print("\nlogged to sqlite:///mlflow.db"
          "   ->   mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
