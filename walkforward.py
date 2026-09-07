"""Expanding-window walk-forward evaluation of the blended model.

Uses wf_predictions.csv (from gen_predictions.py): per game, p_std / p_proj were
each produced with inputs cut off at that game's month start. For each test month
we fit the Blender on all earlier months, then score + backtest that month.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import backtest as bt
from blend import Blender
from metrics import brier_score, log_loss

MONTH_ORDER = ["May", "June", "July", "Aug"]
TEST_MONTHS = ["July", "Aug"]          # earlier months are training-only
EDGE = 0.02


def _stats(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return (round(brier_score(y, p), 4), round(log_loss(y, p), 4),
            round(float(np.mean((p >= 0.5) == (y == 1))), 3))


def main():
    df = pd.read_csv("wf_predictions.csv")
    df["home_implied_win_prob"] = df["market"]

    per_month, test_bets = [], []
    for m in TEST_MONTHS:
        train = df[df.month.isin(MONTH_ORDER[:MONTH_ORDER.index(m)])]
        test = df[df.month == m].reset_index(drop=True)
        if train.empty or test.empty:
            continue

        bl = Blender(mode="average").fit(train.p_std, train.p_proj, train.home_win)
        p_blend = bl.predict(test.p_std, test.p_proj)
        p_avg = 0.5 * (test.p_std.to_numpy() + test.p_proj.to_numpy())

        row = {"month": m, "n": len(test), "train_n": len(train),
               "combiner": bl.coef.get("mode", "?")}
        for tag, p in [("mkt", test.market), ("std", test.p_std), ("proj", test.p_proj),
                       ("avg", p_avg), ("blend", p_blend)]:
            b, ll, a = _stats(test.home_win, p)
            row[f"{tag}_brier"] = b
        per_month.append(row)

        for tag, p in [("std", test.p_std.to_numpy()), ("blend", np.asarray(p_blend))]:
            r = bt.run_backtest(test, pd.Series(p), edge_threshold=EDGE,
                                stake="flat", apply_fee=True)
            r0 = bt.run_backtest(test, pd.Series(p), edge_threshold=EDGE,
                                 stake="flat", apply_fee=False)
            row[f"{tag}_bets"] = r.get("n_bets", 0)
            row[f"{tag}_roi"] = round(r.get("roi", np.nan), 3)
            row[f"{tag}_roi_nofee"] = round(r0.get("roi", np.nan), 3)
            if tag == "blend" and r.get("n_bets", 0):
                b = r["bets"].copy(); b["month"] = m
                test_bets.append(b)

    summ = pd.DataFrame(per_month)
    print("=== per test month (Brier, and betting on the blended prob) ===")
    print(summ.to_string(index=False))

    bets = pd.concat(test_bets, ignore_index=True)
    staked = (bets.contracts * bets.price).to_numpy()
    pnl = bets.pnl.to_numpy()
    rng = np.random.default_rng(0)
    boot = np.sort([pnl[i].sum() / staked[i].sum()
                    for i in (rng.integers(0, len(pnl), len(pnl)) for _ in range(5000))])
    print(f"\n=== POOLED test bets (blend, edge {EDGE}, fees on) ===")
    print(f"{len(bets)} bets  net ${pnl.sum():.2f}  ROI {pnl.sum() / staked.sum():+.3f}  "
          f"win {bets.won.mean():.3f}")
    print(f"bootstrap 90% CI [{boot[250]:+.3f}, {boot[4750]:+.3f}]  P(ROI>0)={ (boot>0).mean():.2f}  "
          f"per-bet ${pnl.mean():+.3f} +/- ${pnl.std():.3f}")

    # deploy blender: fit on everything for forward.py
    full = Blender(mode="average").fit(df.p_std, df.p_proj, df.home_win)
    with open("blender.pkl", "wb") as f:
        pickle.dump(full, f)
    print(f"\nsaved blender.pkl  (fit on all {len(df)} games)  coef {full.coef}")


if __name__ == "__main__":
    main()
