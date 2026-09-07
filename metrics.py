"""Evaluation metrics for binary win-probability models.

Every model in this project predicts P(home team wins) for a set of games.
`evaluate` scores those probabilities against realized outcomes and prints the
numbers that matter, so different models land on one comparable scale.

Reference points (both computed for a 50/50 coin):
    Brier score 0.25, log loss 0.6931
A model that beats the market must beat the market's numbers on the SAME games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true, y_prob) -> float:
    """Mean squared error between probability and outcome. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true, y_prob, eps: float = 1e-12) -> float:
    """Negative log-likelihood per game. Lower is better.

    Probabilities are clipped into [eps, 1-eps] so a confident miss costs a
    large-but-finite amount instead of infinity.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into probability bins and compare predicted vs actual.

    A well-calibrated model has pred_mean ~= actual_rate in every row.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # bin index in [0, bins-1]; interior edges only so 0.0 and 1.0 land in-range
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})",
                "n": n,
                "pred_mean": float(y_prob[mask].mean()),
                "actual_rate": float(y_true[mask].mean()),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap"] = (df["pred_mean"] - df["actual_rate"]).abs()
    return df


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    """Sample-size-weighted average |pred_mean - actual_rate| across bins."""
    tbl = calibration_table(y_true, y_prob, bins=bins)
    if tbl.empty:
        return float("nan")
    return float(np.average(tbl["gap"], weights=tbl["n"]))


def evaluate(y_true, y_prob, name: str = "model", bins: int = 10,
             show_calibration: bool = True) -> dict:
    """Print Brier, log loss, accuracy and calibration; return them as a dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")

    n = len(y_true)
    bs = brier_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    acc = float(np.mean((y_prob >= 0.5) == (y_true == 1)))
    ece = expected_calibration_error(y_true, y_prob, bins=bins)
    base_rate = float(y_true.mean())

    print(f"=== {name} ===")
    print(f"games:          {n}")
    print(f"home-win rate:  {base_rate:.4f}")
    print(f"Brier score:    {bs:.4f}   (coin flip = 0.2500)")
    print(f"Log loss:       {ll:.4f}   (coin flip = 0.6931)")
    print(f"Accuracy @0.5:  {acc:.4f}")
    print(f"Calib. error:   {ece:.4f}   (weighted mean |pred - actual| over bins)")
    if show_calibration:
        tbl = calibration_table(y_true, y_prob, bins=bins)
        print("\ncalibration:")
        print(tbl.to_string(index=False))
    print()

    return {
        "name": name, "n": n, "brier": bs, "log_loss": ll,
        "accuracy": acc, "ece": ece, "base_rate": base_rate,
    }


def compare(results: list[dict]) -> pd.DataFrame:
    """Stack several `evaluate` result dicts into one sorted leaderboard."""
    df = pd.DataFrame(results)[["name", "n", "brier", "log_loss", "accuracy", "ece"]]
    return df.sort_values("log_loss").reset_index(drop=True)
