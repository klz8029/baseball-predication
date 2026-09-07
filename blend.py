"""Combine the two sim probabilities into one.

mode="average"  -- fixed 0.5/0.5 of p_std and p_proj. No fitting.
mode="platt"    -- average, then a 1-parameter regularized Platt recalibration.
mode="logistic" -- learn a weight on each sim (regularized logistic on the logits).

Default is "average": on the walk-forward data (May-Aug, ~500-760 training games,
one of them pathological) every fitted combiner generalized *worse* than the plain
average. Revisit once forward.py has accumulated a few months of live games.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class Blender:
    def __init__(self, mode: str = "average", w_std: float = 0.40, C: float = 0.35):
        """mode="average" -> fixed `w_std`*p_std + (1-w_std)*p_proj (no fitting).
        On the walk-forward data the projection is the stronger single sim, so the
        default leans to it: 0.40 / 0.60."""
        if mode not in ("average", "platt", "logistic"):
            raise ValueError(mode)
        self.mode = mode
        self.w_std = w_std
        self.C = C
        self.lr: LogisticRegression | None = None

    def _avg(self, p_std, p_proj) -> np.ndarray:
        return (self.w_std * np.asarray(p_std, dtype=float)
                + (1.0 - self.w_std) * np.asarray(p_proj, dtype=float))

    def fit(self, p_std, p_proj, y) -> "Blender":
        y = np.asarray(y, dtype=int)
        if self.mode == "average":
            return self
        if self.mode == "platt":
            X = logit(self._avg(p_std, p_proj)).reshape(-1, 1)
        else:  # logistic
            X = np.column_stack([logit(p_std), logit(p_proj)])
        self.lr = LogisticRegression(C=self.C, solver="lbfgs").fit(X, y)
        return self

    def predict(self, p_std, p_proj) -> np.ndarray:
        avg = self._avg(p_std, p_proj)
        if self.mode == "average" or self.lr is None:
            return avg
        if self.mode == "platt":
            return self.lr.predict_proba(logit(avg).reshape(-1, 1))[:, 1]
        X = np.column_stack([logit(p_std), logit(p_proj)])
        return self.lr.predict_proba(X)[:, 1]

    @property
    def coef(self) -> dict:
        if self.lr is None:
            return {"mode": self.mode}
        c = self.lr.coef_[0]
        out = {"mode": self.mode, "intercept": float(self.lr.intercept_[0])}
        if self.mode == "logistic":
            out.update(w_std=float(c[0]), w_proj=float(c[1]),
                       std_share=float(c[0] / (c[0] + c[1])) if (c[0] + c[1]) else float("nan"))
        else:
            out["slope"] = float(c[0])
        return out
