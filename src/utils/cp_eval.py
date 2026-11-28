"""Vanilla split conformal prediction over the steps of a trace, and its metrics.

Inputs
    scores   list of float arrays, one per trace, [T_i]: nonconformity S(x, t), lower = more likely an error step
    y        int array [n]: the labelled error step of each trace (|Y| = 1)

Outputs
    calibrate()        the threshold Q(alpha): the ceil((1-alpha)(m+1))-th smallest calibration score, or +inf
    evaluate()         coverage, mean set size, mean relative set size, empty and full set rates, and coverage by
                       length tercile and by error-position tercile, for the sets {t : S(x, t) <= Q}
    coverage_interval()  the Beta(k, m+1-k) interval of test coverage given the calibration set size
    score functions    hps (1 - pi), log_score (-log(T pi)), aps and raps (cumulative mass, randomised)
    recalibrated()     the same metrics averaged over repeated random calibration/test splits of one set
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta

# ---------------------------------------------------------------- score functions (from logits)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def hps(pi: np.ndarray) -> np.ndarray:
    """The work package's score: S = 1 - pi_t."""
    return 1.0 - pi


def log_score(pi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Length-normalised log-probability score: S = -log(T pi_t)."""
    return -np.log(len(pi) * pi + eps)


def aps(pi: np.ndarray, u: np.ndarray, lam: float = 0.0, k_reg: int = 0) -> np.ndarray:
    """Randomised adaptive-prediction-set score; with lam > 0 the regularised variant.

    S_t = (mass of steps ranked strictly above t) + u_t * pi_t + lam * max(0, rank_t - k_reg), rank from 1.
    """
    order = np.argsort(-pi, kind="stable")
    rank = np.empty(len(pi), dtype=np.int64)
    rank[order] = np.arange(1, len(pi) + 1)
    above = np.empty(len(pi))
    above[order] = np.cumsum(pi[order]) - pi[order]
    return above + u * pi + lam * np.maximum(0, rank - k_reg)


# ---------------------------------------------------------------- split conformal prediction


def calibrate(cal_scores: np.ndarray, alpha: float) -> float:
    """cal_scores [m]: the score of the labelled step of each calibration trace."""
    m = len(cal_scores)
    k = math.ceil((1 - alpha) * (m + 1))
    return float("inf") if k > m else float(np.sort(cal_scores)[k - 1])


def label_scores(scores: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    return np.array([s[t] for s, t in zip(scores, y)])


def _tercile(values: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(values, [1 / 3, 2 / 3])
    return np.where(values <= lo, 0, np.where(values <= hi, 1, 2))


def evaluate(scores: list[np.ndarray], y: np.ndarray, q: float) -> dict:
    n = len(scores)
    T = np.array([len(s) for s in scores])
    size = np.array([int((s <= q).sum()) for s in scores])
    hit = np.array([bool(s[t] <= q) for s, t in zip(scores, y)])
    top1 = np.array([int(np.argmin(s)) == t for s, t in zip(scores, y)])
    out = {
        "n": n,
        "coverage": float(hit.mean()),
        "size": float(size.mean()),
        "size_rel": float((size / T).mean()),
        "empty_rate": float((size == 0).mean()),
        "full_rate": float((size == T).mean()),
        "top1": float(top1.mean()),
    }
    for name, values in (("len", T), ("pos", y / T)):
        g = _tercile(values)
        out[f"coverage_by_{name}_tercile"] = [float(hit[g == j].mean()) if (g == j).any() else None for j in range(3)]
        out[f"size_rel_by_{name}_tercile"] = [float((size / T)[g == j].mean()) if (g == j).any() else None for j in range(3)]
    return out


def coverage_interval(m: int, alpha: float, level: float = 0.95) -> tuple[float, float]:
    """Interval of the coverage of split CP, conditional on a calibration set of size m (infinite test set)."""
    k = math.ceil((1 - alpha) * (m + 1))
    if k > m:
        return (1.0, 1.0)
    lo, hi = beta.ppf([(1 - level) / 2, 1 - (1 - level) / 2], k, m + 1 - k)
    return float(lo), float(hi)


def cal_then_eval(cal_scores, cal_y, test_scores, test_y, alpha: float) -> dict:
    q = calibrate(label_scores(cal_scores, cal_y), alpha)
    return {"q": q, **evaluate(test_scores, test_y, q)}


def recalibrated(scores: list[np.ndarray], y: np.ndarray, alpha: float, n_splits: int = 200, seed: int = 0) -> dict:
    """Mean and standard deviation of the scalar metrics over random 50/50 calibration/test splits."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    runs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, test = perm[: n // 2], perm[n // 2 :]
        runs.append(cal_then_eval([scores[i] for i in cal], y[cal], [scores[i] for i in test], y[test], alpha))
    keys = ["coverage", "size", "size_rel", "empty_rate", "full_rate", "top1"]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out.update({f"{k}_sd": float(np.std([r[k] for r in runs])) for k in keys})
    out["n_cal"], out["n_test"], out["n_splits"] = n // 2, n - n // 2, n_splits
    return out


def two_fold(scores: list[np.ndarray], y: np.ndarray, alpha: float, seed: int = 0) -> dict:
    """Model selection on one set: split it in two fixed halves, calibrate on each, score the other, average."""
    perm = np.random.default_rng(seed).permutation(len(scores))
    a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
    runs = []
    for cal, test in ((a, b), (b, a)):
        runs.append(cal_then_eval([scores[i] for i in cal], y[cal], [scores[i] for i in test], y[test], alpha))
    return {k: float(np.mean([r[k] for r in runs])) for k in ("coverage", "size", "size_rel", "empty_rate", "top1")}
