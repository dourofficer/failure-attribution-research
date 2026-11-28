"""Arms that need no retraining, built from the logits of the cross-entropy scorer, and the trivial floors.

Inputs
    logits   {split: {"logits": [array per trace], "y": array}} of one trained scorer (cp_sweep.load_logits)
    train_rel_pos   y / T of the training traces (for the position-prior floor)
    params["posthoc"]: temperatures, raps_lambda, raps_k_reg, position_bins

Outputs
    arms()   {arm description: {split: [score array per trace]}}; lower score = more likely in the set.
             Tunable arms (temperature, regularised adaptive sets) pick their setting on validation by
             calibrated relative set size at the selection alpha, like every trained arm.
"""

from __future__ import annotations

import numpy as np

from src.utils import cp_eval


def _apply(logits: dict, fn) -> dict:
    return {split: [fn(l, i) for i, l in enumerate(d["logits"])] for split, d in logits.items()}


def _uniforms(logits: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {split: [rng.random(len(l)) for l in d["logits"]] for split, d in logits.items()}


def _val_size(scores: dict, logits: dict, alpha: float) -> float:
    return cp_eval.two_fold(scores["val"], logits["val"]["y"], alpha, seed=0)["size_rel"]


def position_prior(train_rel_pos: np.ndarray, bins: int) -> np.ndarray:
    """Smoothed share of training error steps falling in each of `bins` equal slices of the trace."""
    counts = np.bincount(np.minimum((train_rel_pos * bins).astype(int), bins - 1), minlength=bins) + 1.0
    return counts / counts.sum()


def position_scores(T: int, prior: np.ndarray) -> np.ndarray:
    """1 - pi_t where pi_t is proportional to the prior of the slice that step t (of T) falls in."""
    bins = len(prior)
    p = prior[np.minimum((np.arange(T) / T * bins).astype(int), bins - 1)]
    return 1.0 - p / p.sum()


def arms(logits: dict, train_rel_pos: np.ndarray, post: dict, alpha: float, seed: int) -> tuple[dict, dict]:
    out, chosen = {}, {}
    out["cross-entropy, all training data"] = _apply(logits, lambda l, i: cp_eval.hps(cp_eval.softmax(l)))

    by_t = {t: _apply(logits, lambda l, i, t=t: cp_eval.hps(cp_eval.softmax(l, t))) for t in post["temperatures"]}
    t_best = min(by_t, key=lambda t: _val_size(by_t[t], logits, alpha))
    out["cross-entropy + temperature scaling"], chosen["temperature"] = by_t[t_best], t_best

    out["cross-entropy + length-normalised log-probability score"] = _apply(logits, lambda l, i: cp_eval.log_score(cp_eval.softmax(l)))

    u = _uniforms(logits, seed)
    out["cross-entropy + adaptive-set score"] = {
        s: [cp_eval.aps(cp_eval.softmax(l), u[s][i]) for i, l in enumerate(d["logits"])] for s, d in logits.items()
    }
    raps = {
        (lam, k): {s: [cp_eval.aps(cp_eval.softmax(l), u[s][i], lam, k) for i, l in enumerate(d["logits"])] for s, d in logits.items()}
        for lam in post["raps_lambda"]
        for k in post["raps_k_reg"]
    }
    key = min(raps, key=lambda k: _val_size(raps[k], logits, alpha))
    out["cross-entropy + regularised adaptive-set score"], chosen["raps_lambda_k"] = raps[key], list(key)

    rng = np.random.default_rng(seed + 1000)
    out["floor: random scores"] = {s: [rng.random(len(l)) for l in d["logits"]] for s, d in logits.items()}
    prior = position_prior(train_rel_pos, post["position_bins"])
    out["floor: error-position prior"] = _apply(logits, lambda l, i: position_scores(len(l), prior))
    chosen["position_prior"] = prior.tolist()
    return out, chosen
