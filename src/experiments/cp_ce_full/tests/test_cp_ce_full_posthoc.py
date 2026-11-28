"""Properties of the no-retraining arms and floors (CPU, fast)."""

import numpy as np

from src.experiments.cp_ce_full.posthoc import arms, position_prior, position_scores
from src.utils import cp_eval
from src.utils.cp_report import aggregate, evaluate_arm

POST = {"temperatures": [0.5, 1.0, 2.0], "raps_lambda": [0.01, 0.1], "raps_k_reg": [1, 2], "position_bins": 5}


def _logits(rng, n):
    out, y = [], []
    for _ in range(n):
        T = int(rng.integers(6, 20))
        t = int(rng.integers(0, T))
        l = rng.normal(size=T).astype(np.float32)
        l[t] += 2.5
        out.append(l); y.append(t)
    return {"logits": out, "y": np.array(y)}


def test_position_prior_sums_to_one_and_scores_follow_it():
    prior = position_prior(np.array([0.0, 0.05, 0.1, 0.5, 0.99]), 5)
    assert abs(prior.sum() - 1) < 1e-12 and prior.argmax() == 0
    s = position_scores(10, prior)
    assert abs((1 - s).sum() - 1) < 1e-12 and s[0] == s[1] == s.min()


def test_arms_have_one_score_array_per_trace_and_temperature_one_is_the_reference():
    rng = np.random.default_rng(0)
    logits = {s: _logits(rng, 60) for s in ("val", "cal", "test", "external")}
    out, chosen = arms(logits, rng.random(200), {**POST, "temperatures": [1.0]}, 0.1, seed=0)
    assert len(out) == 7 and chosen["temperature"] == 1.0
    for scores in out.values():
        for split, d in logits.items():
            assert [len(a) for a in scores[split]] == [len(l) for l in d["logits"]]
    ref, temp = out["cross-entropy, all training data"], out["cross-entropy + temperature scaling"]
    assert all(np.allclose(a, b) for a, b in zip(ref["test"], temp["test"]))


def test_every_arm_is_calibrated_and_a_trained_scorer_beats_the_floors():
    rng = np.random.default_rng(1)
    logits = {s: _logits(rng, 400) for s in ("val", "cal", "test", "external")}
    out, _ = arms(logits, rng.random(500), POST, 0.1, seed=0)
    res = {name: evaluate_arm(scores, {s: d["y"] for s, d in logits.items()}, [0.1], n_recal=20)["0.1"] for name, scores in out.items()}
    for name, r in res.items():
        assert 0.84 < r["test"]["coverage"] < 0.96, name
        assert 0.84 < r["external_recal"]["coverage"] < 0.96, name
    assert res["cross-entropy, all training data"]["test"]["size_rel"] < res["floor: random scores"]["test"]["size_rel"] - 0.2
    assert abs(res["floor: random scores"]["test"]["size_rel"] - 0.9) < 0.06
    agg = aggregate([res["floor: random scores"], res["floor: random scores"]])
    assert agg["test"]["coverage"]["sd"] == 0.0
