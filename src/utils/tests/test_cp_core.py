"""Properties of the step scorer, the conformal evaluation and the conformal-training losses (CPU, fast)."""

import math

import numpy as np
import torch

from src.utils import cp_eval
from src.utils.cp_data import Split, pad_batch
from src.utils.cp_scorer import StepScorer, rotary
from src.utils.cp_train import cls_loss, size_loss, threshold_step, train


def _sim(n, rng, noise=1.0):
    """Exchangeable toy traces: logits with a noisy bump at the labelled step."""
    scores, y = [], []
    for _ in range(n):
        T = int(rng.integers(5, 30))
        t = int(rng.integers(0, T))
        logit = rng.normal(size=T) * noise
        logit[t] += 2.0
        scores.append(cp_eval.hps(cp_eval.softmax(logit)))
        y.append(t)
    return scores, np.array(y)


def test_calibrate_matches_the_definition():
    rng = np.random.default_rng(0)
    for m in (1, 9, 10, 19, 200):
        s = rng.random(m)
        for alpha in (0.05, 0.1, 0.2):
            k = math.ceil((1 - alpha) * (m + 1))
            want = float("inf") if k > m else sorted(s)[k - 1]
            assert cp_eval.calibrate(s, alpha) == want


def test_split_cp_covers_at_the_target_rate():
    rng = np.random.default_rng(1)
    alpha, m, cov = 0.1, 99, []
    for _ in range(300):
        cs, cy = _sim(m, rng)
        ts, ty = _sim(50, rng)
        cov.append(cp_eval.cal_then_eval(cs, cy, ts, ty, alpha)["coverage"])
    k = math.ceil((1 - alpha) * (m + 1))
    assert abs(np.mean(cov) - k / (m + 1)) < 0.01          # exact marginal coverage is k / (m + 1)


def test_evaluate_matches_a_slow_loop():
    rng = np.random.default_rng(2)
    scores, y = _sim(200, rng)
    q = 0.9
    out = cp_eval.evaluate(scores, y, q)
    sets = [{t for t in range(len(s)) if s[t] <= q} for s in scores]
    assert out["coverage"] == np.mean([t in c for c, t in zip(sets, y)])
    assert out["size"] == np.mean([len(c) for c in sets])
    assert abs(out["size_rel"] - np.mean([len(c) / len(s) for c, s in zip(sets, scores)])) < 1e-12
    assert out["empty_rate"] == np.mean([len(c) == 0 for c in sets])


def test_aps_score_is_the_mass_ranked_above_plus_a_random_share():
    pi = np.array([0.1, 0.5, 0.15, 0.25])
    assert np.allclose(cp_eval.aps(pi, np.zeros(4)), [0.9, 0.0, 0.75, 0.5])
    assert np.allclose(cp_eval.aps(pi, np.ones(4)), [1.0, 0.5, 0.9, 0.75])
    reg = cp_eval.aps(pi, np.zeros(4), lam=0.1, k_reg=2)
    assert np.allclose(reg, [0.9 + 0.2, 0.0, 0.75 + 0.1, 0.5])


def test_log_score_is_zero_for_a_uniform_scorer_at_any_length():
    for T in (3, 17, 60):
        assert np.allclose(cp_eval.log_score(np.full(T, 1 / T)), 0.0, atol=1e-9)


def test_coverage_interval_contains_the_nominal_level():
    lo, hi = cp_eval.coverage_interval(1000, 0.1)
    assert lo < 0.9 < hi and hi - lo < 0.04


def test_scorer_ignores_padding():
    torch.manual_seed(0)
    model = StepScorer(12, d=16, heads=4, ffn=32).eval()
    h = torch.randn(1, 7, 12)
    alone = model(h, torch.ones(1, 7, dtype=torch.bool))
    padded_h = torch.cat([h, torch.randn(1, 5, 12)], dim=1)
    mask = torch.cat([torch.ones(1, 7), torch.zeros(1, 5)], dim=1).bool()
    padded = model(padded_h, mask)
    assert torch.allclose(alone[0], padded[0, :7], atol=1e-5)
    assert torch.isinf(padded[0, 7:]).all()
    assert abs(torch.softmax(padded, dim=1).sum().item() - 1) < 1e-5


def test_rotary_scores_depend_on_relative_position_only():
    torch.manual_seed(0)
    q, k = torch.randn(8), torch.randn(8)
    x = rotary(q.expand(1, 1, 10, 8))
    z = rotary(k.expand(1, 1, 10, 8))
    dots = x[0, 0] @ z[0, 0].T
    assert torch.allclose(dots[2, 5], dots[4, 7], atol=1e-5) and not torch.allclose(dots[2, 5], dots[2, 6], atol=1e-3)


def test_size_loss_tends_to_the_hard_relative_set_size():
    torch.manual_seed(0)
    logits = torch.randn(6, 9)
    mask = torch.ones(6, 9, dtype=torch.bool)
    mask[0, 5:] = False
    logits = logits.masked_fill(~mask, float("-inf"))
    q = 0.9
    pi = torch.softmax(logits, dim=1)
    hard = (((1 - pi) <= q) & mask).sum(1) / mask.sum(1)
    assert torch.allclose(size_loss(logits, mask, q, tau=1e-4), hard.mean(), atol=1e-3)


def test_cls_loss_is_cross_entropy_for_one_label():
    torch.manual_seed(0)
    logits, y = torch.randn(5, 8), torch.randint(0, 8, (5,))
    assert torch.allclose(cls_loss(logits, y), torch.nn.functional.cross_entropy(logits, y), atol=1e-4)


def test_threshold_step_tracks_the_quantile():
    rng = np.random.default_rng(0)
    pool = torch.tensor(rng.beta(2, 2, size=5000), dtype=torch.float32)
    q, alpha, tail = 0.5, 0.1, []
    for i in range(6000):
        q = threshold_step(q, pool[rng.integers(0, 5000, size=32)], alpha, gamma=0.01)
        if i >= 3000:
            tail.append(q)
    assert abs(np.mean(tail) - np.quantile(pool.numpy(), 1 - alpha)) < 0.03


def test_training_learns_a_toy_task_in_every_mode():
    rng = np.random.default_rng(0)

    def make(n):
        s = Split()
        for i in range(n):
            T = int(rng.integers(6, 12))
            t = int(rng.integers(0, T))
            e = rng.normal(size=(T, 8)).astype(np.float16)
            e[t, 0] += 4.0
            s.emb.append(torch.as_tensor(e)); s.y.append(t); s.ids.append(str(i))
        return s

    splits = {"a": make(200), "b": make(200), "val": make(200)}
    base = {"lr": 3e-3, "weight_decay": 0.01, "batch_size": 32, "epochs": 6, "alpha": 0.1, "lam": 0.5, "tau": 0.1}
    for extra in (
        {"mode": "ce", "cls_splits": ["a", "b"], "size_splits": []},
        {"mode": "size_fixed", "cls_splits": ["a"], "size_splits": ["b"], "q_fix": 0.8},
        {"mode": "dpsm", "cls_splits": ["a"], "size_splits": ["b"], "gamma": 0.01},
    ):
        out = train(splits, {**base, **extra}, seed=0, eval_splits=["val"])
        assert out["best_val"]["top1"] > 0.8 and out["best_val"]["size_rel"] < 0.5
        assert len(out["logits"]["val"]) == 200 and len(out["history"]) == 6


def test_saved_runs_keep_dotted_tags_and_round_trip(tmp_path):
    from src.utils.cp_sweep import grid_runs, load_logits, save_run

    runs = grid_runs({"fixed": {"mode": "ce"}, "grid": {"layer": [8], "lr": [0.0001, 0.001]}})
    assert [r["_tag"] for r in runs] == ["layer8_lr0.0001", "layer8_lr0.001"]
    logits = [np.arange(3, dtype=np.float32), np.arange(5, dtype=np.float32)]
    split = Split(emb=[None, None], y=[1, 4], ids=["a", "b"])
    for r in runs:
        save_run(tmp_path / r["_tag"], r, 0, {"logits": {"val": logits}, "best_epoch": 0, "best_val": {}, "history": []}, {"val": split})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["layer8_lr0.0001.json", "layer8_lr0.0001.npz", "layer8_lr0.001.json", "layer8_lr0.001.npz"]
    back = load_logits(tmp_path / "layer8_lr0.0001.npz")["val"]
    assert all(np.array_equal(a, b) for a, b in zip(back["logits"], logits)) and back["y"].tolist() == [1, 4]
