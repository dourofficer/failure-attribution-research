"""Properties of the split builder and the step pooling (no GPU, no model)."""

import numpy as np
import torch

from src.experiments.cp_step_embeddings.extract import chunks_of
from src.experiments.cp_step_embeddings.pooling import format_trace, layer_indices, pool_steps, token_step_ids
from src.experiments.cp_step_embeddings.splits import assign_splits, take_smoke

PF, SF = "Problem:\n{prompt}\n\n", "Step {k}:\n{step}\n\n"


def _records(n_tasks=60, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for t in range(n_tasks):
        for j in range(int(rng.integers(1, 4))):
            T = int(rng.integers(2, 12))
            has_error = bool(rng.random() < 0.8)
            steps = [f"s{k}" for k in range(T)]
            if rng.random() < 0.05:
                steps[0] = "  "
            recs.append(
                {
                    "id": f"d:{t}:{j}", "task_id": f"task{t}", "dataset": "d", "subset": "x", "prompt": "p",
                    "steps": steps, "num_steps": T, "has_error": has_error,
                    "first_error_step": int(rng.integers(0, T)) if has_error else None,
                }
            )
    return recs


SETUP = {
    "seed": 7, "group_key": "task_id", "label": "first_error_step",
    "filter": {"has_error": True, "min_steps": 4, "max_steps": 9, "drop_empty_steps": True},
    "splits": {"test": 5, "cal": 5, "train": 10},
}


def test_splits_are_task_disjoint_filtered_and_sized():
    rows, stats = assign_splits(_records(), SETUP)
    assert [stats[f"split_{s}"] for s in ("test", "cal", "train")] == [5, 5, 10]
    assert len({r["task_id"] for r in rows}) == len(rows) == 20          # one trace per task, no task twice
    for r in rows:
        assert 4 <= r["num_steps"] <= 9 and all(s.strip() for s in r["steps"]) and 0 <= r["y"] < r["num_steps"]
    assert stats["tasks"] == 20 + stats["reserve_tasks"]


def test_splits_do_not_depend_on_record_order():
    recs = _records()
    a, _ = assign_splits(recs, SETUP)
    b, _ = assign_splits(recs[::-1], SETUP)
    assert a == b


def test_all_split_takes_every_task_and_smoke_keeps_a_prefix():
    setup = {**SETUP, "splits": {"external": "all"}}
    rows, stats = assign_splits(_records(), setup)
    assert len(rows) == stats["tasks"] and stats["reserve_tasks"] == 0
    full, _ = assign_splits(_records(), SETUP)
    small = take_smoke(full, 2)
    assert [r["split"] for r in small] == ["test"] * 2 + ["cal"] * 2 + ["train"] * 2
    assert all(r in full for r in small)


def test_spans_recover_the_step_text():
    steps = ["first {weird} step", "x", "third\nwith newline"]
    text, spans = format_trace("a {braced} prompt", steps, PF, SF)
    assert [text[a:b] for a, b in spans] == steps
    assert text.startswith("Problem:\na {braced} prompt\n\nStep 1:\n")


def test_token_assignment_matches_a_slow_loop():
    rng = np.random.default_rng(0)
    text, spans = format_trace("p" * 13, ["a" * int(n) for n in rng.integers(1, 9, size=7)], PF, SF)
    cuts = np.unique(np.concatenate([[0, len(text)], rng.integers(0, len(text), size=40)]))
    offsets = np.stack([cuts[:-1], cuts[1:]], axis=1)
    offsets = np.concatenate([offsets, [[5, 5]]])                           # a zero-width token
    slow = []
    for a, b in offsets:
        hit = [t for t, (lo, hi) in enumerate(spans) if lo <= a < hi and b > a]
        slow.append(hit[0] if hit else -1)
    assert token_step_ids(offsets, spans).tolist() == slow


def test_pooling_matches_a_slow_mean_and_rejects_empty_steps():
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(50, 6, generator=g).to(torch.bfloat16)
    step_ids = torch.randint(-1, 4, (50,), generator=g)
    step_ids[:4] = torch.arange(4)
    fast = pool_steps(hidden, step_ids, 4)
    slow = torch.stack([hidden[step_ids == t].float().mean(0) for t in range(4)])
    assert torch.allclose(fast, slow, atol=1e-5)
    try:
        pool_steps(hidden, step_ids, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("a step with no token must raise")


def test_layer_indices_and_chunks():
    assert layer_indices(36, [0.25, 0.5, 0.75, 1.0]) == [9, 18, 27, 36]
    chunks = chunks_of(10, 4)
    assert [list(c) for c in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
