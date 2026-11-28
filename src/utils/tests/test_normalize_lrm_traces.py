"""Tests for src.utils.normalize_lrm_traces schema invariants and dataset parsers."""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.utils.normalize_lrm_traces import (
    ALL_DATASETS,
    build_record,
    iter_grace,
    iter_prmbench,
    iter_processbench,
    stable_task_id,
)

NORM_DIR = Path("data/lrm_traces/normalized")


def test_build_record_all_steps():
    rec = build_record(
        record_id="x:0",
        dataset="grace",
        subset="musique",
        split="train",
        task_id="t0",
        prompt="Q?",
        steps=["a", "b", "c"],
        step_labels=[1, 0, 0],
        label_granularity="all_steps",
    )
    assert rec["num_steps"] == 3
    assert rec["error_steps"] == [1, 2]
    assert rec["first_error_step"] == 1
    assert rec["has_error"] is True


def test_build_record_first_error_only():
    rec = build_record(
        record_id="p:0",
        dataset="processbench",
        subset="math",
        split="test",
        task_id="t1",
        prompt="Q?",
        steps=["a", "b", "c", "d"],
        step_labels=[1, 0, None, None],
        label_granularity="first_error_only",
    )
    assert rec["error_steps"] == [1]
    assert rec["first_error_step"] == 1
    assert rec["has_error"] is True

    with pytest.raises(ValueError):
        build_record(
            record_id="p:bad",
            dataset="processbench",
            subset="math",
            split="test",
            task_id="t1",
            prompt="Q?",
            steps=["a", "b", "c"],
            step_labels=[1, 0, 1],
            label_granularity="first_error_only",
        )


def test_iter_prmbench_converts_1based_and_filters_oob(tmp_path):
    p = tmp_path / "prmbench"
    p.mkdir()
    rows = [
        {
            "idx": "r0",
            "original_question": "1+1?",
            "modified_question": "1+1?",
            "modified_process": ["s1", "s2", "s3"],
            "error_steps": [2, 5],  # 5 is out of bounds for T=3
            "classification": "confidence",
        },
        {
            "idx": "r1",
            "original_question": "2+2?",
            "modified_question": "2+2?",
            "modified_process": ["s1", "s2"],
            "error_steps": [9],  # all OOB -> skipped
            "classification": "confidence",
        },
    ]
    (p / "prmbench_preview.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    skipped: Counter = Counter()
    out = list(iter_prmbench(tmp_path, skipped))
    assert len(out) == 1
    assert out[0]["error_steps"] == [1]
    assert out[0]["step_labels"] == [1, 0, 1]
    assert out[0]["metadata"]["had_oob_error_steps"] is True
    assert skipped["all_errors_oob"] == 1


def test_iter_processbench_and_grace(tmp_path):
    pb = tmp_path / "processbench"
    pb.mkdir()
    (pb / "gsm8k.json").write_text(
        json.dumps([
            {"id": "g0", "problem": "P", "steps": ["s0", "s1", "s2"], "label": 1, "generator": "m", "final_answer_correct": False},
            {"id": "g1", "problem": "P2", "steps": ["s0", "s1"], "label": -1, "generator": "m", "final_answer_correct": True},
        ])
    )
    skipped: Counter = Counter()
    out = list(iter_processbench(tmp_path, skipped))
    assert [r["step_labels"] for r in out] == [[1, 0, None], [1, 1]]
    assert [r["has_error"] for r in out] == [True, False]

    gr = tmp_path / "grace" / "dataset" / "train"
    gr.mkdir(parents=True)
    (tmp_path / "grace" / "dataset" / "test").mkdir(parents=True)
    (gr / "musique.jsonl").write_text(
        json.dumps({
            "id": "gr0",
            "original_question_id": "q42",
            "dataset": "musique",
            "track": "grounding",
            "model": "llama",
            "context": "C",
            "question": "Q",
            "options": None,
            "steps": [
                {"text": "a", "faithfulness": "faithful", "error_category": None},
                {"text": "b", "faithfulness": "unfaithful", "error_category": "Contradiction"},
            ],
        }) + "\n"
    )
    gout = list(iter_grace(tmp_path, Counter()))
    assert len(gout) == 1
    assert gout[0]["task_id"] == "grace:q42"
    assert gout[0]["error_steps"] == [1]
    assert gout[0]["step_labels"] == [1, 0]


def test_stable_task_id_whitespace_invariant():
    assert stable_task_id("x", "a  b\nc") == stable_task_id("x", "a b c")


@pytest.mark.skipif(not (NORM_DIR / "stats.json").exists(), reason="Normalized LRM traces not built")
@pytest.mark.parametrize("name", ALL_DATASETS)
def test_normalized_jsonl_head_valid(name):
    path = NORM_DIR / name / "data.jsonl"
    assert path.exists()
    required = {
        "id", "dataset", "subset", "split", "task_id", "prompt",
        "steps", "num_steps", "step_labels", "error_steps",
        "first_error_step", "has_error", "label_granularity", "metadata",
    }
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            rec = json.loads(line)
            assert set(rec.keys()) == required
            assert rec["dataset"] == name
            assert rec["num_steps"] == len(rec["steps"]) == len(rec["step_labels"]) >= 1
            errs = [k for k, v in enumerate(rec["step_labels"]) if v == 0]
            assert rec["error_steps"] == errs
            assert rec["first_error_step"] == (errs[0] if errs else None)
            assert rec["has_error"] == bool(errs)
