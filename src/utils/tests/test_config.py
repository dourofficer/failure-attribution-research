"""Tests for setup resolution: a wrong setup silently changes what every result means."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import check, resolve, uses


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny repo: two models, one dataset, two experiments sharing the dataset."""
    for kind, setup_id, body in [
        ("model", "lm_v1", {"hub_id": "org/lm", "dtype": "bf16"}),
        ("model", "lm_v2", {"hub_id": "org/lm", "dtype": "fp32"}),
        ("dataset", "dev_v1", {"split": "dev", "n": 100}),
    ]:
        path = tmp_path / "setup" / kind / f"{setup_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(body))
    for name, setup in [
        ("exp_a", {"model": "lm_v1", "dataset": "dev_v1"}),
        ("exp_b", {"model": ["lm_v1", "lm_v2"], "dataset": "dev_v1"}),
    ]:
        path = tmp_path / "scripts" / "experiments" / name / "config.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"setup": setup, "params": {"seed": 0}}))
    return tmp_path


def config_of(repo: Path, name: str) -> Path:
    return repo / "scripts" / "experiments" / name / "config.yaml"


def test_resolve_expands_ids_and_keeps_params(repo):
    resolved = resolve(config_of(repo, "exp_a"), repo)
    assert resolved["setup"]["model"]["id"] == "lm_v1"
    assert resolved["setup"]["model"]["content"] == {"hub_id": "org/lm", "dtype": "bf16"}
    assert resolved["setup"]["dataset"]["content"]["n"] == 100
    assert resolved["params"] == {"seed": 0}


def test_resolve_keeps_a_list_of_ids_as_a_list(repo):
    models = resolve(config_of(repo, "exp_b"), repo)["setup"]["model"]
    assert [m["id"] for m in models] == ["lm_v1", "lm_v2"]


def test_same_id_resolves_identically_across_experiments(repo):
    a = resolve(config_of(repo, "exp_a"), repo)["setup"]["dataset"]
    b = resolve(config_of(repo, "exp_b"), repo)["setup"]["dataset"]
    assert a == b


def test_resolve_rejects_missing_setup_file(repo):
    config_of(repo, "exp_a").write_text(yaml.safe_dump({"setup": {"model": "nope"}}))
    with pytest.raises(FileNotFoundError, match="setup/model/nope.yaml"):
        resolve(config_of(repo, "exp_a"), repo)


def test_resolve_rejects_inline_setup_overrides(repo):
    config_of(repo, "exp_a").write_text(yaml.safe_dump({"setup": {"model": {"dtype": "fp16"}}}))
    with pytest.raises(ValueError, match="must be an id"):
        resolve(config_of(repo, "exp_a"), repo)


def test_resolve_rejects_unknown_top_level_keys(repo):
    config_of(repo, "exp_a").write_text(yaml.safe_dump({"setup": {}, "model": "lm_v1"}))
    with pytest.raises(ValueError, match="unknown top-level keys"):
        resolve(config_of(repo, "exp_a"), repo)


def test_uses_finds_single_and_list_references(repo):
    assert uses("dataset", "dev_v1", repo) == ["exp_a", "exp_b"]
    assert uses("model", "lm_v2", repo) == ["exp_b"]
    assert uses("model", "absent", repo) == []


def test_check_flags_a_setup_file_edited_after_the_run(repo):
    out = repo / "assets" / "experiments" / "exp_a"
    out.mkdir(parents=True)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(resolve(config_of(repo, "exp_a"), repo)))
    assert check(repo) == []

    (repo / "setup" / "model" / "lm_v1.yaml").write_text(yaml.safe_dump({"hub_id": "org/lm", "dtype": "fp16"}))
    problems = check(repo)
    assert len(problems) == 1 and "setup/model/lm_v1.yaml" in problems[0] and "edited" in problems[0]

    (repo / "setup" / "dataset" / "dev_v1.yaml").unlink()
    assert any("no longer exists" in p for p in check(repo))
