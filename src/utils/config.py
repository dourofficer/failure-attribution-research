"""Experiment configs that name shared setup files by id.

Inputs
    scripts/experiments/<name>/config.yaml   {"setup": {<kind>: <id> | [<id>, ...]}, "params": {...}}
    setup/<kind>/<id>.yaml                   one shared setup (a dataset, a model, a metric, ...)

Outputs
    resolve()  the config with every id expanded to the content of its setup file, plus the file's
               sha256; the CLI writes it to <out>/config.resolved.yaml
    uses()     the experiments whose config names a given setup id
    check()    the resolved configs whose setup file was edited after the run that cited it

CLI (from the repo root)
    uv run python -m src.utils.config resolve scripts/experiments/<name>/config.yaml --out assets/experiments/<name>
    uv run python -m src.utils.config uses model example_lm_v1
    uv run python -m src.utils.config check
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _ids(value) -> list[str]:
    """A setup reference is one id or a list of ids."""
    return list(value) if isinstance(value, list) else [value]


def _load_setup(kind: str, setup_id: str, repo: Path) -> dict:
    path = repo / "setup" / kind / f"{setup_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"setup file not found: {path.relative_to(repo)}")
    raw = path.read_bytes()
    return {"id": setup_id, "sha256": hashlib.sha256(raw).hexdigest(), "content": yaml.safe_load(raw)}


def resolve(config_path: Path, repo: Path = REPO) -> dict:
    """Expand every setup id in an experiment config. Raises on anything but `setup` and `params`."""
    config = yaml.safe_load(Path(config_path).read_text()) or {}
    extra = set(config) - {"setup", "params"}
    if extra:
        raise ValueError(f"{config_path}: unknown top-level keys {sorted(extra)}; use only `setup` and `params`")

    resolved_setup = {}
    for kind, value in (config.get("setup") or {}).items():
        if isinstance(value, dict):
            raise ValueError(f"{config_path}: `setup.{kind}` must be an id or a list of ids, not inline fields")
        loaded = [_load_setup(kind, setup_id, repo) for setup_id in _ids(value)]
        resolved_setup[kind] = loaded if isinstance(value, list) else loaded[0]
    return {"setup": resolved_setup, "params": config.get("params") or {}}


def uses(kind: str, setup_id: str, repo: Path = REPO) -> list[str]:
    """Names of the experiments whose config.yaml references setup/<kind>/<setup_id>.yaml."""
    names = []
    for config_path in sorted((repo / "scripts" / "experiments").glob("*/config.yaml")):
        setup = (yaml.safe_load(config_path.read_text()) or {}).get("setup") or {}
        if kind in setup and setup_id in _ids(setup[kind]):
            names.append(config_path.parent.name)
    return names


def check(repo: Path = REPO) -> list[str]:
    """One message per setup file that changed (or vanished) after a run recorded it."""
    problems = []
    for resolved_path in sorted((repo / "assets" / "experiments").glob("**/config.resolved.yaml")):
        resolved = yaml.safe_load(resolved_path.read_text()) or {}
        for kind, value in (resolved.get("setup") or {}).items():
            for entry in _ids(value):
                setup_path = repo / "setup" / kind / f"{entry['id']}.yaml"
                where = f"{resolved_path.relative_to(repo)} cites setup/{kind}/{entry['id']}.yaml"
                if not setup_path.is_file():
                    problems.append(f"{where}, which no longer exists")
                elif hashlib.sha256(setup_path.read_bytes()).hexdigest() != entry["sha256"]:
                    problems.append(f"{where}, which was edited after the run")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_resolve = sub.add_parser("resolve", help="write config.resolved.yaml for one experiment")
    p_resolve.add_argument("config", type=Path)
    p_resolve.add_argument("--out", type=Path, required=True, help="output folder of the run")
    p_uses = sub.add_parser("uses", help="list the experiments that share a setup id")
    p_uses.add_argument("kind")
    p_uses.add_argument("id")
    sub.add_parser("check", help="detect setup files edited after a run cited them")
    args = parser.parse_args()

    if args.command == "resolve":
        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / "config.resolved.yaml"
        out_path.write_text(yaml.safe_dump(resolve(args.config), sort_keys=False))
        print(out_path)
    elif args.command == "uses":
        print("\n".join(uses(args.kind, args.id)) or f"no experiment uses {args.kind}/{args.id}")
    else:
        problems = check()
        print("\n".join(problems) or "all cited setup files are unchanged")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
