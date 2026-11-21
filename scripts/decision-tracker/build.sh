#!/usr/bin/env bash
# One-off build of the decision tracker views (index.md + graph.html).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run python src/decision-tracker/build.py "$@"
