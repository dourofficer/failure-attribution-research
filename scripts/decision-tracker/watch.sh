#!/usr/bin/env bash
# Launch the decision tracker in a detached tmux session:
#   - pane 1: build.py --watch  (rebuilds index.md/graph.html on any change)
#   - pane 2: static http server (so graph.html auto-reload + assets work)
#
# Usage:
#   bash scripts/decision-tracker/watch.sh          # start (idempotent)
#   bash scripts/decision-tracker/watch.sh stop     # stop the session
#   tmux attach -t dt-watch                          # view logs
#
# Then open:  http://localhost:${PORT}/graph.html
# ---------------------------------------------------------------------------

set -euo pipefail

SESSION="dt-watch"
PORT="8148"

REPO="$(git rev-parse --show-toplevel)"
DATA_DIR="$REPO/docs/decisions"

if [[ "${1:-}" == "stop" ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "no session $SESSION"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running."
  echo "open http://localhost:${PORT}/graph.html   (tmux attach -t $SESSION for logs)"
  exit 0
fi

# Ensure the views exist before serving.
( cd "$REPO" && uv run python src/decision-tracker/build.py )

# Pane 1: watcher (rebuilds on change).
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "uv run python src/decision-tracker/build.py --watch"

# Pane 2: static server rooted at the data dir, so '.build-stamp' + assets are fetchable.
tmux split-window -t "$SESSION" -c "$DATA_DIR" \
  "uv run python -m http.server ${PORT}"
tmux select-layout -t "$SESSION" even-vertical >/dev/null

echo "started '$SESSION'."
echo "open  http://localhost:${PORT}/graph.html"
echo "logs  tmux attach -t $SESSION      stop  bash scripts/decision-tracker/watch.sh stop"
