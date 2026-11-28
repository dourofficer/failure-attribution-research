#!/usr/bin/env bash
# Print the state of cp_step_embeddings, read from the output folder only. Safe while script.sh runs.
#   bash scripts/experiments/cp_step_embeddings/status.sh          # real run
#   SMOKE=1 bash scripts/experiments/cp_step_embeddings/status.sh  # smoke run
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT=assets/experiments/cp_step_embeddings
[[ "${SMOKE:-0}" == 1 ]] && OUT=$OUT/smoke
if [[ ! -d "$OUT" ]]; then echo "not started ($OUT missing)"; exit 0; fi

for stage in splits extract verify; do
  if [[ -f "$OUT/.$stage.done" ]]; then state=done; else state=pending; fi
  echo "$stage: $state"
done
rows=$(cat "$OUT"/splits/*.jsonl 2>/dev/null | wc -l)
chunks=$(find "$OUT/emb" -name 'chunk*.npz' 2>/dev/null | wc -l)
echo "traces in splits: $rows   chunks written: $chunks (256 traces each, last one partial)"
for log in "$OUT"/logs/extract_shard*.log; do
  [[ -f "$log" ]] && echo "$(basename "$log"): $(tr '\r' '\n' < "$log" | tail -n 1 | cut -c1-110)"
done
if [[ -f "$OUT/.failed" ]]; then echo "run: FAILED"
elif [[ -f "$OUT/.done" ]]; then echo "run: done"
else echo "run: in progress or not started"; fi
