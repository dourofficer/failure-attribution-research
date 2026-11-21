#!/usr/bin/env bash
# Print the state of probe_layer_sweep, read from the output folder only. Safe
# to run while script.sh is running.
#   bash scripts/experiments/probe_layer_sweep/status.sh          # real run
#   SMOKE=1 bash scripts/experiments/probe_layer_sweep/status.sh  # smoke run
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT=assets/experiments/probe_layer_sweep
[[ "${SMOKE:-0}" == 1 ]] && OUT=$OUT/smoke

if [[ ! -d "$OUT" ]]; then
  echo "not started ($OUT missing)"
  exit 0
fi

total=$(cat "$OUT/.num_shards" 2>/dev/null || echo "?")
shards=$(find "$OUT/hidden" -name 'shard*.npz' 2>/dev/null | wc -l)

for stage in extract sweep; do
  if [[ -f "$OUT/.$stage.done" ]]; then state=done; else state=pending; fi
  echo "$stage: $state"
done
echo "extract shards: $shards/$total"
if [[ -f "$OUT/.failed" ]]; then echo "run: FAILED"
elif [[ -f "$OUT/.done" ]]; then echo "run: done"
else echo "run: in progress or not started"; fi
