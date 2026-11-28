#!/usr/bin/env bash
# Cross-entropy plus the soft set-size loss with a constant threshold tuned on validation (campaign D002).
# Uses the embedding depth selected by cp_ce_full (assets/experiments/cp_ce_full/best.json).
#
#   bash scripts/experiments/cp_ce_fixed_threshold/script.sh              # real run
#   SMOKE=1 bash scripts/experiments/cp_ce_fixed_threshold/script.sh      # rehearsal on the smoke embedding cache
#   GPUS="0" bash scripts/experiments/cp_ce_fixed_threshold/script.sh     # pin to assigned cards
#   GPUS=cpu SMOKE=1 bash scripts/experiments/cp_ce_fixed_threshold/script.sh   # rehearsal without a card
#
# Rerunning is safe: finished stages are skipped via their marker, and finished training runs are skipped.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

NAME=cp_ce_fixed_threshold
SMOKE=${SMOKE:-0}
PROCS_PER_GPU=${PROCS_PER_GPU:-3}   # each process holds one embedding depth (~2 GB) and a tiny model
MIN_FREE_MIB=8000
OUT=assets/experiments/$NAME
EXTRA=()
if [[ "$SMOKE" == 1 ]]; then
  OUT=$OUT/smoke
  EXTRA=(--emb assets/experiments/cp_step_embeddings/smoke/emb --layer-from assets/experiments/cp_ce_full/smoke/best.json --limit 2 --epochs 2)
fi
mkdir -p "$OUT/logs"

rm -f "$OUT/.failed" "$OUT/.done"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then touch "$OUT/.failed"; fi' EXIT

uv run python -m src.utils.config resolve "scripts/experiments/$NAME/config.yaml" --out "$OUT"
CONFIG=$OUT/config.resolved.yaml

if [[ "${GPUS:-}" == cpu ]]; then
  GPU_IDS=("")                       # GPUS=cpu hides every card (smoke while the cards are busy)
elif [[ -n "${GPUS:-}" ]]; then
  read -r -a GPU_IDS <<< "$GPUS"
else
  mapfile -t GPU_IDS < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' -v m="$MIN_FREE_MIB" '$2 >= m {print $1}'
  )
fi
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then echo "no GPU with >= ${MIN_FREE_MIB} MiB free" >&2; exit 1; fi
NUM_SHARDS=$(( ${#GPU_IDS[@]} * PROCS_PER_GPU ))
echo "using GPUs: ${GPU_IDS[*]} -> ${NUM_SHARDS} processes  smoke=$SMOKE"

# Train one shard per process; finished runs are skipped, so the shard count may change between launches.
train_stage() {
  local stage=$1 pids=() shard=0 fail=0
  for gpu in "${GPU_IDS[@]}"; do
    for _ in $(seq "$PROCS_PER_GPU"); do
      CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 uv run python -m src.utils.cp_sweep run \
        --config "$CONFIG" --out "$OUT" --stage "$stage" --shard "$shard" --num-shards "$NUM_SHARDS" "${EXTRA[@]}" \
        > "$OUT/logs/${stage}_shard${shard}.log" 2>&1 &
      pids+=($!)
      shard=$((shard + 1))
    done
  done
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [[ "$fail" -ne 0 ]]; then echo "a $stage process failed; check $OUT/logs/" >&2; exit 1; fi
}

# ------------------------------------------------ stage 1: sweep (GPU), then selection on validation
if [[ -f "$OUT/.sweep.done" ]]; then
  echo "sweep: already done, skipping"
else
  train_stage sweep
  uv run python -m src.utils.cp_sweep select --config "$CONFIG" --out "$OUT" "${EXTRA[@]}" | tee "$OUT/logs/select.log"
  [[ -s "$OUT/best.json" ]] || { echo "selection wrote no best.json" >&2; exit 1; }
  touch "$OUT/.sweep.done"
fi

# ------------------------------------------------ stage 2: final seeds at the selected configuration
if [[ -f "$OUT/.final.done" ]]; then
  echo "final: already done, skipping"
else
  train_stage final
  n_seeds=$(uv run python -c "import yaml,sys; print(len(yaml.safe_load(open('$CONFIG'))['params']['seeds']))")
  found=$(find "$OUT/final" -name 'seed*.json' | wc -l)
  [[ "$found" -eq "$n_seeds" ]] || { echo "expected $n_seeds final runs, found $found" >&2; exit 1; }
  touch "$OUT/.final.done"
fi

# ------------------------------------------------ stage 3: evaluation (CPU)
uv run python -m src.utils.cp_report --config "$CONFIG" --out "$OUT" --arm "cross-entropy + set-size loss, fixed threshold" > "$OUT/logs/evaluate.log" 2>&1
[[ -s "$OUT/results.json" ]] || { echo "evaluation wrote no results.json" >&2; exit 1; }
touch "$OUT/.evaluate.done"

touch "$OUT/.done"
echo "done -> $OUT/results.md"
