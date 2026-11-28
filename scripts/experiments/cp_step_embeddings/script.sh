#!/usr/bin/env bash
# Build the task-disjoint splits and cache frozen Qwen3.5-9B step embeddings (campaign D002).
#
#   bash scripts/experiments/cp_step_embeddings/script.sh              # real run
#   SMOKE=1 bash scripts/experiments/cp_step_embeddings/script.sh      # rehearsal on a few tasks per split
#   GPUS="0 1" bash scripts/experiments/cp_step_embeddings/script.sh   # pin to assigned cards
#   bash scripts/experiments/cp_step_embeddings/status.sh              # progress
#
# Rerunning is safe: finished stages are skipped via their marker, and extract skips chunks on disk.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

NAME=cp_step_embeddings
SMOKE=${SMOKE:-0}
PROCS_PER_GPU=1            # one model copy per card; two do not fit on a 24 GB card
MIN_FREE_MIB=20000         # use a card only with at least this much free memory
OUT=assets/experiments/$NAME
SMOKE_FLAG=()
if [[ "$SMOKE" == 1 ]]; then
  OUT=$OUT/smoke
  SMOKE_FLAG=(--smoke)
fi
mkdir -p "$OUT"/{logs,splits,emb}

rm -f "$OUT/.failed" "$OUT/.done"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then touch "$OUT/.failed"; fi' EXIT

uv run python -m src.utils.config resolve "scripts/experiments/$NAME/config.yaml" --out "$OUT"
CONFIG=$OUT/config.resolved.yaml

# ------------------------------------------------------ stage 1: splits (CPU)
if [[ -f "$OUT/.splits.done" ]]; then
  echo "splits: already done, skipping"
else
  uv run python -m src.experiments.$NAME.splits --config "$CONFIG" --out "$OUT/splits" "${SMOKE_FLAG[@]}" \
    2>&1 | tee "$OUT/logs/splits.log"
  [[ -s "$OUT/splits/summary.json" ]] || { echo "splits wrote no summary" >&2; exit 1; }
  touch "$OUT/.splits.done"
fi

# --------------------------------------------- stage 2: extract (GPU, sharded)
if [[ -f "$OUT/.extract.done" ]]; then
  echo "extract: already done, skipping"
else
  if [[ -n "${GPUS:-}" ]]; then
    read -r -a GPU_IDS <<< "$GPUS"
  else
    mapfile -t GPU_IDS < <(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v m="$MIN_FREE_MIB" '$2 >= m {print $1}'
    )
  fi
  if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "no GPU with >= ${MIN_FREE_MIB} MiB free" >&2
    exit 1
  fi
  NUM_SHARDS=$(( ${#GPU_IDS[@]} * PROCS_PER_GPU ))
  # Chunks are assigned to shards by index, and finished chunks are skipped, so a changed
  # card count on rerun only redistributes the chunks that are still missing.
  echo "$NUM_SHARDS" > "$OUT/.num_shards"
  echo "using GPUs: ${GPU_IDS[*]} -> ${NUM_SHARDS} shards  smoke=$SMOKE"

  pids=()
  shard=0
  for gpu in "${GPU_IDS[@]}"; do
    for _ in $(seq "$PROCS_PER_GPU"); do
      CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.experiments.$NAME.extract \
        --config "$CONFIG" --splits "$OUT/splits" --out "$OUT/emb" \
        --shard "$shard" --num-shards "$NUM_SHARDS" \
        > "$OUT/logs/extract_shard${shard}.log" 2>&1 &
      pids+=($!)
      shard=$((shard + 1))
    done
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [[ "$fail" -ne 0 ]]; then
    echo "an extract shard failed; check $OUT/logs/" >&2
    exit 1
  fi
  touch "$OUT/.extract.done"
fi

# ------------------------------------ stage 3: verify (CPU; artifacts, not exit codes)
uv run python -m src.experiments.$NAME.verify --config "$CONFIG" --splits "$OUT/splits" --emb "$OUT/emb" --out "$OUT" \
  2>&1 | tee "$OUT/logs/verify.log"
[[ -s "$OUT/manifest.json" ]] || { echo "verify wrote no manifest" >&2; exit 1; }
touch "$OUT/.verify.done"

touch "$OUT/.done"
echo "done -> $OUT/manifest.json"
