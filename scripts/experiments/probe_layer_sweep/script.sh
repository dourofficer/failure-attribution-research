#!/usr/bin/env bash
# EXAMPLE experiment runner that meets the run standard in CLAUDE.md: resolved
# config, SMOKE=1 rehearsal, GPU detection, data-parallel shards, resume, one
# marker per stage, .done / .failed for whoever waits on it.
#
#   bash scripts/experiments/probe_layer_sweep/script.sh              # real run
#   SMOKE=1 bash scripts/experiments/probe_layer_sweep/script.sh      # rehearsal
#   GPUS="0 1" bash scripts/experiments/probe_layer_sweep/script.sh   # pin to assigned cards
#   bash scripts/experiments/probe_layer_sweep/status.sh              # progress
#
# Rerunning after a crash is safe: finished stages are skipped via their marker,
# and extract.py skips shards whose output file already exists.
#
# NOTE: extract.py and sweep.py are not included. This file shows the shape and
# is not runnable as-is. Delete it with the rest of the example.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ---------------------------------------------------------------- parameters
# Only the run's shape lives here. What is measured lives in config.yaml.
NAME=probe_layer_sweep
SMOKE=${SMOKE:-0}
PROCS_PER_GPU=2            # processes stacked per card
MIN_FREE_MIB=8000          # use a card only with at least this much free memory
N_JOBS=24                  # CPU workers; never exceed 32
OUT=assets/experiments/$NAME
LIMIT=0                    # 0 = every example in the dataset setup

# Smoke: same stages, same code path, a handful of items, in its own folder.
if [[ "$SMOKE" == 1 ]]; then
  LIMIT=16
  OUT=$OUT/smoke
fi

mkdir -p "$OUT"/{logs,plots,hidden}

# ------------------------------------------------------------------- markers
# .done when every stage finished, .failed on any non-zero exit.
rm -f "$OUT/.failed" "$OUT/.done"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then touch "$OUT/.failed"; fi' EXIT

# -------------------------------------------------------------------- config
# Expand the setup ids once; every stage reads the resolved file.
uv run python -m src.utils.config resolve "scripts/experiments/$NAME/config.yaml" --out "$OUT"
CONFIG=$OUT/config.resolved.yaml

# ---------------------------------------------------------------------- GPUs
# GPUS="0 1" pins the run to its assigned cards (parallel experiments get
# disjoint cards). Otherwise take every card with enough free memory.
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
echo "using GPUs: ${GPU_IDS[*]}  (${PROCS_PER_GPU}/card -> ${NUM_SHARDS} shards)  smoke=$SMOKE"

# ------------------------------------------------ stage 1: extract (GPU, sharded)
if [[ -f "$OUT/.extract.done" ]]; then
  echo "extract: already done, skipping"
else
  # A partial run pins the shard count; a changed GPU set must not re-split it.
  if [[ -f "$OUT/.num_shards" ]]; then
    NUM_SHARDS=$(cat "$OUT/.num_shards")
  else
    echo "$NUM_SHARDS" > "$OUT/.num_shards"   # status.sh reads this too
  fi
  shard=0
  while [[ "$shard" -lt "$NUM_SHARDS" ]]; do
    for gpu in "${GPU_IDS[@]}"; do
      for _ in $(seq "$PROCS_PER_GPU"); do
        [[ "$shard" -ge "$NUM_SHARDS" ]] && break
        # extract.py writes $OUT/hidden/shard{i}.npz at the end and exits early
        # if it already exists. Its tqdm bar goes to the log.
        CUDA_VISIBLE_DEVICES="$gpu" \
        uv run python "src/experiments/$NAME/extract.py" \
          --config "$CONFIG" --limit "$LIMIT" \
          --shard "$shard" --num-shards "$NUM_SHARDS" \
          --out "$OUT/hidden" \
          > "$OUT/logs/extract_shard${shard}.log" 2>&1 &
        shard=$((shard + 1))
      done
    done
    wait
  done

  # Verify artifacts, not exit codes: a run that exits 0 may have produced nothing.
  found=$(find "$OUT/hidden" -name 'shard*.npz' | wc -l)
  if [[ "$found" -ne "$NUM_SHARDS" ]]; then
    echo "expected $NUM_SHARDS shards, found $found; check $OUT/logs/" >&2
    exit 1
  fi
  echo "extraction OK: $found/$NUM_SHARDS shards"
  touch "$OUT/.extract.done"
fi

# --------------------------------------------- stage 2: sweep (CPU, capped)
if [[ -f "$OUT/.sweep.done" ]]; then
  echo "sweep: already done, skipping"
else
  # Cap BLAS threads or the joblib workers oversubscribe the box.
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1

  uv run python "src/experiments/$NAME/sweep.py" \
    --config "$CONFIG" \
    --hidden "$OUT/hidden" \
    --n-jobs "$N_JOBS" \
    --out "$OUT/layer_auroc.csv" \
    2>&1 | tee "$OUT/logs/sweep.log"

  if [[ ! -s "$OUT/layer_auroc.csv" ]]; then
    echo "sweep produced no table; check $OUT/logs/sweep.log" >&2
    exit 1
  fi
  touch "$OUT/.sweep.done"
fi

touch "$OUT/.done"
echo "done -> $OUT/layer_auroc.csv"
