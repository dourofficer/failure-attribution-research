# cp_step_embeddings

**Campaign:** `docs/campaigns/2026-11-28_conformal_training_first_error_proxy/` (node D002)

## What it does

Builds the task-disjoint splits of failed, first-error-labelled reasoning traces and caches frozen Qwen3.5-9B step embeddings for them, so that every scorer in the campaign trains and is evaluated on the same cached inputs.

## Setup

Named in `scripts/experiments/cp_step_embeddings/config.yaml`:

- dataset `actprm_first_error_16to60_v1`: failed ActPRM traces with 16 to 60 steps, one per task; 1,000 test, 1,000 calibration, 1,000 validation, and two training halves of 5,000 tasks.
- dataset `processbench_first_error_10plus_v1`: every failed ProcessBench task with 10 or more steps, one trace per task, as the external test.
- model `qwen3_5_9b_frozen_v1`: the text decoder of Qwen3.5-9B (local copy) in bf16, hidden states at 25/50/75/100% depth, mean-pooled over the tokens of each step.

## Stages

1. **splits** (CPU, `splits.py`): filter, keep one trace per task, order tasks by a seeded hash and cut them into splits.
2. **extract** (GPU, `extract.py`): one forward pass per trace, one process per card, chunks of 256 traces; a chunk on disk is skipped. `pooling.py` holds the trace formatting, token-to-step assignment and pooling.
3. **verify** (CPU, `verify.py`): checks every chunk against the splits and writes `manifest.json`.

```bash
SMOKE=1 bash scripts/experiments/cp_step_embeddings/script.sh   # rehearsal, a few tasks per split
bash scripts/experiments/cp_step_embeddings/script.sh           # real run
bash scripts/experiments/cp_step_embeddings/status.sh           # progress
uv run pytest src/experiments/cp_step_embeddings -q             # tests
```

Outputs go to `assets/experiments/cp_step_embeddings/`: `splits/<dataset_id>.jsonl`, `emb/chunk*.npz` (step embeddings of all traces concatenated in split order, one array per kept layer, float16) and `manifest.json`. A trace longer than the model setup's `max_tokens` stops the run: the overlapping-window path of the work package is not implemented, because no trace here needs it.
