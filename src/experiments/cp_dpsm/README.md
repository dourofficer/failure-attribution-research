# cp_dpsm

**Campaign:** `docs/campaigns/2026-11-28_conformal_training_first_error_proxy/` (node D002)

## What it does

Conformal training as in Algorithm 1 of Work Package 1: cross-entropy on one half of the training data, the soft set-size loss on the other half, and a threshold moved by a pinball step after every batch towards the (1 − α) quantile of the label scores. Each epoch logs the threshold next to the exact quantile on the classification half, to check that the threshold keeps up. It has no code of its own: the config selects the mode in the shared trainer.

## Setup

Named in `scripts/experiments/cp_dpsm/config.yaml`: datasets `actprm_first_error_16to60_v1` and `processbench_first_error_10plus_v1`, model `qwen3_5_9b_frozen_v1`, metric `split_cp_cov_ineff_v1`. Inputs are the cached step embeddings in `assets/experiments/cp_step_embeddings/emb/`.

## Stages

1. **sweep** (GPU, `src/utils/cp_sweep.py`): one training run per grid point with seed 0, each keeping the epoch with the lowest calibrated relative set size on validation at α = 0.1; then the best grid point is written to `best.json`.
2. **final** (GPU): the selected configuration with seeds 0, 1, 2.
3. **evaluate** (CPU): Split conformal calibration on the calibration split, coverage and set size on the held-out test split and on ProcessBench (recalibrated and zero-shot), at α = 0.05, 0.1, 0.2 (`src/utils/cp_report.py`).

```bash
SMOKE=1 bash scripts/experiments/cp_dpsm/script.sh             # rehearsal on the smoke embedding cache
GPUS=cpu SMOKE=1 bash scripts/experiments/cp_dpsm/script.sh    # the same without a card
bash scripts/experiments/cp_dpsm/script.sh                     # real run
```

Outputs go to `assets/experiments/cp_dpsm/`: `sweep/`, `best.json`, `final/`, `results.json`, `results.md`. Training, losses and evaluation live in `src/utils/cp_train.py`, `cp_scorer.py`, `cp_eval.py` and `cp_report.py`, tested in `src/utils/tests/test_cp_core.py`.
