# cp_ce_full

**Campaign:** `docs/campaigns/2026-11-28_conformal_training_first_error_proxy/` (node D002)

## What it does

Trains the step scorer with cross-entropy on all 10,000 training tasks, over four embedding depths and three learning rates, and applies split conformal prediction. It is the reference every other arm is compared with, it fixes the embedding depth for them, and it reports the arms that need no retraining: temperature scaling, the adaptive-set, regularised adaptive-set and length-normalised log-probability scores, and the random and error-position-prior floors (`posthoc.py`, `evaluate.py`).

## Setup

Named in `scripts/experiments/cp_ce_full/config.yaml`: datasets `actprm_first_error_16to60_v1` and `processbench_first_error_10plus_v1`, model `qwen3_5_9b_frozen_v1`, metric `split_cp_cov_ineff_v1`. Inputs are the cached step embeddings in `assets/experiments/cp_step_embeddings/emb/`.

## Stages

1. **sweep** (GPU, `src/utils/cp_sweep.py`): one training run per grid point with seed 0, each keeping the epoch with the lowest calibrated relative set size on validation at α = 0.1; then the best grid point is written to `best.json`.
2. **final** (GPU): the selected configuration with seeds 0, 1, 2.
3. **evaluate** (CPU): the reference arm, the no-retraining arms and the floors, each with split conformal calibration on the calibration split, coverage and set size on the held-out test split and on ProcessBench (recalibrated and zero-shot), at α = 0.05, 0.1, 0.2 (`src/utils/cp_report.py`).

```bash
SMOKE=1 bash scripts/experiments/cp_ce_full/script.sh             # rehearsal on the smoke embedding cache
GPUS=cpu SMOKE=1 bash scripts/experiments/cp_ce_full/script.sh    # the same without a card
bash scripts/experiments/cp_ce_full/script.sh                     # real run
```

Outputs go to `assets/experiments/cp_ce_full/`: `sweep/`, `best.json`, `final/`, `results.json`, `results.md`. Training, losses and evaluation live in `src/utils/cp_train.py`, `cp_scorer.py`, `cp_eval.py` and `cp_report.py`, tested in `src/utils/tests/test_cp_core.py`.
