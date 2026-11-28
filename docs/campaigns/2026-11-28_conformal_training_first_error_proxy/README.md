# Campaign: conformal training on first-error reasoning traces (proxy for the synthetic corpus)

**Opened:** 2026-11-28   **Node:** D002   **Launch:** auto   **Status:** closed 2026-11-28

`Launch: auto` because [RESEARCHER] said on 2026-11-28, after agreeing the overview: "You can carry on with the experiments directly."

## Question

Does conformal training with a learned threshold give smaller calibrated prediction sets than plain cross-entropy training, at valid coverage, on first-error traces?

## Decision

Whether the Work Package 1 pipeline (`notes/workpackages/wp1.tex`) goes to the synthetic agentic corpus unchanged, or the loss is revised first.

## Done condition

Coverage and relative set size at α = 0.05, 0.1 and 0.2 over 3 seeds, for the conformal-trained scorer and each comparison arm, on the `actprm` held-out test split and on `processbench`.

## Kill bar

None ([RESEARCHER], 2026-11-28: "no kill bar, i want to see how good it can get").

## Experiments

One row per experiment. `after` names the experiments that must finish first; rows with nothing pending there can run in parallel on disjoint cards.

| Experiment | Answers | After | Cards | Status |
| ---------- | ------- | ----- | ----- | ------ |
| `cp_step_embeddings` | Builds the task-disjoint splits and caches frozen Qwen3.5-9B mean-pooled step embeddings at 25/50/75/100% depth for every trace used below | none | 2 | done → `worklogs/cp_step_embeddings.md` |
| `cp_ce_full` | Reference: scorer trained with cross-entropy on all 10,000 training tasks, then split conformal calibration. Picks the embedding depth on validation for all arms. Also reports the arms that need no retraining: temperature scaling, post-hoc scores (adaptive sets, regularised adaptive sets, length-normalised log-probability), and the random and position-prior floors | `cp_step_embeddings` | 1 | done → `worklogs/cp_ce_full.md` |
| `cp_ce_half` | Control for halved labels: the same cross-entropy scorer trained on one 5,000-task half only | `cp_ce_full` (for the depth) | 1 | done → `worklogs/cp_ce_half.md` |
| `cp_ce_fixed_threshold` | Value of learning the threshold: cross-entropy plus the soft set-size loss with a constant threshold tuned on validation | `cp_ce_full` (for the depth) | 1 | done (stopped early by [RESEARCHER] at 129 of 180 sweep runs; seed 0 only, no final seeds) → `worklogs/cp_ce_fixed_threshold.md` |
| `cp_dpsm` | The method: conformal training with a threshold learned by pinball updates (Algorithm 1 of the work package), over its sweep grid | `cp_ce_full` (for the depth) | 1 | done (stopped early by [RESEARCHER] at 90 of 180 sweep runs; seed 0 only, no final seeds) → `worklogs/cp_dpsm.md` |

## Approach

- **Method:** Work Package 1 as written, with `|Y| = 1` (the first error step) and one change: the frozen backbone is Qwen3.5-9B, not the work package's Qwen3-8B ([RESEARCHER], 2026-11-28: it is already on disk). Frozen Qwen3.5-9B hidden states are mean-pooled per step; a 2-layer Transformer encoder (width 256) scores steps with a softmax over the trace; the score is `1 − π_t`; evaluation is vanilla split conformal prediction, discarding any training threshold. Hyperparameters and checkpoints are selected on validation by calibrated relative set size at α = 0.1, identically for every arm; 3 seeds at the best configuration.
- **Data, `actprm`:** failed traces with 16 to 60 steps, one trace per task, split by `task_id` into 10,000 training (halved once into a classification-loss half and a set-size-loss half), 1,000 validation (halved internally to calibrate and score), 1,000 calibration, 1,000 held-out test. No balancing of error position. The cap of 60 drops 569 traces whose steps are a few characters each (broken step segmentation).
- **Data, `processbench`:** all 527 failed tasks with 10 or more steps, one trace per task, unbalanced. Evaluated recalibrated (200 random 50/50 calibration/test splits) and zero-shot with the `actprm` threshold as a shift diagnostic.
- **Shared code:** `src/utils/cp_scorer.py` (scorer), `cp_train.py` (the three training modes), `cp_eval.py` and `cp_report.py` (conformal evaluation), `cp_sweep.py` (sweep, selection, seeds), `cp_data.py` (cache loader).
- **Encoder details the work package leaves open:** pre-norm layers, GELU, rotary position encoding over step positions with base 10000. The fixed-threshold arm sweeps the threshold over 0.9, 0.95, 0.98, 0.99.
- **Setup ids (all new):** `dataset/actprm_first_error_16to60_v1`, `dataset/processbench_first_error_10plus_v1`, `model/qwen3_5_9b_frozen_v1`, `metric/split_cp_cov_ineff_v1`.
- **Guardrails reported with every number:** empty-set rate, full-set rate, top-1 hit rate, coverage by length tercile and by first-error position.
- **Left out by agreement (2026-11-28):** the in-batch differentiable-quantile alternative, the architecture and backbone ablations, and the external LLM-judge arm. Each can be added as a row later.
- **What could invalidate the result:**
  - `actprm` labels are, as far as we know, model-judged and not human; the files do not record their origin. Label noise hits all arms alike but lowers the ceiling.
  - "Earliest error" is not the counterfactual error set of the work package, and 23% of error-labelled `processbench` traces end with a correct final answer. A win supports the training algorithm, not the attribution claim.
  - No agents, no tool calls, and no trace long enough to trigger the overlapping-window path, so those parts of the pipeline stay untested.
  - Long traces skew to the hardest problems: the `processbench` subset is almost entirely olympiad-level and Omni-MATH.

## Closing summary

- **Outcome:** stopped by [RESEARCHER] because the two set-size sweeps, half done, showed no setting clearly ahead of the cross-entropy control. The done condition is met for the two cross-entropy arms (3 seeds) and only on seed 0 for the two set-size arms, which were stopped early at 90 and 129 of 180 sweep runs.
- **Result:** test relative set size on 1,000 held-out `actprm` tasks, with valid coverage in every arm. Cross-entropy on all data: 0.532 / 0.387 ± 0.016 / 0.251 at α = 0.05 / 0.1 / 0.2 over 3 seeds (`assets/experiments/cp_ce_full/results.md`). Cross-entropy on half the data: 0.594 / 0.451 ± 0.012 / 0.267 over 3 seeds (`assets/experiments/cp_ce_half/results.md`). Conformal training with a learned threshold: 0.562 / 0.424 / 0.257, seed 0, best of 90 finished runs (`assets/experiments/cp_dpsm/partial_results.md`). Cross-entropy plus set-size loss with a fixed threshold: 0.583 / 0.439 / 0.262, seed 0, best of 129 finished runs (`assets/experiments/cp_ce_fixed_threshold/partial_results.md`). On `processbench` after recalibration at α = 0.1: 0.328 (all data, 3 seeds), 0.374 (half data, 3 seeds), 0.361 (learned threshold, seed 0), 0.365 (fixed threshold, seed 0). Every arm is selected within the first 12 of 50 epochs and then overfits (`assets/experiments/cp_dpsm/plots/`).
- **Ruling (D002, worked):** Work Package 1 does not go to the synthetic corpus unchanged. On this proxy, conformal training with a learned threshold gives at most a small, unconfirmed reduction in set size over cross-entropy on the same data and none over cross-entropy on all the data; the scorer's overfitting and the threshold being estimated on memorised traces are fixed before the loss is judged.
- **Rules out:** post-hoc scores (temperature scaling, adaptive sets, regularised adaptive sets, length-normalised log-probability) as a way to shrink sets on this scorer (0.383 to 0.403 against 0.387 at α = 0.1); the early threshold lag as the sole cause of the weak effect (a step size of 0.005 removes most of the lag with no gain); a large gain from the set-size loss with a constant threshold on this setup; rerunning the same grid on the same scorer and data.
- **Left open:** whether conformal training helps once overfitting is slowed (stronger regularisation, a smaller scorer, or more `actprm` training traces); estimating the threshold on held-out traces instead of the classification half; threshold step sizes 0.01 and 0.05 and a warm start for the threshold, none of which ran; the in-batch differentiable-quantile variant, which uses all the data; under-coverage on the longest third of traces (0.869 against 0.908).
