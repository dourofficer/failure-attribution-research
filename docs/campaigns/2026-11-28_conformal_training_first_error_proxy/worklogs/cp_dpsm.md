# Worklog: cp_dpsm

## 2026-11-28, run 1 (launched)

```
Run:      Conformal training with a learned threshold (Algorithm 1): 180 sweep runs (3 learning rates x 5 weights x 3 temperatures x 4 threshold step sizes), then 3 seeds
Command:  bash scripts/experiments/cp_dpsm/script.sh
Needs:    card 0 (5 processes), ~4-8 hours; tmux session cp_dpsm
Writes:   assets/experiments/cp_dpsm/{sweep,best.json,final,results.json,results.md}
Smoke:    passed, 24 traces (4 per split), 0.3 minutes, CPU only
Resume:   yes, skips finished stages and finished training runs
Decides:  None (no kill bar); the campaign question is settled by its test relative set size against 0.387 +/- 0.016 for full-data cross-entropy at alpha = 0.1
```

- **Launched under** `Launch: auto`. Uses hidden state 32, selected by `cp_ce_full`. Duration is an estimate from the 4 minutes per 50-epoch run measured there.

## 2026-11-28, run 1 (stopped by [RESEARCHER] at 18:58, sweep half done)

- **What was run:** 90 of 180 sweep runs finished (seed 0, 50 epochs each): all 45 settings at threshold step size γ = 0.001 and all 45 at γ = 0.005. γ = 0.01 and 0.05 never ran, and neither did the 3-seed final stage. The tmux session was killed on [RESEARCHER]'s request, so the folder has no `.done` and no `.failed`; rerunning the script would resume from run 91.
- **Result (validation, α = 0.1, seed 0):** best relative set size 0.427 (lr 3e-4, λ = 1, τ = 0.05, γ = 0.001, epoch 2), median over the 90 runs 0.441, worst 0.464. The same-data cross-entropy control is 0.434 and full-data cross-entropy is 0.388. By step size: γ = 0.001 best 0.427, median 0.439; γ = 0.005 best 0.428, median 0.444. Path: `assets/experiments/cp_dpsm/sweep/`.
- **Result (test, seed 0, best finished run, `assets/experiments/cp_dpsm/partial_results.md`):** written by `src/utils/cp_partial.py`, which applies the selection rule of `cp_sweep select` to the runs on disk. Seed-0 rows for the two cross-entropy arms are their sweep-selected runs, so all four rows are like for like. External is processbench after recalibration.

| Arm (seed 0) | Test coverage at α = 0.1 | Test relative set size at α = 0.05 / 0.1 / 0.2 | External relative set size at α = 0.05 / 0.1 / 0.2 | Top-1 |
| --- | --- | --- | --- | --- |
| Conformal training, learned threshold (best of 90) | 0.894 | 0.562 / 0.424 / 0.257 | 0.470 / 0.361 / 0.220 | 0.393 |
| Cross-entropy + set-size loss, fixed threshold (best of 129) | 0.904 | 0.583 / 0.439 / 0.262 | 0.457 / 0.365 / 0.225 | 0.384 |
| Cross-entropy, half data | 0.905 | 0.634 / 0.459 / 0.278 | 0.539 / 0.397 / 0.246 | 0.397 |
| Cross-entropy, all data | 0.899 | 0.531 / 0.393 / 0.242 | 0.421 / 0.330 / 0.222 | 0.419 |

- **Figures (`assets/experiments/cp_dpsm/plots/`, drawn by `src/utils/cp_plots.py`):** `training_curves.png` (best run of each of the four arms: classification loss, set-size loss, threshold, validation top-1, validation relative set size, validation coverage), `all_runs.png` (validation relative set size per epoch for every finished run), `sweep_sensitivity.png` (best validation relative set size of every run by setting), `threshold_tracking.png` (learned q against the exact quantile at γ = 0.001 and 0.005), `loss_and_q_best_run.png` (the best run alone).
- **Insight 1, a small gain over the same-data control, not established:** on seed 0 the learned-threshold arm is 0.035 below half-data cross-entropy at α = 0.1 on test (0.424 against 0.459) and 0.072 below at α = 0.05. The control's seed spread is ±0.012 at α = 0.1 and ±0.028 at α = 0.05, and this is the best of 90 settings on one seed, so the gain is suggestive only. On validation the gain is 0.007 and the median setting is worse than the control.
- **Insight 2, it does not recover the cost of halving the data:** full-data cross-entropy is better at every α on test (0.393 against 0.424 at α = 0.1).
- **Insight 3, the learned threshold does not track its target in the epochs that matter:** q starts at 0.5 while the exact 0.9 quantile of the true-step scores is about 0.97 from the first epoch. At γ = 0.001, q is 0.80 at the selected epoch (2), so the set-size loss is 0.01 to 0.08 against a classification loss near 2 and the selected model is close to plain cross-entropy. At γ = 0.005, q reaches 0.97 within one epoch, and validation set size does not improve (best 0.428).
- **Insight 4, the target itself collapses:** the quantile is estimated on the half the classifier is trained on. From about epoch 8 the scorer memorises that half (classification loss 0.15 at epoch 20, 0.006 at epoch 49), the exact quantile falls to 0 by epoch 25, and q follows it down. After that q describes memorisation and not held-out scores.
- **Insight 5, overfitting dominates every arm:** validation relative set size is best at epochs 2 to 9 in all 219 finished runs of the two set-size arms and rises to a median of 0.65 by epoch 49 (cross-entropy alone: 0.62). Validation top-1 falls from about 0.40 to 0.36 over the same span. The set-size loss therefore acts for a few hundred updates only before the selected checkpoint.
- **Insight 6, no setting matters much:** medians by learning rate, λ, τ and γ all lie between 0.439 and 0.447; λ = 5 is the only value visibly worse.
- **What it rules out:** that the early threshold lag alone explains the weak effect (γ = 0.005 removes most of the lag with no gain). It does not rule out conformal training as a method: the scorer overfits before the method can act, and γ = 0.01 and 0.05 were not run.
- **For the campaign's question:** on this proxy and with this scorer, conformal training gives at most a small reduction in set size over cross-entropy on the same data and none over cross-entropy on all the data. The test is not yet a fair one, because all arms are early-stopped at epochs 2 to 9. The departure from the intended method to fix first is the overfitting (and the threshold being estimated on memorised traces), not the loss.
- **Correction to the entry above (checked against the run files):** the selected epoch ranges from 1 to 12 in the 90 learned-threshold runs and from 1 to 9 in the 129 fixed-threshold runs, not "2 to 9 in all 219". Median validation top-1 goes from 0.39 at the selected epoch to 0.35 at epoch 49, not "0.40 to 0.36". The conclusions are unchanged.
