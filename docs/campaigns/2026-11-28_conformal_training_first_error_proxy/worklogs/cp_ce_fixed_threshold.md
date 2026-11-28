# Worklog: cp_ce_fixed_threshold

## 2026-11-28, run 1 (launched)

```
Run:      Cross-entropy plus the soft set-size loss with a constant threshold: 180 sweep runs (3 learning rates x 5 weights x 3 temperatures x 4 thresholds), then 3 seeds
Command:  bash scripts/experiments/cp_ce_fixed_threshold/script.sh
Needs:    card 1 (5 processes), ~3-5 hours; runs second in tmux session cp_card1, after cp_ce_half
Writes:   assets/experiments/cp_ce_fixed_threshold/{sweep,best.json,final,results.json,results.md}
Smoke:    passed, 24 traces (4 per split), 0.3 minutes, CPU only
Resume:   yes, skips finished stages and finished training runs
Decides:  None; compared with the learned-threshold arm at alpha = 0.05, 0.1, 0.2
```

- **Launched under** `Launch: auto`. Uses hidden state 32, selected by `cp_ce_full`. Duration is an estimate from the 4 minutes per 50-epoch run measured there.

## 2026-11-28, run 1 (stopped by [RESEARCHER] at 18:58, sweep 129 of 180)

- **What was run:** 129 of 180 sweep runs finished (seed 0, 50 epochs each): λ = 0.1, 0.5 and 1 complete, λ = 2 partly, λ = 5 not at all; every fixed threshold (0.9, 0.95, 0.98, 0.99) has 30 or more runs. The 3-seed final stage never ran. The tmux session was killed on [RESEARCHER]'s request, so the folder has no `.done` and no `.failed`; rerunning the script would resume.
- **Result (validation, α = 0.1, seed 0):** best relative set size 0.416 (lr 3e-4, λ = 0.5, τ = 0.01, threshold 0.98, epoch 5), median 0.440, worst 0.461. By threshold: 0.9 best 0.431, median 0.442; 0.95 best 0.423, median 0.439; 0.98 best 0.416, median 0.440; 0.99 best 0.421, median 0.440. Path: `assets/experiments/cp_ce_fixed_threshold/sweep/`.
- **Result (test, seed 0, best finished run, `assets/experiments/cp_ce_fixed_threshold/partial_results.md`):** coverage 0.943 / 0.904 / 0.800 and relative set size 0.583 / 0.439 / 0.262 at α = 0.05 / 0.1 / 0.2; processbench after recalibration 0.457 / 0.365 / 0.225; top-1 0.384. The four-arm table is in `worklogs/cp_dpsm.md`, and the figures for all arms are in `assets/experiments/cp_dpsm/plots/`.
- **Insight:** the best validation number (0.416) does not hold on test (0.439), which is what selecting the best of 129 single-seed runs on 1,000 validation traces would produce. On test it sits 0.020 below half-data cross-entropy at α = 0.1 (seed 0, 0.459) and 0.015 above the learned-threshold arm (0.424); both gaps are within about two seed standard deviations (±0.012).
- **Insight:** the value of the fixed threshold barely matters (medians 0.439 to 0.442), and the selected epoch is 2 to 9 in every run, as in the other arms.
- **What it rules out:** a large gain from the set-size loss with a well-chosen constant threshold on this setup. It does not separate a learned from a fixed threshold: the difference between the two arms is within seed noise, on one seed.
- **For the campaign's question:** the set-size loss, with or without a learned threshold, gives a small and unconfirmed gain over same-data cross-entropy and none over full-data cross-entropy; see `worklogs/cp_dpsm.md` for why the comparison is not yet fair.
- **Correction to the entry above:** the selected epoch ranges from 1 to 9 in the 129 runs, not 2 to 9.
