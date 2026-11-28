# Worklog: cp_ce_half

## 2026-11-28, run 1 (launched)

```
Run:      Cross-entropy scorer on one 5,000-task half of the training data over 3 learning rates, then 3 seeds: the control for conformal training seeing only half of the labels in its classification loss
Command:  bash scripts/experiments/cp_ce_half/script.sh
Needs:    card 1 (5 processes), ~10 minutes; runs first in tmux session cp_card1, before cp_ce_fixed_threshold
Writes:   assets/experiments/cp_ce_half/{sweep,best.json,final,results.json,results.md}
Smoke:    passed, 24 traces (4 per split), 0.3 minutes, CPU only
Resume:   yes, skips finished stages and finished training runs
Decides:  None (control number)
```

- **Launched under** `Launch: auto`. Uses hidden state 32, selected by `cp_ce_full`. Duration is an estimate from the 4 minutes per 50-epoch run measured there.

## 2026-11-28, run 1 (finished)

- **What was run:** the run card above, unchanged; 16:28 to 16:33 on card 1, no restart.
- **Result:** `assets/experiments/cp_ce_half/results.md` and `results.json`. Selected learning rate 3e-4 (best epoch 6). Mean ± sd over 3 seeds, next to full-data cross-entropy from `cp_ce_full`:

| α | Test coverage | Test relative set size, half data | Test relative set size, full data | ProcessBench recalibrated relative set size, half data | Same, full data |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.951 ± 0.007 | 0.594 ± 0.028 | 0.532 ± 0.004 | 0.519 ± 0.014 | 0.435 ± 0.011 |
| 0.1 | 0.906 ± 0.001 | 0.451 ± 0.012 | 0.387 ± 0.016 | 0.374 ± 0.019 | 0.328 ± 0.017 |
| 0.2 | 0.796 ± 0.012 | 0.267 ± 0.012 | 0.251 ± 0.019 | 0.234 ± 0.010 | 0.221 ± 0.001 |

- **Top-1 hit rate:** 0.396 ± 0.005 (full data: 0.410 ± 0.014). Coverage sits inside the 95% interval at every level; no empty sets.
- **Insight:** halving the labelled training data costs about 0.06 in relative set size at α = 0.05 and 0.1 (roughly 1.2 steps per set), several times the seed spread, and little at α = 0.2. The scorer is data-limited at 5,000 to 10,000 traces.
- **Rules out:** that the label split is free. Any arm that puts its classification loss on 5,000 tasks starts about 0.06 behind full-data cross-entropy at α = 0.1.
- **For the campaign question:** conformal training has two bars at α = 0.1: 0.451 (same labels, this control) shows whether the set-size loss helps at all, and 0.387 (full-data cross-entropy) is the bar the campaign question asks about. It must recover the 0.06 lost to the split and more to count as a win.
