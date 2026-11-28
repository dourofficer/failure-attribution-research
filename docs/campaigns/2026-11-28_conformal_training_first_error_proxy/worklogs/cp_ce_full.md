# Worklog: cp_ce_full

## 2026-11-28, run 1 (launched)

```
Run:      Cross-entropy scorer on all 10,000 training tasks over 4 embedding depths x 3 learning rates, then 3 seeds at the best; the reference arm, which also fixes the depth for the other arms and yields the no-retraining arms and floors
Command:  bash scripts/experiments/cp_ce_full/script.sh
Needs:    2 cards (3 processes each, ~3 GB per process), duration not yet measured (12 + 3 short training runs)
Writes:   assets/experiments/cp_ce_full/{sweep,best.json,final,results.json,results.md}
Smoke:    passed, 24 traces (4 per split), 0.3 minutes, CPU only (cards were busy)
Resume:   yes, skips finished stages and finished training runs
Decides:  None (reference numbers); selection is by calibrated relative set size on validation at alpha = 0.1
```

- **Launched under** `Launch: auto`, tmux session `cp_ce_full`.

## 2026-11-28, run 1 (finished)

- **What was run:** the run card above, unchanged; 16:10 to 16:27 on cards 0 and 1 (6 processes), no restart. Each 50-epoch training run took about 4 minutes.
- **Result:** `assets/experiments/cp_ce_full/results.md` and `results.json`; sweep table in `best.json`. Selected: final-layer embeddings (hidden state 32), learning rate 1e-4. Mean ± sd over 3 seeds, calibration and test of 1,000 tasks each:

| α | Test coverage | 95% interval for coverage | Test relative set size | Test set size (steps) | ProcessBench recalibrated coverage | ProcessBench relative set size | ProcessBench zero-shot coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.942 ± 0.003 | 0.936–0.963 | 0.532 ± 0.004 | 9.86 | 0.949 | 0.435 ± 0.011 | 0.985 |
| 0.1 | 0.895 ± 0.010 | 0.881–0.918 | 0.387 ± 0.016 | 7.11 | 0.900 | 0.328 ± 0.017 | 0.971 |
| 0.2 | 0.807 ± 0.020 | 0.775–0.824 | 0.251 ± 0.019 | 4.56 | 0.804 | 0.221 ± 0.001 | 0.925 |

- **Top-1 hit rate:** 0.410 ± 0.014 on the ActPRM test split, 0.487 on ProcessBench. No empty and no full sets.
- **Depth:** relative set size on validation at α = 0.1 improves with depth: about 0.53 at hidden state 8, 0.42–0.45 at 16, 0.41–0.43 at 24, 0.39–0.41 at 32. All other arms use hidden state 32.
- **Arms without retraining, α = 0.1, test relative set size:** temperature scaling 0.394 ± 0.018, length-normalised log-probability score 0.383 ± 0.021, adaptive-set score 0.403 ± 0.022, regularised adaptive-set score 0.392 ± 0.015. None differs from the reference (0.387 ± 0.016) by more than the seed spread.
- **Floors, α = 0.1, test relative set size:** random scores 0.894, error-position prior 0.857 (with 3.3% empty sets). The scorer cuts the set to well under half of either floor, so it is not reading position alone.
- **Guardrails, α = 0.1:** coverage by length tercile 0.908 / 0.900 / 0.869 (shortest to longest), by error-position tercile 0.918 / 0.877 / 0.890. The longest traces are under-covered by about 3 points, with smaller relative sets (0.326 against 0.416 for the shortest).
- **Insight, overfitting:** the selected run is best at epoch 3 of 50. Training loss goes from 2.17 to 0.01 while validation relative set size goes 0.469 → 0.388 (epoch 3) → 0.624 (epoch 49). Every sweep run peaks by epoch 9. The scorer memorises 10,000 traces, so what the comparison between arms measures is each arm's best early checkpoint.
- **Insight, shift:** applying the ActPRM threshold to ProcessBench over-covers (0.971 at α = 0.1) because the scorer is more accurate there (top-1 0.487), so zero-shot sets are valid but larger than needed; recalibrating gives 0.900 with smaller sets (0.328).
- **Rules out:** that post-hoc score design or temperature scaling closes any gap on this scorer; that a shallower embedding depth is better; that the scorer is only a position prior.
- **For the campaign question:** the number to beat is test relative set size 0.387 ± 0.016 at α = 0.1 (0.532 at 0.05, 0.251 at 0.2), with seed spread of about 0.02. A conformal-training gain smaller than that spread will not be distinguishable with 3 seeds.
