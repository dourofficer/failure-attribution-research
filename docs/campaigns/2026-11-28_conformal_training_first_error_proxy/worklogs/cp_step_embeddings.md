# Worklog: cp_step_embeddings

## 2026-11-28, run 1 (launched)

```
Run:      Build the task-disjoint splits and cache frozen Qwen3.5-9B step embeddings (4 depths) for 13,000 ActPRM and 527 ProcessBench traces, the shared input of every scorer in the campaign
Command:  bash scripts/experiments/cp_step_embeddings/script.sh
Needs:    2 cards (RTX 3090, one process each), ~1 hour
Writes:   assets/experiments/cp_step_embeddings/{splits,emb,manifest.json}
Smoke:    passed, 24 traces (4 per split), 0.6 minutes, 1 card
Resume:   yes, skips finished stages and finished chunks of 256 traces
Decides:  None (an input for the other experiments); complete when manifest.json lists 13,527 traces
```

- **Launched under** `Launch: auto`, tmux session `cp_step_embeddings`.
- **Backbone:** Qwen3.5-9B from `/home/thanhdo/hub/Qwen/Qwen3.5-9B`, text decoder only, not the work package's Qwen3-8B ([RESEARCHER], 2026-11-28). It has 32 layers, so the kept depths are hidden states 8, 16, 24 and 32.
- **Speed:** about 2 traces per second per card in the smoke run. The model's linear-attention layers fall back to reference PyTorch kernels because `causal_conv1d` and `flash-linear-attention` are not installed; the fallback is correct, only slower.
- **Smoke checks:** all four depths finite in float16; mean step-embedding norm 17 / 31 / 68 / 76 by depth; adjacent-step cosine in one trace falls from 0.96 at the shallowest depth to 0.71 at the deepest, so steps are distinguishable. A second smoke launch skipped every stage.
- **Split dry run at full scale:** 338,922 failed ActPRM traces seen; 303,924 dropped as shorter than 16 steps, 569 as longer than 60, 2 for a blank step; 32,778 tasks remain, 13,000 used, 19,778 in reserve. ProcessBench: 527 tasks.

## 2026-11-28, run 1 (finished)

- **What was run:** the run card above, unchanged; 15:00 to about 16:10 on cards 0 and 1, no restart.
- **Result:** `assets/experiments/cp_step_embeddings/manifest.json`. All 13,527 traces are cached and verified against the splits (ids, lengths, labels, finite values): 5,000 + 5,000 training, 1,000 validation, 1,000 calibration, 1,000 test from ActPRM, and 527 from ProcessBench; 236,648 steps in total; 4 depths (hidden states 8, 16, 24, 32), width 4,096, float16, 7.9 GB in 53 chunks. Formatted traces have a median of 1,303 tokens and a maximum of 5,703.
- **Insight:** throughput was 1.6 to 1.7 traces per second per card on the fallback kernels, so a rebuild costs about 70 minutes on two cards.
- **Rules out:** the need for the overlapping-window path on this data; the longest trace is about a third of the 16,384-token limit.
- **For the campaign question:** nothing yet; this is the shared input. Every scorer arm now trains and is evaluated on identical cached embeddings, so differences between arms cannot come from the backbone pass.
- **Caveat for any rebuild:** this cache was made with the PyTorch fallback for the linear-attention layers. `flash-linear-attention` has since been added to the environment and activates once `python3.12-dev` is installed; its output differs from the fallback by about 0.1–0.5%, so a partial rebuild must not be mixed into this cache.
