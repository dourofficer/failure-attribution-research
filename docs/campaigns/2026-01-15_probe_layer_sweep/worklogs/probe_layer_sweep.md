# Worklog: probe_layer_sweep

> **Example of a worklog. Delete with the rest of the example campaign.** One file per experiment, append-only: each run adds an entry with the sections below, pass or fail.

## 2026-01-15: full sweep, all 32 layers

**Status:** done

### Run card

```
Run:      probe AUROC for all 32 layers, to test the final-layer default
Command:  bash scripts/experiments/probe_layer_sweep/script.sh
Needs:    8 cards (2 procs/card), ~20 min GPU + ~5 min CPU
Writes:   assets/experiments/probe_layer_sweep/{hidden/,layer_auroc.csv,logs/}
Smoke:    passed, 16 items, 3 min, 8 cards
Resume:   yes, skips finished shards and stages
Decides:  best layer beats final layer by > 0.02 AUROC -> switch; else close
```

Approved by [RESEARCHER] on 2026-01-15.

### What was run

```bash
SMOKE=1 bash scripts/experiments/probe_layer_sweep/script.sh   # rehearsal, passed
bash scripts/experiments/probe_layer_sweep/script.sh           # real run, in tmux
```

Setup: dataset `example_dev_v1`, model `example_lm_v1`, metric `auroc_grouped5_v1`. The exact values are in `assets/experiments/probe_layer_sweep/config.resolved.yaml`. Extraction took about 18 minutes on 8 cards.

### Result

AUROC peaks at **layer 16 (0.84)**, against **0.68** at the final layer, a gap of +0.16. The curve is a single hump: 0.71 at layer 8 and 0.79 at layer 24.

Table: `assets/experiments/probe_layer_sweep/layer_auroc.csv`

### Insight

The signal sits at mid-depth. The fold spread at layer 16 is ±0.02, so the +0.16 gap does not come from one lucky fold.

### Rules out

- The "later layers are better" assumption behind the final-layer default.
- A finer sweep near the final layers: that region is clearly worse, not slightly worse.

It does **not** rule out the peak being specific to this model. Only one model was swept.

### For the campaign's question

Answers it: layer 16. The gap clears the kill bar (0.02) by a wide margin. Results already computed on the final layer are understated and need a re-run before being quoted.
