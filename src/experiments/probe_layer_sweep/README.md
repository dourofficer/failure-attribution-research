# probe_layer_sweep

> **Example of an experiment README. Delete with the rest of the example.**

**Campaign:** `docs/campaigns/2026-01-15_probe_layer_sweep/` (node D001)

## What it does

Measures the probe's AUROC at each of the 32 hidden layers, to test the default of reading the final layer.

## Setup

Named in `scripts/experiments/probe_layer_sweep/config.yaml`:

- dataset `example_dev_v1`: 4,000 dev examples, grouped by `source_id`
- model `example_lm_v1`: 32 layers, bf16, `flash_attention_2`
- metric `auroc_grouped5_v1`: logistic probe, 5-fold `GroupKFold`

## Stages

1. **extract** (GPU): one forward pass per example, all layers captured, sharded 2 processes per card.
2. **sweep** (CPU): one probe per layer; folds grouped by source document so that no document is in both train and test.

```bash
SMOKE=1 bash scripts/experiments/probe_layer_sweep/script.sh   # rehearsal
bash scripts/experiments/probe_layer_sweep/script.sh           # real run
bash scripts/experiments/probe_layer_sweep/status.sh           # progress
```

Outputs go to `assets/experiments/probe_layer_sweep/`. Results and their meaning are in the campaign worklog.

## Tests

```bash
uv run pytest src/experiments/probe_layer_sweep/tests/ -q
```

`tests/test_metrics.py` compares the fast AUROC with the slow pairwise definition, checks hand-derivable cases and invariants (label flip, monotone transform, permutation), and pins the sign convention of the mean-difference direction.
