# Campaign: probe layer sweep

> **Example of a campaign. Delete once the project has real ones.**

**Opened:** 2026-01-15   **Node:** D001   **Launch:** ask   **Status:** closed 2026-01-16

## Question

Which hidden layer carries the most signal for the probe? We read the final layer by default, on assumption and not on evidence.

## Decision

Which layer every later probe experiment reads from. If a mid-depth layer wins, results already computed on the final layer must be re-run before they are quoted.

## Done condition

An AUROC-by-layer table over all 32 layers, 5-fold grouped by source document, at `assets/experiments/probe_layer_sweep/layer_auroc.csv`.

## Kill bar

If no layer beats the final layer by more than **0.02 AUROC**, keep the default, close the line, and do not revisit it with a finer sweep.

## Experiments

One row per experiment. `after` names the experiments that must finish first; rows with nothing pending there can run in parallel on disjoint cards.

| Experiment | Answers | After | Cards | Status |
| ---------- | ------- | ----- | ----- | ------ |
| `probe_layer_sweep` | AUROC of the probe at each of the 32 layers | none | all free | done → `worklogs/probe_layer_sweep.md` |

## Approach

Extract hidden states once for all layers (one forward pass per example, sharded across cards), then sweep the layers on CPU. Folds are grouped by source document, because examples from one document share vocabulary and would leak across folds.

## Closing summary

Written by `/close` on 2026-01-16.

- **Outcome:** done condition met; kill bar not triggered.
- **Result:** AUROC peaks at layer 16 (0.84) against 0.68 at the final layer, a gap of +0.16.
- **Ruling (D001, worked):** read the probe at layer 16 from here on.
- **Rules out:** the "later layers are better" assumption, and any finer sweep near the final layers.
- **Left open:** whether the peak holds on other models. That is a new campaign.
