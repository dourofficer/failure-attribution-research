---
id: D001
title: EXAMPLE. Which layer should the probe read? Layer 16, not the final layer
parents: []
status: worked
campaign: docs/campaigns/2026-01-15_probe_layer_sweep
---

> **Example of a decision node. Delete it once the project has real ones.**

## Question

Written by `/campaign` when the node opened.

The probe reads the final hidden layer by default, on the assumption that late layers carry the most signal. Nothing had tested that. If the signal peaks earlier, every later result is read off the wrong layer, so this had to be settled before building on the probe.

Kill bar agreed in advance: if no layer beats the final layer by more than 0.02 AUROC, keep the default.

## Ruling

Written by `/close`.

Read the probe at layer 16 from here on. Across all 32 layers, 5-fold grouped by source document:

| Layer | AUROC |
| ----- | ----- |
| 8  | 0.71 |
| 16 | **0.84** |
| 24 | 0.79 |
| 32 (final) | 0.68 |

Evidence: `assets/experiments/probe_layer_sweep/layer_auroc.csv`, worklog `worklogs/probe_layer_sweep.md` in the campaign folder.

## Rules out

- "Later layers are better." Do not default to the final layer again.
- A finer sweep near the final layers.

## Left open

Whether layer 16 holds on other models. That needs its own campaign.
