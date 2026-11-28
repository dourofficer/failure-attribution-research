---
id: D002
title: Does conformal training with a learned threshold shrink calibrated step sets against cross-entropy on first-error traces?
parents: []
status: open
campaign: docs/campaigns/2026-11-28_conformal_training_first_error_proxy
---

## Question

The research line is prediction sets over error steps with a coverage guarantee. Work Package 1 (`notes/workpackages/wp1.tex`) specifies conformal training of a small encoder over frozen step embeddings, but the synthetic agentic corpus it targets does not exist yet. This node tests the training algorithm first on single-error reasoning traces (`actprm` for training, calibration and test; `processbench` as an external test), where data is plentiful: does a learned-threshold set-size loss give smaller calibrated sets than cross-entropy at valid coverage? The answer decides whether the pipeline goes to the synthetic corpus unchanged or the loss is revised first.

No parent: this is the first real node (D001 is the template example).

Kill bar: None, by [RESEARCHER]'s call on 2026-11-28.
