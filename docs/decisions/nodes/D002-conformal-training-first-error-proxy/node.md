---
id: D002
title: Conformal training gives at most a small, unconfirmed set-size gain over same-data cross-entropy on first-error traces, and none over full-data cross-entropy; fix scorer overfitting first
parents: []
status: worked
campaign: docs/campaigns/2026-11-28_conformal_training_first_error_proxy
---

## Question

The research line is prediction sets over error steps with a coverage guarantee. Work Package 1 (`notes/workpackages/wp1.tex`) specifies conformal training of a small encoder over frozen step embeddings, but the synthetic agentic corpus it targets does not exist yet. This node tests the training algorithm first on single-error reasoning traces (`actprm` for training, calibration and test; `processbench` as an external test), where data is plentiful: does a learned-threshold set-size loss give smaller calibrated sets than cross-entropy at valid coverage? The answer decides whether the pipeline goes to the synthetic corpus unchanged or the loss is revised first.

No parent: this is the first real node (D001 is the template example).

Kill bar: None, by [RESEARCHER]'s call on 2026-11-28.

## Ruling

Work Package 1 does not go to the synthetic corpus unchanged. On this proxy, conformal training with a learned threshold gives at most a small, unconfirmed reduction in set size over cross-entropy on the same data and none over cross-entropy on all the data; the scorer's overfitting and the threshold being estimated on memorised traces are fixed before the loss is judged. Closed 2026-11-28 with the two set-size arms stopped early (seed 0 only); numbers and paths are in the campaign's closing summary.

## Rules out

post-hoc scores (temperature scaling, adaptive sets, regularised adaptive sets, length-normalised log-probability) as a way to shrink sets on this scorer (0.383 to 0.403 against 0.387 at α = 0.1); the early threshold lag as the sole cause of the weak effect (a step size of 0.005 removes most of the lag with no gain); a large gain from the set-size loss with a constant threshold on this setup; rerunning the same grid on the same scorer and data.

## Left open

whether conformal training helps once overfitting is slowed (stronger regularisation, a smaller scorer, or more `actprm` training traces); estimating the threshold on held-out traces instead of the classification half; threshold step sizes 0.01 and 0.05 and a warm start for the threshold, none of which ran; the in-batch differentiable-quantile variant, which uses all the data; under-coverage on the longest third of traces (0.869 against 0.908).
