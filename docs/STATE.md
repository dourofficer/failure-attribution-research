# Current state

Read this first. Update it in the same step whenever a campaign opens or closes, a run starts or ends, or something starts waiting on [RESEARCHER]. Rewrite "Next step" at the end of every session.

## Active campaigns
<!-- one line each: folder · node id · what it waits on -->
- `docs/campaigns/2026-11-28_conformal_training_first_error_proxy` · D002 · `Launch: auto`; `cp_step_embeddings`, `cp_ce_full` and `cp_ce_half` done; `cp_dpsm` and `cp_ce_fixed_threshold` stopped mid-sweep by [RESEARCHER] on 2026-11-28; waits on [RESEARCHER] for what to run next

## Live runs
<!-- one line each: experiment · cards · tmux session · output folder · started YYYY-MM-DD -->
- none

## Waiting on [RESEARCHER]
- Campaign D002: what to run next after the stopped sweeps (see Next step), and a commit of the campaign's code, setups and docs (still uncommitted).
- `sudo apt install python3.12-dev` (safe now: `cp_step_embeddings` finished 2026-11-28). `flash-linear-attention` is installed (2026-11-28) but its Triton kernels cannot compile without `Python.h`, so Qwen3.5-9B still runs its linear-attention layers on the slow PyTorch fallback. Do not install it while an embedding run is live or half-finished: the fast kernel differs from the fallback by about 0.1–0.5% and the two must not be mixed in one cache.

## Next step
<!-- what the next session should pick up, in a few lines -->
- Campaign D002 (conformal training on first-error traces, `Launch: auto`): both set-size arms were stopped mid-sweep on 2026-11-28 (90 and 129 of 180 runs, seed 0 only, no final seeds); findings and figures are in `worklogs/cp_dpsm.md` and `assets/experiments/cp_dpsm/plots/`. On seed 0 the learned-threshold arm reaches test relative set size 0.424 at α = 0.1 against 0.459 for same-data cross-entropy and 0.393 for full-data cross-entropy: a small, unconfirmed gain. Every arm overfits and is selected within the first 12 of 50 epochs, and the learned threshold is estimated on traces the scorer memorises, so the comparison is not yet fair. Open choices for [RESEARCHER]: add an experiment that slows the overfitting (stronger regularisation, smaller scorer, or more actprm training traces) before re-running the two arms, resume the sweeps as they are, or `/close`.
- Qwen3-8B was never downloaded: `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B` holds about 100 MB of a download stopped on 2026-11-28 when [RESEARCHER] chose Qwen3.5-9B. It can be deleted.
- The example campaign, node `D001`, example setups and `probe_layer_sweep` are template references; now that a real campaign exists they can be deleted, on [RESEARCHER]'s word.
- `notes/far-survey/scripts/` lacks the survey's build helpers (see its README), so `notes/far-survey/build.sh` cannot run until they are restored.
