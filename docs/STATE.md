# Current state

Read this first. Update it in the same step whenever a campaign opens or closes, a run starts or ends, or something starts waiting on [RESEARCHER]. Rewrite "Next step" at the end of every session.

## Active campaigns
<!-- one line each: folder · node id · what it waits on -->
- none

## Live runs
<!-- one line each: experiment · cards · tmux session · output folder · started YYYY-MM-DD -->
- none

## Waiting on [RESEARCHER]
- A commit of campaign D002 (code, setups, docs); everything is still uncommitted.
- `sudo apt install python3.12-dev` (safe now: `cp_step_embeddings` finished 2026-11-28). `flash-linear-attention` is installed (2026-11-28) but its Triton kernels cannot compile without `Python.h`, so Qwen3.5-9B still runs its linear-attention layers on the slow PyTorch fallback. Do not install it while an embedding run is live or half-finished: the fast kernel differs from the fallback by about 0.1–0.5% and the two must not be mixed in one cache.

## Next step
<!-- what the next session should pick up, in a few lines -->
- Campaign D002 (conformal training on first-error traces) closed 2026-11-28 as `worked`: Work Package 1 does not go to the synthetic corpus unchanged; scorer overfitting and the threshold being estimated on memorised traces come first. The node's "Left open" list holds the candidate follow-ups (slow the overfitting, held-out threshold estimate, larger threshold step sizes or a warm start, the in-batch quantile variant); none is agreed, and any of them needs a new campaign.
- Qwen3-8B was never downloaded: `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B` holds about 100 MB of a download stopped on 2026-11-28 when [RESEARCHER] chose Qwen3.5-9B. It can be deleted.
- The example campaign, node `D001`, example setups and `probe_layer_sweep` are template references; now that a real campaign exists they can be deleted, on [RESEARCHER]'s word.
- `notes/far-survey/scripts/` lacks the survey's build helpers (see its README), so `notes/far-survey/build.sh` cannot run until they are restored.
