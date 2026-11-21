# Current state

Read this first. Update it in the same step whenever a campaign opens or closes, a run starts or ends, or something starts waiting on [RESEARCHER]. Rewrite "Next step" at the end of every session.

## Active campaigns
<!-- one line each: folder · node id · what it waits on -->
- none

## Live runs
<!-- one line each: experiment · cards · tmux session · output folder · started YYYY-MM-DD -->
- none

## Waiting on [RESEARCHER]
- none

## Next step
<!-- what the next session should pick up, in a few lines -->
- No campaign is open yet. The repo was moved onto this workflow on 2026-11-21; the example campaign, node `D001`, example setups and `probe_layer_sweep` are template references to delete once a real campaign exists.
- Candidate first line, not yet agreed: prediction sets over error steps with a coverage guarantee (conformal), per `notes/proposal/`. It needs a discussion to settle the four items before `/campaign`.
- `notes/scripts/` lacks the survey's build helpers (see its README), so `notes/build.sh` cannot run until they are restored.
