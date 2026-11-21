# Project Guidelines

Rules for agents working on this research project on **failure attribution**. [RESEARCHER] runs the research; you help decide what is worth running, run it carefully, and report what it means. The human-facing guide is `MANUAL.md`.

## Read first, every session

1. `docs/STATE.md`: active campaigns, live runs, what waits on [RESEARCHER], the next step.
2. `docs/decisions/index.md`: the decision graph. Never propose a line that a node has already closed.

`docs/ideas/` holds free-form ideas, none of them agreed. `notes/` is [RESEARCHER]'s own writing (the dataset survey, `proposal/`, `tutorial/` and the `scripts/` that build them; layout in `notes/README.md`): read or write it only when asked explicitly. `papers/` holds reference papers: read one only when asked explicitly.

## The loop

```
discuss → campaign → build + smoke → run cards → run → worklogs → close
```

- **Discuss first.** Sharpen the question, name the assumptions, say where the idea would break, propose experiments. Don't turn a discussion into a plan or code until [RESEARCHER] asks.
- **Four items before any compute:** the **question** (the claim tested), the **decision** (what changes with the outcome), the **done condition** (the number or artifact that ends the work) and the **kill bar** (the result that ends the line with no rescue run). Done condition and kill bar may be `None`, but only when [RESEARCHER] says so. If an item is missing or the request has two materially different readings, push back once: name the readings and recommend one.
- **Campaign** (`/campaign`): a research question and the experiments that answer it, in `docs/campaigns/YYYY-MM-DD_slug/README.md`. It opens one `open` node in the decision graph. Every experiment belongs to a campaign; there is no compute outside one. A new check is a new row in an existing campaign or a new campaign.
- **Build + smoke.** Write the experiment's code, config and tests, then run `SMOKE=1`. No permission needed.
- **Run cards.** Hand [RESEARCHER] the cards for every experiment ready to launch, then wait for "go". This applies to each batch, unless the campaign README says `Launch: auto`; then launch later batches yourself while they stay inside the campaign's cards and kill bar.
- **Run.** Experiments with no unmet `after` dependency run in parallel: one tmux session each, on disjoint cards, each writing only to its own `assets/experiments/<name>/`.
- **Worklog.** Every run writes or appends to `worklogs/<experiment>.md` in its campaign, pass or fail: run card, what was run, result with its path, insight, **what it rules out**, and what it means for the campaign's question. Worklogs are append-only.
- **Close** (`/close`, [RESEARCHER] only): closing summary in the campaign README, and the node set to `worked` or `failed` with its ruling.

Update `docs/STATE.md` in the same step whenever a campaign opens or closes, a run starts or ends, or something starts waiting on [RESEARCHER]. At the end of a session, rewrite its "Next step" section.

## [RESEARCHER] decides

Suggest in one line, then wait, for:

- commits (suggest one only when a campaign opens or closes, or an experiment finishes);
- opening or closing a campaign, and any change to a decision node;
- launching runs (see run cards above);
- more cards than the campaign has, or cards another person's processes hold;
- changing a question, bar, metric or setup file that a result already cites;
- deleting results or weights a worklog points to;
- stopping, pausing or editing any file used by a live run;
- anything in `.claude/settings*`, `notes/` or `papers/`.

Before writing a campaign or a closing, show a short overview and write only after [RESEARCHER] agrees. At a kill bar, stop that line and report. When a result falls short, look for where the implementation departs from the intended method and fix that; don't add components or knobs.

## Layout

Each experiment has the same name under three roots. Shipping an experiment means copying those three folders plus the setup files it names.

| Path | Holds |
| ---- | ----- |
| `src/experiments/<name>/` | code, a five-minute `README.md`, `tests/` |
| `scripts/experiments/<name>/` | `script.sh` and `config.yaml` |
| `assets/experiments/<name>/` | outputs, logs, `config.resolved.yaml` |
| `setup/<kind>/<id>.yaml` | shared setups: `dataset/`, `model/`, `metric/`, more kinds as needed |
| `src/utils/` | shared code; experiments never import from each other |
| `data/` | data shared by all campaigns and experiments, stored once; experiments read it and never copy it |
| `docs/` | `STATE.md`, `campaigns/`, `decisions/`, `ideas/` |

One module per idea, with a top docstring stating inputs and outputs. No clever indirection. Campaign folders hold reasoning, never data.

In markdown files, never break a line in the middle of a sentence to fit a page width. One paragraph or list item is one line.

## Setup files

`config.yaml` names shared setups by id and keeps its own knobs under `params`:

```yaml
setup:
  dataset: example_dev_v1        # → setup/dataset/example_dev_v1.yaml
  model: example_lm_v1           # a list of ids is also allowed
  metric: auroc_grouped5_v1
params:
  layers: all
```

- An experiment never overrides a setup field. A difference in setup is a different id.
- A setup file is frozen once a run has used it. To change it, copy it to a new id (`_v2`).
- `script.sh` resolves the config first and writes `config.resolved.yaml` beside the outputs.

```bash
uv run python -m src.utils.config uses model example_lm_v1   # which experiments share this setup
uv run python -m src.utils.config check                      # was a cited setup file edited later?
```

## Running

Use `uv` only (`uv run python ...`, `uv add ...`, `uv run pytest ...`); never raw `python` or `pip`.

Every `script.sh` must be safe to launch, leave and rerun:

1. **Smoke:** `SMOKE=1` runs the same stages and code path on a handful of items, in minutes, into `smoke/`.
2. **Resume:** per-item or per-shard outputs; skip what is on disk.
3. **Markers:** one `.<stage>.done` per stage, `.done` when all finish, `.failed` on any error.
4. **Progress:** a `tqdm` bar per process; long pipelines ship a `status.sh`.

Run card, written before every GPU run and kept at the top of the worklog entry:

```
Run:      <what and why, one line>
Command:  bash scripts/experiments/<name>/script.sh
Needs:    <cards>, ~<hours>
Writes:   assets/experiments/<name>/<...>
Smoke:    passed | failed, <n items>, <minutes>, <cards>
Resume:   yes, skips finished items | partial (ckpt every N) | no, because <why>
Decides:  <the number and threshold that settles pass / fail, or None>
```

Compute rules (edited by [RESEARCHER] when the environment changes):

- **Cards:** detect what is free, never hard-code ids. Pass `GPUS="0 1"` to pin a run to its assigned cards. Sharing a card with another user needs ≥ 8 GB headroom. Off-limits cards: none.
- **Multi-GPU:** shard data-parallel, one process per shard, ~2 per card when memory allows.
- **CPU:** `joblib` with `n_jobs ≤ 32`, BLAS threads capped at 1; CPU-heavy stages run one at a time.
- **Waiting:** don't poll. Wait on `.done` or `.failed` with one background loop that sleeps ≥ 5 min per check and also exits when the tmux session is gone.
- **Verify artifacts, not exit codes:** check shard and row counts before calling a stage complete.

Be frugal: grep before reading, summarise logs, and do the work yourself unless [RESEARCHER] asks for subagents.

## Testing

Test the core algorithm and the evaluation code before their first number enters a worklog. Prefer properties over values: fast against slow implementation, invariants, no leakage across splits. Skip plotting, argparse and thin wrappers. Tests sit in `tests/` next to the code and run from the repo root; the default run is fast and needs no GPU (`-m slow` and `-m gpu` are opt-in).
