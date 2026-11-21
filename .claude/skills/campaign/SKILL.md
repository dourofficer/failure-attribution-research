---
name: campaign
description: Turn a settled discussion into a campaign under docs/campaigns/ (one README with the four items and the experiment table) and open its node in the decision graph, or add an experiment to an existing campaign. Use when the user invokes /campaign or asks to turn an agreed direction into a plan.
---

# Campaign

A campaign is one research question and the experiments that answer it. This skill writes down what a discussion has already agreed; it does not start work on its own. When a discussion converges, say "this is ready for `/campaign`" in one line and wait.

It writes no experiment code and launches nothing. Building, smoking and run cards come afterwards (see `CLAUDE.md`).

## 1. Check the four items

All four must be settled in the conversation:

1. **Question:** the claim this work tests.
2. **Decision:** what we do differently depending on the outcome.
3. **Done condition:** the number, table or artifact that ends the campaign.
4. **Kill bar:** the result that closes the line with no rescue run.

Done condition and kill bar may be `None`, but only if the researcher said so. Never fill one in yourself. If an item is missing, stop, name the gap, and return to the discussion.

## 2. Place it

- Read `docs/decisions/index.md`. If a node already closed this line, say so and stop.
- The new node id is the highest existing id plus one, zero-padded (`D041` → `D042`).
- `parents` are the nodes whose rulings or open questions led here; `[]` for a root.
- Campaign folder: `docs/campaigns/YYYY-MM-DD_snake_slug/`, with today's date and a 2–4 word slug naming the subject.

## 3. List the experiments

One row per experiment. An experiment is one mirrored folder name under `src/experiments/`, `scripts/experiments/` and `assets/experiments/`, with one `script.sh` and one `config.yaml`.

For each row decide:

- **Answers:** the part of the question this experiment settles.
- **After:** the experiments that must finish first, or `none`. Rows with nothing pending can run in parallel.
- **Cards:** how many it needs. Parallel rows must fit on disjoint cards.
- **Setup:** the setup ids it uses (`setup/<kind>/<id>.yaml`). Reuse existing ids so results stay comparable; list any new setup file that has to be written.

Split experiments so that each has one output folder and one clear result. Don't create an experiment for something that is a parameter of another.

## 4. Overview, then write

Show a short overview first and write nothing yet: folder name, node id and parents, the four items in one line each, and the experiment table. After the researcher agrees, write:

**`docs/campaigns/<folder>/README.md`**

```markdown
# Campaign: <title>

**Opened:** YYYY-MM-DD   **Node:** D0XX   **Launch:** ask   **Status:** active

## Question
## Decision
## Done condition
## Kill bar

## Experiments

| Experiment | Answers | After | Cards | Status |
| ---------- | ------- | ----- | ----- | ------ |
| `<name>` | ... | none | 4 | planned |

## Approach
Method in a few lines, the setup ids used, and anything that could invalidate the result.
```

`Launch: ask` is the default: every batch of run cards waits for the researcher. Write `Launch: auto` only if the researcher asked for it.

Status values for a row: `planned`, `built` (smoke passed), `running`, `done → worklogs/<name>.md`, `failed → worklogs/<name>.md`, `dropped (<why>)`. Once a run has started, the four items are frozen; a change needs the researcher.

**`docs/decisions/nodes/<D0XX-short-slug>/node.md`** with `status: open`, the `campaign:` path, and only the `## Question` section filled in (why it follows from its parents, and the kill bar). Format: `docs/decisions/README.md`.

Then:

- create an empty `worklogs/` folder in the campaign;
- run `bash scripts/decision-tracker/build.sh` so `index.md` lists the new node;
- add the campaign to `docs/STATE.md` under "Active campaigns";
- report the folder path and which experiments can be built first.

## Adding an experiment to an existing campaign

When a new check belongs to a campaign that is still active, add a row to its table instead of opening a new campaign. Show the row first, write it after the researcher agrees. The four items do not change.
