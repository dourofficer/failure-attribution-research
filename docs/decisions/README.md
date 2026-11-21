# Decision graph

One node per campaign. A node is a research question and, once the campaign closes, its ruling. Read `index.md` before proposing a direction, so a closed line is not run again.

## Life of a node

1. `/campaign` creates `nodes/D0XX-short-slug/node.md` with `status: open` and the question.
2. The campaign runs. The node does not change.
3. `/close` writes the ruling and sets `status: worked` or `failed`.

A campaign that ends with two separate rulings adds a second node at close. A node overturned by later work is set to `superseded`, and the new node lists it as a parent.

## Format

```markdown
---
id: D007
title: One line: the question, and after closing, its answer
parents: [D005]        # nodes this followed from; [] for a root
status: open           # open | worked | failed | superseded
campaign: docs/campaigns/YYYY-MM-DD_slug
---

## Question      why this follows from its parents, and the kill bar
## Ruling        what we now do, with numbers and paths to the evidence
## Rules out     what must not be re-run
## Left open     questions this raises
```

Only `id`, `title`, `parents` and `status` shape the graph. Other frontmatter keys are kept and ignored.

- `id` is unique, zero-padded and increasing.
- `parents` are edges parent → this node. Several parents and cycles are allowed. An unknown parent id is skipped with a warning.
- A failed node is worth as much as a worked one: it is what stops a line being re-run.
- Figures go in the node's own `assets/` folder and are referenced relatively.

## Rebuilding the index

```bash
bash scripts/decision-tracker/build.sh           # regenerates index.md and graph.html
bash scripts/decision-tracker/watch.sh           # optional viewer at http://localhost:8148/graph.html
bash scripts/decision-tracker/watch.sh stop
```

`index.md` is generated and tracked in git; do not edit it by hand. `graph.html` is generated and git-ignored.
