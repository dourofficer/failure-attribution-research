---
name: close
description: Close a campaign - check that every experiment has a worklog, write the closing summary in the campaign README, and set its decision node to worked or failed with the ruling. Use only when the user invokes /close.
---

# Close

Ends a campaign and records its ruling in the decision graph. Only the researcher invokes it. When a campaign has met its done condition or hit its kill bar, say "this is ready for `/close`" in one line and wait.

## 1. Check the campaign is finished

Read the campaign README and its worklogs.

- Every row in the experiment table is `done`, `failed` or `dropped`. Nothing is `running`.
- Every `done` or `failed` row has a worklog with a "Rules out" section.
- Every number you plan to quote has a file under `assets/` that you have opened and checked.
- `uv run python -m src.utils.config check` reports no setup file edited after a run cited it.

If something is missing, stop and say what. Don't close around a gap.

## 2. Decide the outcome with the researcher

- **Status:** `worked` if the question got a usable answer that the project will act on; `failed` if the line is closed with nothing to build on. A kill bar reached is normally `failed`. With no done condition and no kill bar, ask the researcher which it is.
- **Ruling:** one or two sentences on what the project now does.
- **Rules out:** what must not be re-run, taken from the worklogs.
- **Left open:** questions raised. These are candidates for `docs/ideas/`, not new work.
- **More than one ruling?** If the campaign settled two separate things, propose a second node with the same parents.
- **Overturns an earlier node?** Propose setting that node to `superseded` and listing it as a parent.

## 3. Overview, then write

Show a short overview first and write nothing yet: status, ruling, rules out, left open, and any extra or superseded node. After the researcher agrees:

**Campaign README:** set `Status: closed YYYY-MM-DD` in the header and append:

```markdown
## Closing summary

- **Outcome:** done condition met | kill bar reached | stopped by [RESEARCHER] because <why>
- **Result:** the numbers, each with its path under assets/
- **Ruling (D0XX, worked | failed):** ...
- **Rules out:** ...
- **Left open:** ...
```

**Decision node:** set `status`, sharpen the `title` so it states the answer, and fill in `## Ruling`, `## Rules out` and `## Left open`. Leave `## Question` as it was written when the node opened.

Then:

- run `bash scripts/decision-tracker/build.sh`;
- remove the campaign from "Active campaigns" in `docs/STATE.md` and update "Next step";
- suggest a commit in one line, with a proposed message, and wait.
