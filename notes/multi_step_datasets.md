# Datasets with several labelled error steps per trajectory

Computed 2026-11-21 from the copies under `data/`. Count scripts were run from the session scratchpad and are not kept; the counting rules are given under each table so the numbers can be rebuilt.

## Per subset

| Dataset / subset | Questions from | System that produced the trajectories | Backbone | Labelled failures (+ empty) | > 1 labelled step | Steps per traj. (mean / median / max) |
|---|---|---|---|---|---|---|
| MP-Bench / automatic (Who&When Algorithm-Generated logs) | GAIA 98, AssistantBench 22 | CaptainAgent (AG2), teams assembled automatically per task | GPT-4o | 120 (+0) | union 119 (99%); 2-of-3 94 (79%) | union 5.3 / 5 / 10; 2-of-3 2.8 / 3 / 7 |
| MP-Bench / manual, from Who&When Hand-Crafted | GAIA 30, AssistantBench 27 | Magentic-One | GPT-4o | 57 (+0) | union 55 (96%); 2-of-3 47 (82%) | union 13.2 / 10 / 78; 2-of-3 7.1 / 4 / 51 |
| MP-Bench / manual, from MAST | GAIA validation, levels 2–3 (86 + 26) | Magentic-One | GPT-4o | 100 (+12) | union 91 (91%); 2-of-3 80 (91%, of 88 non-empty) | union 13.7 / 9.5 / 41; 2-of-3 5.4 / 4 / 29 |
| TRAIL / GAIA | GAIA validation | HF OpenDeepResearch on smolagents: a manager CodeAgent that can call a search ToolCallingAgent | o3-mini | 114 (+3) | 88 (77%) | 3.4 / 2 / 12 |
| TRAIL / SWE-bench | SWE-bench Lite | smolagents CodeAgent, single agent | Claude 3.7 Sonnet | 30 (+1) | 29 (97%) | 6.3 / 6 / 12 |
| AgentRx / τ-retail | τ-bench retail | One tool-calling agent plus a simulated user | GPT-4o (from the paper, not checked in the data) | 29 (+0) | 7 (24%) | 1.3 / 1 / 4 |
| AgentRx / Magentic-One (from Who&When Hand-Crafted) | GAIA 22, AssistantBench 22 | Magentic-One | GPT-4o | 44 (+0) | 30 (68%) | 6.7 / 3 / 55 |
| MCP-Atlas / OAT | MCP-Atlas public subset | Single tool-calling agent using MCP servers | Qwen3.5-27B | 88 (+7 with no usable label) | 22 (25%) | 1.6 / 1 / 12 |

Label sets and counting rules:

- **MP-Bench** (`data/mp_bench/adobe_multi-agent-eval-bench/MP-Bench/<annotator>/<manual|automatic>/`): the steps with `fail_annotation == "1"`. "Union" counts every step any of the 3 annotators flagged; "2-of-3" keeps steps that at least two flagged. The subset is decided by `log_source`.
- **TRAIL** (`data/trail/whowhen_format/`): `error_steps`.
- **AgentRx** (`data/agentrx/github/data/ground_truth/`): the distinct `failures[].step_number`.
- **MCP-Atlas / OAT** (`data/mcp_atlas_oat/`): the distinct valid `errors[].scope`.
- **Empty:** a trajectory with no labelled step; it is left out of the step statistics.

Where the facts come from:

- GAIA vs AssistantBench: read from the question ids. GAIA ids are UUIDs, AssistantBench ids are 64-character hex.
- TRAIL backbones: read from `llm.model_name` in the traces.
- MAST Magentic-One backbone: all 195 GAIA runs in `MAD_full_dataset.json` are GPT-4o.
- Systems and the τ-retail backbone: from the survey (`far-survey/`) and the papers.

## What a label set means differs

- **MP-Bench:** expert opinions that disagree, not one ground truth. The paper reports all three annotators agreeing on a single step in only 16% of logs.
- **TRAIL:** every error, including LOW-impact formatting slips. There is no decisive-error notion, and some traces end with the right answer.
- **AgentRx:** every failure event, plus one marked `root_cause`.
- **MCP-Atlas / OAT:** only the steps that contributed to the failure. In 39 of 88 the final answer step is the only label, and 13 failures end in an empty final step (a tool call left inside `reasoning_content`) that never occurs in a success. Details are in `data/mcp_atlas_oat/README.md`.
- **Step units:** TRAIL steps are LLM and TOOL spans from an observability trace. The others are messages or turns.

## Overlap and diversity

- **Shared trajectories.** Who&When Hand-Crafted (Magentic-One, GPT-4o) also underlies MP-Bench's 57 Hand-Crafted logs and AgentRx's 44 Magentic-One trajectories. These are different labels on overlapping runs, not independent data.
- **Few distinct sources.** Five of the eight subsets are GAIA/AssistantBench questions on GPT-4o. Only TRAIL / SWE-bench, AgentRx / τ-retail and MCP-Atlas / OAT differ in both system and backbone.
- **Where the multi-step labels are.** Mostly in the long Magentic-One and SWE-bench logs. AgentRx / τ-retail and MCP-Atlas / OAT are mostly single-step.

## Datasets that look multi-step but are not

- **AgentErrorBench:** the paper describes a minimal set of root-cause steps, but the released labels have exactly one step in all 200 trajectories.
- **Aegis:** 77% of its 600 test trajectories have several labels, but they are faulty agents, not steps.
- **SearchAuditBench:** one critical step with a tolerance window around it.
- **Failure as a Process:** three time points of different kinds, and the trajectory text is not released.
- **MemTrace:** one faulty operation per query.
- **One step each:** Who&When, TraceElephant, Who&When Pro, LongRCA, SymTrace, TracerTraj, CORRECT-Error, AgentHallu and TrajErrBench.
