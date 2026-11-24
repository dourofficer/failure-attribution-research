# Synthetic multi-error dataset for conformal failure attribution

Status: idea, not agreed as a campaign. Written 2026-11-22 from a discussion with [RESEARCHER].

## Why we need our own dataset

Five papers were read: AgenTracer, AEGIS, CORRECT, OAT and Conformal Agent Error Attribution (CAEA). All follow the same strategy: build a dataset, show good results on its test split, then check on an existing benchmark, usually Who&When. We follow the same strategy. If our method performs favourably in a clearly stated scenario, that is a contribution.

Who&When cannot carry a conformal method alone: its 184 trajectories cover calibration and testing but leave nothing for training a scorer or for varying conditions.

## How existing papers build their data

| | AgenTracer (TracerTraj) | AEGIS (Aegis) | CORRECT (CORRECT-Error) | OAT | CAEA |
|---|---|---|---|---|---|
| Source tasks | MBPP+, KodCode, Blackjack, MATH, GSM8K, GAIA, HotpotQA | MATH, GSM8K, HumanEval, SciBench, MMLU-Pro, GAIA | HotpotQA, Musique, WikiMQA, ARC, MMLU-Pro, Math500, GAIA | MCP-Atlas | GSM8K, MATH |
| Agent systems | MetaGPT, AutoGen, Smolagents, AgentPrune, AFlow, OWL-Workforce | MacNet, DyLAN, LLM-Debate, AgentVerse, Magentic-One, SmolAgents | A Magentic-One variant in AutoGen | Single tool-calling agent | MacNet, DyLAN |
| Agent backbone | Not stated | GPT-4o-mini, temperature 0 | GPT-5 series and GPT-4o series | Qwen3.5-27B | Not stated |
| Error source | Counterfactual replay on real failures, plus fault injection on successes | Injection only (prompt injection or response corruption, 14 MAST modes) | Errors transplanted from human-labelled seed failures | Real failures, labelled by hand | Injection only |
| Labeller | DeepSeek-R1 | Not named | GPT-5 | Humans | Not named |
| Size | 2,476 labelled | 9,533 | Over 2,000 | 103 successes, 88 failures | 4,800 |
| Label | One (agent, step) | Several (agent, error mode), no step | One step | A set of steps | One step |

Who&When itself: 184 failed trajectories on GAIA and AssistantBench, run by Magentic-One (hand-crafted subset, 58) and CaptainAgent (algorithm-generated subset, 126), both on GPT-4o.

Gaps we can use:

- No synthetic set uses AssistantBench or CaptainAgent, and only AEGIS (449 runs) and CORRECT use Magentic-One. Every Who&When result in these papers is cross-system transfer.
- Every generated set has one labelled error step per trajectory (AEGIS has several errors but no step index). Only OAT has step sets, from 88 hand-labelled failures.

## Agreed direction

- **Agent systems:** Magentic-One and CaptainAgent, so the data has the same structure as Who&When, TraceElephant, AgentRx and MP-Bench. This makes the real-data check meaningful.
- **Source tasks:** not limited to GAIA and AssistantBench. Any source compatible with both systems can be added.
- **Labels:** a set of confirmed error steps per trajectory, which is the shape set prediction needs.

## Labelling method

A variant of AgenTracer's counterfactual replay that yields several labelled steps:

1. Collect failed trajectories.
2. A strong LLM suggests N candidate error steps per trajectory, each with a rationale and a correction.
3. Replay the trajectory from each candidate step with the oracle correction. If the run turns from failure to success, the step is confirmed.
4. Each trajectory ends with a set of confirmed error steps.
5. The train set stops here. For the test set, human annotators verify the confirmed steps.

Positions taken by [RESEARCHER] on the known risks:

- **Late steps confirmed for the wrong reason** (the correction supplies what the agent failed to find): handled by heuristic rules, to be designed. Assume solvable for now.
- **Several independent errors where no single fix rescues the run:** expected to be rare, since one root cause usually shows up in several places. Such trajectories are discarded.
- **A flip that is luck:** the proposer's rationale gives a targeted correction, which limits this. Human verification covers the test set. Accepted as a limitation of the synthetic train set.
- **Coverage definition:** all labelled steps of a trajectory are equally valid, and coverage is measured from the overlap between the prediction set and the labelled set. To be formalised later.
- **Replay support:** if the systems do not support it, we code it ourselves.

## TraceElephant as a starting point

Repository: https://github.com/TraceElephant/TraceElephant (ACL 2026, arXiv 2604.22708). 220 annotated failures from 380 executions on Captain-Agent, Magentic-One and SWE-Agent, over GAIA, AssistantBench and SWE-Bench Verified.

- It has no replay toolkit for Magentic-One or CaptainAgent. Its README lists replay from intermediate steps as something the release makes possible.
- It ships runnable versions of both systems with GAIA and AssistantBench runners, patched to work today (DuckDuckGo instead of Bing, Patchright against bot detection, a replacement for the deprecated Bing API in CaptainAgent).
- It ships an LLM API proxy (`code/llm_api_middleware/middleware.py`, 226 lines, OpenAI-compatible) that logs every request and response without changing agent code. The README names intercepting and modifying requests and responses as the intended extension.
- SWE-agent is the only one of the three with native replay (`run_replay.py`).

Replay design that follows: serve the logged responses up to step t, substitute the correction at t, let the run continue live. Tool calls (browser, code) execute again, so the environment is the part that is not reproducible.

## Task sources scouted

Sizes come from web searches on 2026-11-22, except BrowseComp, which is from memory.

| Source | Usable tasks | Environment | Fit |
|---|---|---|---|
| GAIA validation | 165 | live web, files | Native to both systems; same tasks as Who&When and TraceElephant |
| AssistantBench dev | 33 | live web | Same as above |
| WebWalkerQA | 680 (English and Chinese) | live web, deep site navigation | Good for the browsing agents; GPT-4o stays under 40% |
| FRAMES | 824 | Wikipedia, 2–15 articles per question | Stable pages, exact answers; easier (66% multi-step baseline) |
| BrowseComp | 1,266 | live web | Very hard, so many failures, but an oracle fix may rarely rescue a run |
| BrowseComp-Plus | 830 | fixed corpus of 100,195 documents | Reproducible, but swaps the browser for a retrieval tool |
| WebArena-Verified | 812 | self-hosted sites, resettable | Magentic-One was evaluated on WebArena (32.8%); CaptainAgent is untested there |
| InfiAgent-DABench | 257 | CSV files, code | CaptainAgent's own data-analysis benchmark; no web |
| TaskCraft | about 41k, synthetic | web and tools | Unlimited volume, uneven quality |
| HotpotQA, Musique, Math500 | thousands | mostly none | CORRECT ran Magentic-One on them; short trajectories |

Only tasks with a public answer are usable, since both the success check and the oracle correction need one. GAIA's 301 test answers are hidden. AssistantBench has 214 tasks, of which 33 are the development set; whether the other 181 answers are withheld is unverified.

Recommendation: WebWalkerQA (English) and FRAMES as the main expansion, with a BrowseComp sample for hard cases. They are short-answer, live-web tasks, so trajectories keep the same structure as Who&When. WebArena-Verified is the fallback if live-web replay proves too noisy, because its sites reset.

Smaller sets seen but not recommended: WebShaper (500 public items), SealQA seal-0 (111), xbench-DeepSearch (100, Chinese).

## Draft plan

1. **Infrastructure.** Take the two runners from TraceElephant and extend the proxy to record, serve and override responses.
2. **Pilot.** Run about 20 tasks per source on both systems and keep sources with a workable failure rate and Who&When-like trajectory lengths.
3. **Rollouts.** Run the full pool and score with each benchmark's checker, keeping the successful runs too.
4. **Propose.** A strong LLM reads each failed trace with the ground truth and returns N candidate steps, each with a rationale and a correction.
5. **Verify.** Replay each candidate with its correction and confirm the step if the run succeeds. Discard trajectories with no confirmed step.
6. **Split.** Split by task into train, calibration and test; annotators verify the test set.
7. **Real-data check.** Evaluate on Who&When, TraceElephant, AgentRx and MP-Bench.

## To decide later

- **N** and the proposer model.
- **Agent backbone:** GPT-4o to match Who&When, or a current model.
- **GAIA and AssistantBench in training.** The real benchmarks use these same tasks, so training on them overlaps with the real-data check.
- **Step indexing.** Who&When logs outputs only and TraceElephant logs full traces, so our step index must map to both.
- **Locality rules** for late steps.
- **Coverage definition.**
- **Replays per candidate** and whether to cache tool outputs.
- **An injection arm** on the successful runs, which would make the set hybrid.
- **Scale and budget.** Verification costs up to N extra partial rollouts per failed run; the pilot will give the real number.
- **Annotation protocol** for the test set.

## Unverified

- Whether TracerTraj and Aegis-Bench are publicly released.
- CAEA's claim that Who&When is the only academic dataset with human step-level labels. TRAIL and AgentErrorBench may also have them.
- The structure of AgentRx and MP-Bench traces (not read in this session).
- CaptainAgent on WebArena, WebWalkerQA and FRAMES: no published runs found.

## Sources

- Papers in `papers/`: `AgenTracer_2509.03312.pdf`, `AEGIS_2509.14295.pdf`, `CORRECT_2509.24088.pdf`, `OAT_2607.12747.pdf`, `ConformalAgentErrorAttribution_2605.06788.pdf`.
- TraceElephant: https://github.com/TraceElephant/TraceElephant
- BrowseComp-Plus: https://huggingface.co/datasets/Tevatron/browsecomp-plus and https://arxiv.org/abs/2508.06600
- WebWalker: https://arxiv.org/abs/2501.07572
- FRAMES: https://huggingface.co/datasets/google/frames-benchmark
- TaskCraft: https://arxiv.org/abs/2506.10055
- Magentic-One: https://arxiv.org/abs/2411.04468
- CaptainAgent: https://arxiv.org/abs/2405.19425
- GAIA: https://arxiv.org/abs/2311.12983
- AssistantBench: https://arxiv.org/abs/2407.15711
- WebArena-Verified: https://github.com/ServiceNow/webarena-verified
- InfiAgent-DABench: https://arxiv.org/abs/2401.05507
