I'd run three groups of experiments ("tracks"). Each one pairs released step labels with an agent system you can still run today, and two of them depend on a backbone deadline. One finding changes the setup: the Magentic-One benchmarks come from two different implementations of Magentic-One, so you have to choose which one to run.

1. **τ-bench retail.** This is the cheapest track and the only one where injected labels can be checked by re-running from the injected step.
2. **GAIA-style questions on GPT-4o (Magentic-One and CaptainAgent).** This has the most human labels but a hard deadline: OpenAI shuts down the last GPT-4o snapshots (`2024-08-06` and `2024-11-20`) on **January 14, 2027**, 54 days from now. `2024-05-13` already shut down on October 23, 2026.
3. **MCP-Atlas on Qwen3.5-27B.** The weights are open, so the backbone matches exactly and nothing expires. It also gives a direct comparison with OAT.

Use TRAIL only as a transfer test. Both of its backbones are gone: o3-mini shut down on October 23, 2026, and Claude 3.7 Sonnet was retired from the Claude API on February 19, 2026.

## 1. Evaluation subsets and the system that matches each

| Eval subset (labeled failures)                                                              | Questions from (S)                                  | System to run (A)                                        | Backbone (M)                                                       |
| ------------------------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------ |
| AgentRx τ-retail (29, several steps)                                                        | τ-bench retail **test**                             | τ-bench v1 tool-calling agent plus an LLM user simulator | GPT-4o (from the paper)                                            |
| TrajErrBench τ² retail 179 / airline 221 (one step each, 3 annotators)                      | τ²-bench                                            | τ²-bench agent                                           | DeepSeek-Reasoner, GPT-5, Qwen3-Max                                |
| Who&When Hand-Crafted 58 · MP-Bench manual (Hand-Crafted part) 57 · AgentRx Magentic-One 44 | GAIA validation 30 + AssistantBench 28              | **Legacy** Magentic-One (`autogen-magentic-one`)         | GPT-4o (snapshot not recorded)                                     |
| MP-Bench manual (MAST part) 100 · TraceElephant Magentic-One 91                             | GAIA validation (levels 2–3); GAIA + AssistantBench | **AgentChat** Magentic-One (`MagenticOneGroupChat`)      | GPT-4o                                                             |
| Who&When Algorithm-Generated 126 · MP-Bench automatic 120 · TraceElephant CaptainAgent 85   | GAIA + AssistantBench                               | CaptainAgent (AG2)                                       | GPT-4o (`2024-08-06` and `2024-11-20` in the TraceElephant traces) |
| MCP-Atlas / OAT 88 (+103 successes)                                                         | MCP-Atlas public subset                             | MCP-Atlas harness                                        | Qwen3.5-27B                                                        |
| *Optional:* TraceElephant SWE-agent 44                                                      | SWE-bench Verified                                  | SWE-agent                                                | GPT-4o                                                             |
| *Transfer only:* TRAIL GAIA 114 / SWE 30                                                    | GAIA validation / SWE-bench Lite                    | smolagents Open Deep Research / CodeAgent                | o3-mini / Claude 3.7 Sonnet, both retired                          |

**Two findings from the data that affect the choice of system:**
- **"Magentic-One" is two implementations with different log structure.**
  - Who&When Hand-Crafted uses the legacy package. Its roles look like `Orchestrator (thought)` and `Orchestrator (-> WebSurfer)`, and 1,563 of its 2,993 steps are orchestrator thoughts. MP-Bench's Hand-Crafted logs and all 59 AgentRx Magentic-One files are these same runs.
  - MAST's 195 Magentic-One GAIA traces are AgentChat console logs (`---------- MagenticOneOrchestrator ----------`); MP-Bench's other 100 logs come from these. TraceElephant also runs AgentChat (autogen 0.5.3).
  - Step boundaries and run lengths differ between the two, so generate with whichever one matches the subset you evaluate on.
- **AgentRx τ-retail runs come from τ-bench's test split.** All 100 runs in `tau_dataset_full.json` match test tasks by index and user id, and 73 of them succeed. The 500 train tasks therefore don't overlap with it.

## 2. Where to get each system

- **τ-bench:** `sierra-research/tau-bench`. Pin v1 for AgentRx. Its README now says the tasks are no longer updated and points to τ²/τ³-bench, and τ² has replaced 50 retail tasks. For TrajErrBench, pin the τ² version that paper used.
- **AgentChat Magentic-One and CaptainAgent:** already on disk in `data/attribution/traceelephant/code_TraceElephant/code/agent_system/`.
  - It comes with `llm_api_middleware`, a proxy that logs every LLM call and can intercept or change responses. That proxy is the hook for both caching a run's prefix and injecting errors.
  - TraceElephant changed the tool layer: Patchright instead of Chromium, DuckDuckGo for search, and CaptainAgent's Bing search replaced. The agent logic is the same but the tools are not.
- **Legacy Magentic-One:** available only from the autogen GitHub tag `v0.4.4` (`python/packages/autogen-magentic-one`); PyPI only has 0.0.1. Because TraceElephant's middleware sits in front of the API, it should work here too.
- **MCP-Atlas:** `scaleapi/mcp-atlas`, a containerized harness with 36 MCP servers and 220 tools.
  - OAT's repo does not include its generation settings (sampling, thinking mode, tool-call parser).
  - The empty-final-step artifact we found depends on the parser, so pin your settings and record them.

## 3. Task sources for unlabeled trajectories

Rule: leave out every task that any evaluation subset in the same track uses.

| Track          | Leave out                                                                                                                                                                    | Use for generation                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| τ-bench        | Retail test (115)                                                                                                                                                            | Retail **train 500 + dev 20** (counted from the repo), several runs per task. Use the 50 airline test tasks only for cross-domain checks.                                                                                                                                                                                                                                                                                           |
| GAIA-style     | All 165 GAIA validation questions (used by Who&When, MAST, TRAIL and TraceElephant) and the AssistantBench tasks in Who&When and TraceElephant (AssistantBench has only 214) | • **GAIA test** (300): answers are withheld, so these runs get no outcome label, but the style and file attachments match.<br>• **TaskCraft**: about 32k–41k GAIA-style tasks depending on the version, with checkable answers, covering web, PDF and image (ICLR 2026).<br>• **WebWalkerQA** (680): website navigation, a good fit for WebSurfer.<br>• Harder extras: FRAMES and BrowseComp (sizes 824 and 1,266 are from memory). |
| MCP-Atlas      | OAT's 198 tasks                                                                                                                                                              | The remaining ~302 of the 500 public tasks, assuming OAT's 198 all come from the public set. The other 500 are private.                                                                                                                                                                                                                                                                                                             |
| SWE (optional) | SWE-bench Verified and Lite                                                                                                                                                  | SWE-Gym (2,438), SWE-smith (about 50k, built for SWE-agent), SWE-rebench (21k). Check which repositories they share with Verified.                                                                                                                                                                                                                                                                                                  |

**How well these sources fit:**
- **File attachments:** GAIA questions often come with xlsx, pdf, audio or image files that exercise the FileSurfer and Coder agents. Plain web QA (WebWalkerQA, FRAMES) won't; TaskCraft's PDF and image tasks partly cover this.
- **Outcome labels:** if your method needs outcomes (OAT trains on successes only), GAIA test gives none and BrowseComp will be almost all failures on GPT-4o.
- **MCP-Atlas pool size:** it is the smallest pool. A run counts as a success when an LLM judge finds every rubric claim covered, so any new task also needs rubric claims.

## 4. Error injection for pseudo-labels

| Track                            | Can a step be re-run from its saved state?                                                                                 |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| τ-bench                          | Yes. The database resets exactly. The user simulator is an LLM, so continuations vary, but the prefix is restored exactly. |
| SWE                              | Yes, via containers.                                                                                                       |
| MCP-Atlas                        | Partly. The servers are containerized, but some call live APIs.                                                            |
| Live-web GAIA and AssistantBench | Only by serving the prefix from recorded responses; the continuation can't be reproduced.                                  |

**Protocol I'd use:**
1. **Inject into successful runs, using the same model.** Take a successful run and pick a step k. Have the *same backbone* write a faulty version of step k for a named error type (as the Aegis and Who&When Pro injectors do), then let the run continue live. Keep runs that flip to failure and label them {k}.
2. **Control run.** Also re-run from k with the original step.
   - On τ-bench, require most of 3–5 control continuations to succeed and most injected ones to fail.
   - On live web, a single control run is a filter, not a measurement.
3. **Decoy rewrites labeled clean.** Pass some steps through the same rewriting pipeline with a meaning-preserving instruction and label them clean. Otherwise "this step was rewritten" becomes a cue the attributor can learn.
4. **Match where errors sit.** Sample k from the evaluation subset's labeled-position distribution. The survey's `fig_errpos` shows natural failures spread across the run with a spike at the end. Who&When Pro injects late (mean relative position 0.74) and TracerTraj injects early.
5. **Repair labels on natural failures (τ-bench only).** Each τ-bench task ships its gold action sequence (`info.task.actions`).
   - Swap the agent's step k for the gold action, re-run, and see whether the run now succeeds. This labels *natural* failures, which are closer to the evaluation data than injected ones.
   - The cost is about (steps × runs) re-runs per failure, which is cheap here at about 30 messages per run.
6. **Faults in what a tool returns** (wrong number, 404, truncated page, MCP tool error): decide in advance which step gets the label. Who&When-style labels blame the agent step that accepts the bad result, not the tool output itself.
7. **Multi-step labels:** MP-Bench, AgentRx and TRAIL label sets of steps, but one injection yields one step. Either inject several errors, or say that you score the decisive step only.

## 5. Where this could break

- **The GPT-4o deadline.** Anything in track 2 that needs the matched backbone must be collected by January 14, 2027. Azure's retirement dates may differ; I haven't checked them. Paid API runs count as compute under CLAUDE.md, so they need a campaign first.
- **Track 2 cost**, from token usage recorded in TraceElephant's failed runs:
  - Magentic-One on GAIA: median 183k prompt + 4.3k completion tokens (90th percentile 479k prompt); on AssistantBench, 371k prompt.
  - CaptainAgent on GAIA: median 47k prompt tokens.
  - At $2.50 per million input and $10 per million output tokens, that is about $0.50 per Magentic-One GAIA run and $0.14 per CaptainAgent run, before search API costs. 2,000 Magentic-One runs would cost about $1k–1.5k.
- **The web has changed since 2024.** TraceElephant had to change browser and search engine to get past bot detection, and many AssistantBench answers are time-sensitive. Expect more tool and page failures than in the 2024 logs, so the mix of failure types shifts, not just the tasks.
- **Who&When and MP-Bench store agent outputs only.** Record full traces with the middleware and cut them down to the evaluation view; you keep both.
- **Conformal calibration.** Pseudo-labels can train the score, but coverage only holds on human labels if you calibrate on human-labeled trajectories from the same subset.
  - The calibration splits would be tiny (29 trajectories for AgentRx τ-retail, 58 for Who&When Hand-Crafted).
  - MP-Bench's "all annotators" and "2-of-3" label sets give different coverage targets.
- **Re-running from a saved state may not match the original run.** Two recent papers that aren't in `notes/` cover this:
  - [ReplayRCA](https://arxiv.org/abs/2610.41025) scores diagnoses by whether a replayed repair fixes the run, and finds that blaming the last step doesn't hold up under that test.
  - [Pakhomov & Nijkamp](https://arxiv.org/abs/2611.05650) report that restored snapshots diverged from the live run in 9 and 20 of 24 states until the random-number state was also restored.
  - Measure how faithfully your replay reproduces the prefix before you trust injected labels.

Before any runs, the campaign still needs its four items: question, decision, done condition, kill bar. With the deadline, track 2 is the one to settle first. Should I save this to `docs/ideas/`?

Sources:
- Model shutdowns and pricing: [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations), [Claude model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations), [GPT-4o pricing](https://developers.openai.com/api/docs/models/gpt-4o)
- MCP-Atlas: [Hugging Face dataset](https://huggingface.co/datasets/ScaleAI/mcp-atlas), [paper](https://arxiv.org/abs/2602.00933)
- τ-bench: [τ-bench](https://github.com/sierra-research/tau-bench), [τ²-bench](https://github.com/sierra-research/tau2-bench)
- Agent systems: [autogen-magentic-one on PyPI](https://pypi.org/project/autogen-magentic-one)
- Task sources: [TaskCraft](https://arxiv.org/abs/2506.10055), [WebWalkerQA](https://arxiv.org/abs/2501.07572), [SWE-smith](https://arxiv.org/abs/2504.21798), [SWE-Gym](https://arxiv.org/abs/2412.21139), [SWE-rebench](https://arxiv.org/pdf/2505.20411)
- Injection and replay: [AgenTracer](https://arxiv.org/abs/2509.03312), [ReplayRCA](https://arxiv.org/abs/2610.41025), [Pakhomov & Nijkamp](https://arxiv.org/abs/2611.05650)