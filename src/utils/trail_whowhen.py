"""Flatten TRAIL traces into the Who&When record format.

Inputs:  the TRAIL GitHub `benchmarking/` folder, which holds `data/{GAIA,SWE Bench}/<trace_id>.json` (span trees)
         and `processed_annotations_{gaia,swe_bench}/<trace_id>.json` (errors keyed by span id).
Outputs: one JSON per trace in `<out>/{GAIA,SWE_Bench}/<trace_id>.json`, shaped like a Who&When record.

Mapping:
- Steps are the LLM and TOOL spans of the tree in DFS (document) order, which is chronological in TRAIL. CHAIN, AGENT
  and framework spans are wrappers and are dropped; every TRAIL error that resolves to a span sits on an LLM or TOOL span.
- `history[0]` is the task as the top-level agent received it (`role: "human"`), as in Who&When. Steps follow from index 1,
  and every step index below is an index into `history` (0-based, Who&When convention).
- LLM step: `content` is the output message text plus one `Tool call: name(args)` line per tool call; `role` is the agent
  that owns the call (`manager` for a CodeAgent with managed agents, the managed agent's name for a ToolCallingAgent,
  `CodeAgent` for a lone CodeAgent, `answer_formatter` for the post-hoc LLM call outside any agent).
- TOOL step: `content` is `Input: ...` and `Output: ...`; `role` is the tool span name (e.g. `SearchInformationTool`).
- Who&When fields: `mistake_step` (str) is the earliest labelled step, `mistake_agent` its role, `mistake_reason` the
  descriptions on that step. TRAIL labels every error, so the full set is kept in `error_steps` and `errors`.
- Extra fields Who&When lacks: per-step `span_id`, `span_kind`, `span_name`; record-level `split`, `task_id`, `errors`,
  `error_steps`, `unresolved_errors` (location not in the trace), `scores`, `annotation_repaired`.
- `groundtruth`: GAIA `true_answer`; SWE-bench gold `patch`. `is_corrected` is always False.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SPLITS = {"GAIA": "processed_annotations_gaia", "SWE Bench": "processed_annotations_swe_bench"}
STEP_KINDS = ("LLM", "TOOL")


def load_annotation(path: Path) -> tuple[dict, bool]:
    """Parse an annotation file; repair trailing commas (one TRAIL file has them). Returns (data, repaired)."""
    text = path.read_text()
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[\]}])", r"\1", text)), True


def _kind(span: dict) -> str:
    return span.get("span_attributes", {}).get("openinference.span.kind", "-")


def _agent_role(agent_span: dict | None, parent_agent: dict | None, trace_has_agent: bool) -> str:
    if agent_span is None:
        # Outside every agent: the post-hoc formatter call, unless the trace lost its AGENT spans (truncated root).
        return "answer_formatter" if trace_has_agent else "CodeAgent"
    attrs = agent_span.get("span_attributes", {})
    if agent_span["span_name"].startswith("ToolCallingAgent"):
        pa = (parent_agent or {}).get("span_attributes", {})
        return pa.get("smolagents.managed_agents.0.name", "ToolCallingAgent")
    return "manager" if "smolagents.managed_agents.0.name" in attrs else "CodeAgent"


def _llm_content(attrs: dict) -> str:
    parts = [attrs.get("llm.output_messages.0.message.content") or ""]
    i = 0
    while f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.name" in attrs:
        name = attrs[f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.name"]
        args = attrs.get(f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.arguments", "")
        parts.append(f"Tool call: {name}({args})")
        i += 1
    return "\n".join(p for p in parts if p)


def flatten_spans(roots: list[dict]) -> list[dict]:
    """DFS over the span tree; return LLM/TOOL spans as Who&When-style steps (without the human turn)."""
    steps: list[dict] = []
    has_agent = _find(roots, lambda s: _kind(s) == "AGENT") is not None

    def walk(span: dict, agents: list[dict]) -> None:
        kind = _kind(span)
        attrs = span.get("span_attributes", {})
        if kind in STEP_KINDS:
            if kind == "LLM":
                role = _agent_role(agents[-1] if agents else None, agents[-2] if len(agents) > 1 else None, has_agent)
                content = _llm_content(attrs)
            else:
                role = span["span_name"]
                content = f"Input: {attrs.get('input.value', '')}\nOutput: {attrs.get('output.value', '')}"
            steps.append({"content": content, "role": role, "span_id": span["span_id"], "span_kind": kind,
                          "span_name": span["span_name"]})
        child_agents = agents + [span] if kind == "AGENT" else agents
        for child in span.get("child_spans", []):
            walk(child, child_agents)

    for root in roots:
        walk(root, [])
    return steps


def _find(spans: list[dict], pred) -> dict | None:
    for s in spans:
        if pred(s):
            return s
        hit = _find(s.get("child_spans", []), pred)
        if hit:
            return hit
    return None


def _task_info(roots: list[dict], split: str) -> tuple[str, str, str | None]:
    """Return (task_id, question, groundtruth) from the function-call logs."""
    fn = "answer_single_question" if split == "GAIA" else "process_item"
    span = _find(roots, lambda s: any(l.get("body", {}).get("function.name") == fn for l in s.get("logs", [])))
    args = {}
    if span:
        log = next(l for l in span["logs"] if l.get("body", {}).get("function.name") == fn)
        args = log["body"].get("function.arguments", {})
    if split == "GAIA":
        ex = args.get("example", {})
        return ex.get("task_id", ""), ex.get("question", ""), ex.get("true_answer")
    item = args.get("item", {})
    return item.get("instance_id", ""), item.get("problem_statement", ""), item.get("patch")


def _agent_task(roots: list[dict]) -> str:
    """Task text given to the top agent; for truncated traces with no AGENT span, the first LLM call's user message."""
    agent = _find(roots, lambda s: _kind(s) == "AGENT")
    raw = (agent or {}).get("span_attributes", {}).get("input.value", "")
    if raw:
        try:
            return json.loads(raw).get("task", raw)
        except (json.JSONDecodeError, AttributeError):
            return raw
    llm = _find(roots, lambda s: _kind(s) == "LLM")
    attrs = (llm or {}).get("span_attributes", {})
    i = 0
    while f"llm.input_messages.{i}.message.role" in attrs:
        if attrs[f"llm.input_messages.{i}.message.role"] == "user":
            return str(attrs.get(f"llm.input_messages.{i}.message.content", "")).removeprefix("New task:").strip()
        i += 1
    return ""


def convert(trace: dict, annotation: dict, split: str, repaired: bool = False) -> dict:
    """One TRAIL (trace, annotation) pair -> one Who&When-style record."""
    roots = trace["spans"]
    task_id, question, groundtruth = _task_info(roots, split)
    history = [{"content": _agent_task(roots) or question, "role": "human"}] + flatten_spans(roots)
    index = {h["span_id"]: i for i, h in enumerate(history) if "span_id" in h}

    errors, unresolved = [], []
    for e in annotation.get("errors", []):
        if e.get("location") in index:
            errors.append({"step": index[e["location"]], **e})
        else:
            unresolved.append(e)
    errors.sort(key=lambda e: e["step"])
    error_steps = sorted({e["step"] for e in errors})
    first = error_steps[0] if error_steps else None

    return {
        "history": history,
        "question": question,
        "groundtruth": groundtruth,
        "is_corrected": False,
        "question_ID": trace["trace_id"],
        "mistake_agent": history[first]["role"] if first is not None else "",
        "mistake_step": str(first) if first is not None else "",
        "mistake_reason": " | ".join(e["description"] for e in errors if e["step"] == first),
        "split": split,
        "task_id": task_id,
        "error_steps": error_steps,
        "errors": errors,
        "unresolved_errors": unresolved,
        "scores": annotation.get("scores", []),
        "annotation_repaired": repaired,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data/trail/trail_github/benchmarking")
    ap.add_argument("--out", default="data/trail/whowhen_format")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    for split, ann_dir in SPLITS.items():
        dst = out / split.replace(" ", "_")
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for tp in sorted((src / "data" / split).glob("*.json")):
            ann, repaired = load_annotation(src / ann_dir / tp.name)
            rec = convert(json.loads(tp.read_text()), ann, split, repaired)
            (dst / tp.name).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            n += 1
        print(f"{split}: wrote {n} records to {dst}")


if __name__ == "__main__":
    main()
