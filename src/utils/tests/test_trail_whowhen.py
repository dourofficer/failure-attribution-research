"""Properties of the TRAIL -> Who&When flattening: order, label mapping, no lost labels."""

import json
from pathlib import Path

import pytest

from src.utils.trail_whowhen import SPLITS, convert, load_annotation

SRC = Path("data/trail/trail_github/benchmarking")


def _span(sid, name, kind=None, children=(), **attrs):
    a = dict(attrs)
    if kind:
        a["openinference.span.kind"] = kind
    return {"span_id": sid, "span_name": name, "span_attributes": a, "child_spans": list(children), "logs": []}


def _toy():
    llm1 = _span("l1", "LiteLLMModel.__call__", "LLM", **{"llm.output_messages.0.message.content": "plan"})
    llm2 = _span("l2", "LiteLLMModel.__call__", "LLM", **{
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "web_search",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"q": "x"}'})
    tool = _span("t1", "SearchInformationTool", "TOOL", **{"input.value": "x", "output.value": "y"})
    sub = _span("a2", "ToolCallingAgent.run", "AGENT", [_span("c2", "Step 1", "CHAIN", [llm2, tool])])
    top = _span("a1", "CodeAgent.run", "AGENT", [llm1, _span("c1", "Step 1", "CHAIN", [sub])], **{
        "smolagents.managed_agents.0.name": "search_agent", "input.value": json.dumps({"task": "Q?"})})
    fmt = _span("l3", "LiteLLMModel.__call__", "LLM", **{"llm.output_messages.0.message.content": "FINAL ANSWER: y"})
    return {"trace_id": "toy", "spans": [_span("r", "main", None, [top, fmt])]}


def test_toy_mapping():
    ann = {"errors": [{"location": "t1", "category": "A", "description": "d1", "impact": "LOW", "evidence": ""},
                      {"location": "l2", "category": "B", "description": "d2", "impact": "HIGH", "evidence": ""},
                      {"location": "zz", "category": "C", "description": "d3", "impact": "LOW", "evidence": ""}],
           "scores": []}
    rec = convert(_toy(), ann, "GAIA")
    h = rec["history"]
    assert [x.get("span_id") for x in h] == [None, "l1", "l2", "t1", "l3"]
    assert h[0] == {"content": "Q?", "role": "human"}
    assert [x["role"] for x in h[1:]] == ["manager", "search_agent", "SearchInformationTool", "answer_formatter"]
    assert h[2]["content"] == 'Tool call: web_search({"q": "x"})'
    assert rec["error_steps"] == [2, 3] and rec["mistake_step"] == "2" and rec["mistake_agent"] == "search_agent"
    assert [e["location"] for e in rec["unresolved_errors"]] == ["zz"]


def test_trailing_comma_repair(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"errors": [{"location": "x"},\n], "scores": [],}')
    data, repaired = load_annotation(p)
    assert repaired and data["errors"] == [{"location": "x"}]


@pytest.mark.skipif(not SRC.exists(), reason="TRAIL data not downloaded")
def test_real_data_keeps_every_resolvable_label():
    for split, ann_dir in SPLITS.items():
        for tp in sorted((SRC / "data" / split).glob("*.json")):
            trace = json.loads(tp.read_text())
            ann, _ = load_annotation(SRC / ann_dir / tp.name)
            rec = convert(trace, ann, split)
            assert len(rec["errors"]) + len(rec["unresolved_errors"]) == len(ann["errors"])
            for e in rec["errors"]:
                step = rec["history"][e["step"]]
                assert step["span_id"] == e["location"] and step["span_kind"] in ("LLM", "TOOL")
            assert rec["history"][0]["role"] == "human" and rec["history"][0]["content"]
