# failure-attribution-research

A research project on **failure attribution**.

- `MANUAL.md`: how the project is run. For people; start here.
- `CLAUDE.md`: the same workflow as rules for agents.
- `docs/STATE.md`: what is active right now.
- `docs/decisions/index.md`: what has been decided, and which lines are closed.
- `notes/`: the researcher's own writing: dataset survey, proposal, tutorial.
- `papers/`: reference papers.
- `data/`: the attribution datasets, git-ignored; sources in `data/SOURCES.md`, backed up with `data/push_to_hf.py`.

## Environment

Verified on 2026-11-21 with a bf16 `flash_attention_2` forward pass (`Qwen/Qwen3-0.6B`).

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| torch | 2.6.0+cu124 (pinned to exactly 2.6.0; the flash-attn wheel is built for it) |
| CUDA (wheel) | 12.4 |
| flash-attn | 2.7.4.post1, prebuilt wheel pinned by URL in `pyproject.toml` |
| cxx11abi | FALSE |
| transformers | 5.23.0 |
| GPUs | 2× RTX 3090, 24 GB each (Ampere, driver 580.178.04) |

To change torch, pick the matching flash-attn wheel first (same Python, torch, CUDA and cxx11abi) and never let it build from source.

```bash
uv sync             # install
uv run pytest -q    # test
```
