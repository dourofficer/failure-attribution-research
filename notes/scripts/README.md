# Scripts for the notes

Helpers that the survey build and its generated parts call. They live here, next to the writing they serve, and not under the repo's `scripts/` (which holds experiment launchers only).

Expected here, but missing from the repo as of 2026-11-21:

- `merge_bib.py`: called by `../build.sh` to merge `bib/refs.bib` and every `*/refs*.bib` into `bib/refs_merged.bib`. Without it `build.sh` fails at its first step.
- `render_traj.py`: generates each dataset's `traj.tex`.
- `make_figures.py`: generates each dataset's `figures/`.
- `report_layout.py`: the slug → group map used by the two above.
