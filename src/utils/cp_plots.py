"""Training-run figures for the conformal-training arms.

Inputs
    --arm NAME=FOLDER   repeated; an arm description and its experiment folder (reads `sweep/*.json`)
    --out               folder for the figures

Outputs
    training_curves.png     the best finished run of every arm: classification loss, set-size loss, threshold,
                            validation top-1 accuracy, validation relative set size, validation coverage
    all_runs.png            every finished run of every arm: validation relative set size per epoch
    sweep_sensitivity.png   best validation relative set size of every run, grouped by each swept setting
    threshold_tracking.png  arms with a learned threshold: q against the exact quantile, one line per step size

All curves are per-epoch values logged by `cp_train.train` (seed 0). Validation numbers are at alpha = 0.1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.cp_partial import finished_runs

COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600"]   # fixed order: one per arm, in the order given
STYLES = ["-", "--", "-.", ":"]
TEXT, MUTED = "#191919", "#666666"
SWEPT = ["lr", "lam", "tau", "gamma", "q_fix"]
LABELS = {"lr": "learning rate", "lam": "set-size weight λ", "tau": "temperature τ", "gamma": "threshold step size γ", "q_fix": "fixed threshold"}


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def series(run: dict, key: str) -> list:
    return [e.get(key) for e in run["history"]]


def training_curves(arms: dict[str, list[dict]], out: Path) -> None:
    panels = [("cls", "Classification loss (training)"), ("size", "Set-size loss (training)"), ("q", "Threshold q used by the set-size loss"),
              ("val_top1", "Validation top-1 accuracy"), ("val_size_rel", "Validation relative set size at 90% coverage"), ("val_coverage", "Validation coverage (target 0.9)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for i, (name, runs) in enumerate(arms.items()):
            vals = series(runs[0], key)
            if key in ("size", "q") and not runs[0]["hp"].get("size_splits"):
                continue                                   # cross-entropy arms have no set-size loss and no threshold
            ax.plot(range(len(vals)), vals, color=COLORS[i], ls=STYLES[i], lw=2, label=name)
            ax.plot(runs[0]["best_epoch"], vals[runs[0]["best_epoch"]], "o", color=COLORS[i], ms=7, mec="white", mew=1.5)
        style(ax, title)
    axes[1, 2].axhline(0.9, color=MUTED, lw=1, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(arms), frameon=False, fontsize=10)
    fig.suptitle("Best finished run of each arm (seed 0); dots mark the selected epoch", fontsize=12, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / "training_curves.png", dpi=140)
    plt.close(fig)


def all_runs(arms: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.2), sharey=True, squeeze=False)
    for i, (ax, (name, runs)) in enumerate(zip(axes[0], arms.items())):
        for run in runs:
            ax.plot(series(run, "val_size_rel"), color=COLORS[i], lw=0.6, alpha=0.35)
        ax.plot(np.median([series(r, "val_size_rel") for r in runs], axis=0), color=TEXT, lw=2, label="median over runs")
        style(ax, f"{name}\n{len(runs)} runs")
        ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel("validation relative set size", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "all_runs.png", dpi=140)
    plt.close(fig)


def sweep_sensitivity(arms: dict[str, list[dict]], out: Path) -> None:
    swept = {n: [k for k in SWEPT if len({r["hp"].get(k) for r in runs}) > 1] for n, runs in arms.items()}
    rows = [n for n in arms if len(swept[n]) > 1]
    refs = {n: runs[0]["best_val"]["size_rel"] for n, runs in arms.items() if n not in rows}
    ncol = max(len(swept[n]) for n in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.8 * ncol, 3.6 * len(rows)), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for r, name in enumerate(rows):
        i = list(arms).index(name)
        for c in range(ncol):
            ax = axes[r, c]
            if c >= len(swept[name]):
                ax.axis("off")
                continue
            key = swept[name][c]
            values = sorted({run["hp"][key] for run in arms[name]})
            for x, v in enumerate(values):
                ys = [run["best_val"]["size_rel"] for run in arms[name] if run["hp"][key] == v]
                ax.plot(x + rng.uniform(-0.15, 0.15, len(ys)), ys, "o", color=COLORS[i], ms=4, alpha=0.5, mec="none")
                ax.plot([x - 0.25, x + 0.25], [np.median(ys)] * 2, color=TEXT, lw=2)
            for j, (ref, y) in enumerate(refs.items()):
                ax.axhline(y, color=MUTED, lw=1, ls=["--", ":"][j % 2], label=f"{ref}: {y:.3f}" if (r, c) == (0, 0) else None)
            ax.set_xticks(range(len(values)), [str(v) for v in values])
            ax.set_xlabel(LABELS[key], color=MUTED, fontsize=9)
            style(ax, name if c == 0 else "")
        axes[r, 0].set_ylabel("best validation relative set size", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Every finished sweep run (dot) by setting; bars are medians", fontsize=12, x=0.01, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "sweep_sensitivity.png", dpi=140)
    plt.close(fig)


def threshold_tracking(arms: dict[str, list[dict]], out: Path) -> None:
    for name, runs in arms.items():
        if runs[0]["hp"].get("mode") != "dpsm":
            continue
        best = runs[0]["hp"]
        same = [r for r in runs if all(r["hp"][k] == best[k] for k in ("lr", "lam", "tau"))]
        same.sort(key=lambda r: r["hp"]["gamma"])
        fig, axes = plt.subplots(1, len(same), figsize=(5 * len(same), 4), sharey=True, squeeze=False)
        for ax, run in zip(axes[0], same):
            ax.plot(series(run, "q"), color=COLORS[0], lw=2, label="learned q")
            ax.plot(series(run, "q_exact_d1"), color=COLORS[1], lw=2, ls="--", label="exact 0.9 quantile, classification half")
            ax.axvline(run["best_epoch"], color=MUTED, lw=1, ls=":")
            style(ax, f"step size γ = {run['hp']['gamma']} (selected epoch {run['best_epoch']})")
            ax.set_xlabel("epoch", color=MUTED, fontsize=9)
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"{name}: lr {best['lr']}, λ {best['lam']}, τ {best['tau']}", fontsize=11, x=0.01, ha="left", color=TEXT)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / "threshold_tracking.png", dpi=140)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=FOLDER, at most four")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    arms = {a.split("=", 1)[0]: finished_runs(Path(a.split("=", 1)[1])) for a in args.arm}
    args.out.mkdir(parents=True, exist_ok=True)
    training_curves(arms, args.out)
    all_runs(arms, args.out)
    sweep_sensitivity(arms, args.out)
    threshold_tracking(arms, args.out)
    print(f"wrote 4 figures to {args.out}")


if __name__ == "__main__":
    main()
