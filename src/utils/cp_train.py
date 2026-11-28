"""Train the step scorer with cross-entropy, optionally with a soft set-size loss (Work Package 1, Algorithm 1).

Inputs
    splits   {name: Split} on the training device (see cp_data), with at least `val` and the training splits
    hp       dict of hyperparameters:
               mode        "ce" | "size_fixed" | "dpsm"
               cls_splits  splits carrying the classification loss (D_1, or D_1 and D_2 for full-data "ce")
               size_splits splits carrying the set-size loss (D_2); unused for "ce"
               lr, weight_decay, batch_size, epochs, alpha
               lam, tau    set-size loss weight and temperature ("size_fixed", "dpsm")
               q_fix       the constant threshold ("size_fixed")
               gamma, q0   threshold step size and initial value ("dpsm")
    seed

Outputs
    train()   {"logits": {split: [np.ndarray per trace]} at the best epoch, "history": one dict per epoch,
               "best_epoch", "best_val"}. The best epoch is the one with the lowest calibrated relative set
               size on validation at hp["alpha"] (two fixed halves, calibrate on one, score the other, both ways).

Losses (pi = softmax of the logits over the real steps of a trace, S = 1 - pi)
    classification   mean over B1 of -log(pi_y + eps)
    set size         mean over B2 of (1/T) sum_t sigmoid((q - 1 + pi_t) / tau), q not differentiated
    threshold step   q <- clip(q - gamma * (alpha - fraction of B1 with 1 - pi_y >= q), 0, 1), using pi from
                     before the encoder step
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.utils import cp_eval
from src.utils.cp_data import Split, merge, pad_batch
from src.utils.cp_scorer import StepScorer

EPS = 1e-6


def cls_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pi_y = torch.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    return -torch.log(pi_y + EPS).mean()


def size_loss(logits: torch.Tensor, mask: torch.Tensor, q: float, tau: float) -> torch.Tensor:
    pi = torch.softmax(logits, dim=1)
    soft = torch.sigmoid((q - 1 + pi) / tau) * mask
    return (soft.sum(1) / mask.sum(1)).mean()


def threshold_step(q: float, s_label: torch.Tensor, alpha: float, gamma: float) -> float:
    miss = (s_label >= q).float().mean().item()
    return min(1.0, max(0.0, q - gamma * (alpha - miss)))


@torch.no_grad()
def predict(model: StepScorer, split: Split, batch_size: int = 256) -> list[np.ndarray]:
    model.eval()
    out = []
    for a in range(0, len(split), batch_size):
        idx = range(a, min(a + batch_size, len(split)))
        h, mask, _ = pad_batch(split, idx)
        logits = model(h, mask).cpu().numpy()
        out.extend(logits[b, : int(mask[b].sum())] for b in range(len(idx)))
    return out


def val_metric(logits: list[np.ndarray], y: np.ndarray, alpha: float) -> dict:
    scores = [cp_eval.hps(cp_eval.softmax(l)) for l in logits]
    return cp_eval.two_fold(scores, y, alpha, seed=0)


def train(splits: dict[str, Split], hp: dict, seed: int, eval_splits: list[str], progress=None) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d1 = merge(splits, hp["cls_splits"])
    d2 = merge(splits, hp["size_splits"]) if hp["mode"] != "ce" else None
    dev = d1.emb[0].device
    model = StepScorer(d1.emb[0].shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    b = hp["batch_size"]
    steps_per_epoch = math.ceil(len(d1) / b)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch * hp["epochs"])
    q = hp.get("q0", 0.5) if hp["mode"] == "dpsm" else hp.get("q_fix")
    val_y = np.array(splits["val"].y)

    history, best = [], None
    for epoch in range(hp["epochs"]):
        model.train()
        perm1 = rng.permutation(len(d1))
        tot_cls = tot_size = 0.0
        for s in range(steps_per_epoch):
            h, mask, y = pad_batch(d1, perm1[s * b : (s + 1) * b])
            logits = model(h, mask)
            loss_c = cls_loss(logits, y)
            loss = loss_c
            if d2 is not None:
                h2, mask2, _ = pad_batch(d2, rng.integers(0, len(d2), size=b))
                loss_s = size_loss(model(h2, mask2), mask2, q, hp["tau"])
                loss = loss + hp["lam"] * loss_s
                tot_size += loss_s.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if hp["mode"] == "dpsm":
                s_label = 1 - torch.softmax(logits.detach(), dim=1).gather(1, y[:, None]).squeeze(1)
                q = threshold_step(q, s_label, hp["alpha"], hp["gamma"])
            tot_cls += loss_c.item()
        val_logits = predict(model, splits["val"])
        vm = val_metric(val_logits, val_y, hp["alpha"])
        row = {"epoch": epoch, "cls": tot_cls / steps_per_epoch, "size": tot_size / steps_per_epoch, "q": q, **{f"val_{k}": v for k, v in vm.items()}}
        if hp["mode"] == "dpsm":   # pinball lag: the exact (1 - alpha) quantile of the label scores on D_1
            d1_scores = cp_eval.label_scores([cp_eval.hps(cp_eval.softmax(l)) for l in predict(model, d1)], np.array(d1.y))
            row["q_exact_d1"] = float(np.quantile(d1_scores, 1 - hp["alpha"]))
        history.append(row)
        if best is None or vm["size_rel"] < best["val"]["size_rel"]:
            best = {"epoch": epoch, "val": vm, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if progress is not None:
            progress(row)

    model.load_state_dict(best["state"])
    return {
        "logits": {name: predict(model, splits[name]) for name in eval_splits},
        "history": history,
        "best_epoch": best["epoch"],
        "best_val": best["val"],
    }
