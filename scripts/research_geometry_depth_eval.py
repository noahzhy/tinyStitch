"""Depth-only evaluation; test labels enter metrics after frozen predictions."""

import argparse, json, time
from pathlib import Path
import torch
from backend.research.inference import load_model
from backend.research.data import load_input, load_supervised
from backend.research.depthadapt import multiscale_gradient_loss


def run(checkpoint, output):
    torch.set_num_threads(4)
    model, _ = load_model(checkpoint, "mps")
    model.eval().requires_grad_(False)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(738018, 738024):
        scene = Path("data/research/mixed-bays-v1") / str(seed)
        rgb, K = load_input(scene, "mps")
        with torch.no_grad():
            _, _, _, pred = model.features_batched(rgb, 4)
        torch.mps.synchronize()
        # Metrics only: prediction is complete before GT source geometry is read.
        _, _, _, truth = load_supervised(scene, "mps")
        valid = truth > 0
        edges = torch.zeros_like(valid)
        log = truth.clamp_min(0.001).log()
        for axis in (-1, -2):
            jump = log.diff(dim=axis).abs() > 0.015
            if axis == -1:
                edges[..., 1:] |= jump
                edges[..., :-1] |= jump
            else:
                edges[..., 1:, :] |= jump
                edges[..., :-1, :] |= jump
        edges = edges & valid
        rel = (pred - truth).abs() / truth.clamp_min(0.001)
        rows.append(
            dict(
                seed=seed,
                abs_rel=float(rel[valid].mean()),
                edge_abs_rel=float(rel[edges].mean()),
                multiscale_gradient=float(multiscale_gradient_loss(pred, truth)),
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    (out / "geometry-evaluation.json").write_text(
        json.dumps(
            dict(
                checkpoint=checkpoint,
                source_resolution=[256, 192],
                views=60,
                test_fitting=False,
                rows=rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))
