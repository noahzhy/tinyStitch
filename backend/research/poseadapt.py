"""Variable-baseline pose-head adaptation with scene-disjoint model selection.

Frozen JEPA/cross attention descriptors are cached; only the relational MLP trains.
No target-view RGB or geometry is loaded. Renderer/depth remain unchanged.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import load_supervised
from .geometry import rotation_error
from .inference import load_model
from .model import Pipeline
from .optimization import predict_cameras
from .multiview import indices


def pairs_for(n, max_gap=16):
    return [
        (i, j)
        for gap in (1, 2, 4, 6, 8, 12, 16)
        for i in range(n - gap)
        if gap <= max_gap
        for j in (i + gap,)
    ] + [
        (j, i)
        for gap in (1, 2, 4, 6, 8, 12, 16)
        for i in range(n - gap)
        if gap <= max_gap
        for j in (i + gap,)
    ]


def supervised_loss(prediction, truth):
    R, direction, length, confidence = prediction
    norm = truth[:, :3, 3].norm(dim=-1).clamp_min(1e-6)
    target_d = truth[:, :3, 3] / norm[:, None]
    # Log baseline loss gives dense short-baseline pairs equal weight.
    rot = (R - truth[:, :3, :3]).square().mean()
    direction_loss = (direction - target_d).square().mean()
    scale = F.smooth_l1_loss(length.log(), norm.log())
    # Stop gradient prevents confidence from reducing the regression loss.
    correct = (
        ((direction.detach() - target_d).norm(dim=-1) < 0.1)
        & ((length.detach() / norm - 1).abs() < 0.15)
        & (rotation_error(R.detach(), truth[:, :3, :3]) < 0.02)
    ).float()
    confidence_loss = F.binary_cross_entropy(confidence, correct)
    return 30 * rot + direction_loss + scale + 0.05 * confidence_loss


def metrics(pred, gt):
    centers = lambda t: -(t[:, :3, :3].transpose(1, 2) @ t[:, :3, 3, None])[:, :, 0]
    a, b = centers(pred), centers(gt)
    return {
        "extent_ratio": float((a[-1] - a[0]).norm() / (b[-1] - b[0]).norm()),
        "center_l2_mean": float((a - b).norm(dim=-1).mean()),
        "rotation_max_deg": float(
            rotation_error(pred[:, :3, :3], gt[:, :3, :3]).max() * 180 / torch.pi
        ),
    }


def run(data, checkpoint, output, steps=6000, seed=20261007, max_pair_gap=None):
    if max_pair_gap is not None and max_pair_gap < 1:
        raise ValueError("Need positive pair gap")
    if steps < 1:
        raise ValueError("Need positive training steps")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    root, out = Path(data), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        raise ValueError("Refusing to overwrite pose experiment")
    start = time.perf_counter()
    old, state = load_model(checkpoint)
    config = dict(old.config, pose_arch="relational", pose_tokens=192)
    model = Pipeline(**config)
    compatible = {
        k: v for k, v in old.state_dict().items() if not k.startswith("pose.head.")
    }
    model.load_state_dict(compatible, strict=False)
    model.eval().requires_grad_(False)
    model.pose.head.requires_grad_(True)
    manifest = json.loads((root / "manifest.json").read_text())
    cached = []
    for entry in manifest["scenes"]:
        rgb, K, T, depth = load_supervised(root / entry["path"])
        with torch.no_grad():
            groups = [model.jepa.encoder(rgb[i : i + 1])[0] for i in range(len(rgb))]
            tokens = torch.cat(groups)
            pairs = pairs_for(
                len(rgb),
                (
                    max_pair_gap
                    if max_pair_gap is not None
                    else (8 if manifest.get("capture_step") is not None else 16)
                ),
            )
            descriptors = []
            for k in range(0, len(pairs), 16):
                group = pairs[k : k + 16]
                descriptors.append(
                    model.pose.descriptor(
                        tokens[[i for i, j in group]], tokens[[j for i, j in group]]
                    )
                )
            target = torch.stack([T[j] @ torch.linalg.inv(T[i]) for i, j in pairs])
        cached.append(
            dict(
                entry=entry,
                tokens=tokens,
                T=T,
                descriptor=torch.cat(descriptors),
                truth=target,
            )
        )
        print(
            json.dumps(
                {"cached": entry["seed"], "split": entry["split"], "pairs": len(pairs)}
            ),
            flush=True,
        )
    train = [c for c in cached if c["entry"]["split"] == "train"]
    val = [c for c in cached if c["entry"]["split"] == "val"]
    if not train or not val or not any(c["entry"]["split"] == "test" for c in cached):
        raise ValueError("Need scene-disjoint train/val/test splits")
    x = torch.cat([c["descriptor"] for c in train])
    y = torch.cat([c["truth"] for c in train])
    optimizer = torch.optim.AdamW(
        model.pose.head.parameters(), lr=0.0004, weight_decay=0.0001
    )
    best, best_step, best_weights = float("inf"), 0, None
    history = []
    for step in range(1, steps + 1):
        ids = torch.randint(len(x), (128,))
        loss = supervised_loss(model.pose.decode(x[ids]), y[ids])
        if not torch.isfinite(loss):
            raise ValueError("Pose adaptation diverged")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.pose.head.parameters(), 5)
        optimizer.step()
        if step % 200 == 0 or step == steps:
            with torch.no_grad():
                vl = float(
                    torch.stack(
                        [
                            supervised_loss(
                                model.pose.decode(c["descriptor"]), c["truth"]
                            )
                            for c in val
                        ]
                    ).mean()
                )
            if vl < best:
                best, best_step = vl, step
                best_weights = copy.deepcopy(model.pose.head.state_dict())
            row = dict(
                step=step,
                train_loss=float(loss.detach()),
                validation_loss=vl,
                best_step=best_step,
            )
            history.append(row)
            print(json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
    model.pose.head.load_state_dict(best_weights)
    graph_validation = []
    with torch.no_grad():
        for c in val:
            for count in manifest.get("evaluation_counts", [10, 30, 60]):
                ids = indices(count, len(c["tokens"]))
                gt = c["T"][ids] @ torch.linalg.inv(c["T"][ids[count // 2]])
                chain, _ = predict_cameras(model, c["tokens"][ids], graph=False)
                graph, _ = predict_cameras(model, c["tokens"][ids], graph=True)
                graph_validation.append(
                    dict(
                        count=count, chain=metrics(chain, gt), graph=metrics(graph, gt)
                    )
                )
    enabled = sum(r["graph"]["center_l2_mean"] for r in graph_validation) < sum(
        r["chain"]["center_l2_mean"] for r in graph_validation
    )
    model.config["pose_graph"] = bool(enabled)
    report = []
    # Test labels first influence reported metrics only after model selection.
    with torch.no_grad():
        for c in cached:
            for count in manifest.get("evaluation_counts", [10, 30, 60]):
                ids = indices(count, len(c["tokens"]))
                gt = c["T"][ids] @ torch.linalg.inv(c["T"][ids[count // 2]])
                before, _ = predict_cameras(old, c["tokens"][ids], pair_batch=2)
                chain, _ = predict_cameras(
                    model, c["tokens"][ids], pair_batch=8, graph=False
                )
                after, edges = predict_cameras(model, c["tokens"][ids], pair_batch=8)
                report.append(
                    dict(
                        seed=c["entry"]["seed"],
                        split=c["entry"]["split"],
                        count=count,
                        before=metrics(before, gt),
                        after=metrics(after, gt),
                        chain=metrics(chain, gt),
                        confidence_mean=float(
                            torch.stack([e[3] for e in edges]).mean()
                        ),
                    )
                )
    protocol = dict(
        seed=seed,
        steps=steps,
        selected_step=best_step,
        best_validation_loss=best,
        train_seeds=[c["entry"]["seed"] for c in train],
        validation_seeds=[c["entry"]["seed"] for c in val],
        test_seeds=[
            c["entry"]["seed"] for c in cached if c["entry"]["split"] == "test"
        ],
        geometry_resolution=list(rgb.shape[-2:]),
        frozen=[
            "jepa",
            "pose.cross",
            "pose.depth",
            "pose.depth_refiner",
            "vae",
            "field",
        ],
        trainable="pose.head",
        target_view_used=False,
        max_pair_gap=max_pair_gap,
        pose_graph_enabled=enabled,
        pose_graph_validation=graph_validation,
        pose_graph_selection="mean validation camera-center error; no test-based selection",
        camera_intrinsics_conditioned=False,
        renderer_training_scenes=1,
        elapsed_seconds=time.perf_counter() - start,
        source_checkpoint_sha256=hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
    )
    saved = copy.deepcopy(state)
    saved.update(
        config=model.config,
        model=model.state_dict(),
        pose_adaptation=protocol,
        pose_rng_state=torch.get_rng_state(),
    )
    torch.save(saved, out / "model.pt")
    (out / "report.json").write_text(
        json.dumps(dict(protocol=protocol, rows=report), indent=2)
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_step": best_step,
                "elapsed": protocol["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=20261007)
    run(**vars(p.parse_args()))
