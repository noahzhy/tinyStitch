"""Source-only depth adaptation, selecting checkpoints on a disjoint scene."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import load_supervised
from .inference import load_model
from .geometry import correspondence, backproject, project, sample


def depth_loss(pred, truth, edge_weight=0):
    valid = truth > 0
    if not valid.any():
        raise ValueError("Empty depth supervision")
    logp = pred.clamp_min(0.001).log()
    logt = truth.clamp_min(0.001).log()
    result = (logp[valid] - logt[valid]).abs().mean()
    for axis in (-1, -2):
        mask = valid.narrow(axis, 1, valid.shape[axis] - 1) & valid.narrow(
            axis, 0, valid.shape[axis] - 1
        )
        if mask.any():
            result = (
                result
                + 0.25
                * (logp.diff(dim=axis)[mask] - logt.diff(dim=axis)[mask]).abs().mean()
            )
    if edge_weight:
        edges = torch.zeros_like(valid)
        for axis in (-1, -2):
            jump = logt.diff(dim=axis).abs() > 0.015
            if axis == -1:
                edges[..., 1:] |= jump
                edges[..., :-1] |= jump
            else:
                edges[..., 1:, :] |= jump
                edges[..., :-1, :] |= jump
        edges = F.max_pool2d(edges.float()[:, None], 3, 1, 1)[:, 0].bool() & valid
        if edges.any():
            result = result + edge_weight * (logp[edges] - logt[edges]).abs().mean()
    return result


def multiscale_gradient_loss(pred, truth):
    """Supervise depth steps at several spatial offsets without smoothing across them."""
    valid = truth > 0
    logp, logt = pred.clamp_min(0.001).log(), truth.clamp_min(0.001).log()
    terms = []
    for distance in (2, 4, 8):
        for axis in (-1, -2):
            length = truth.shape[axis] - distance
            if length <= 0:
                continue
            a = lambda x: x.narrow(axis, distance, length)
            b = lambda x: x.narrow(axis, 0, length)
            mask = a(valid) & b(valid)
            if mask.any():
                terms.append(
                    ((a(logp) - b(logp)) - (a(logt) - b(logt)))[mask].abs().mean()
                )
    return torch.stack(terms).mean() if terms else pred.sum() * 0


def consistency_loss(pred, truth, K, T, stride=8):
    """Use visible source-source GT correspondence as training supervision only."""
    xy, visible = correspondence(truth[0], truth[1], K[0], K[1], T[0], T[1])
    good = visible[::stride, ::stride].reshape(-1)
    if not good.any():
        return pred.sum() * 0
    points = backproject(pred[0], K[0], T[0])[::stride, ::stride].reshape(-1, 3)
    predicted_xy, z = project(points, K[1], T[1])
    reference_xy = xy[::stride, ::stride].reshape(-1, 2)
    target_depth = sample(pred[1, None], reference_xy)[:, 0]
    good = good & (z > 0) & (target_depth > 0)
    if not good.any():
        return pred.sum() * 0
    depth_residual = (z[good].log() - target_depth[good].log()).abs().mean()
    pixel_residual = (
        predicted_xy[good] - reference_xy[good]
    ).abs().mean() / truth.shape[-1]
    return depth_residual + pixel_residual


def run(
    data,
    checkpoint,
    output,
    steps=1200,
    seed=20261007,
    consistency_weight=0,
    edge_weight=0,
    skip_test=False,
    multiscale_weight=0,
):
    if steps < 1 or consistency_weight < 0 or edge_weight < 0 or multiscale_weight < 0:
        raise ValueError("Need positive steps and nonnegative loss weights")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    root = Path(data)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        raise ValueError("Use a fresh depth experiment directory")
    start = time.perf_counter()
    model, state = load_model(checkpoint)
    model.requires_grad_(False)
    model.pose.depth.requires_grad_(True)
    if model.pose.refinement_patch:
        model.pose.depth_refiner.requires_grad_(True)
    cached = []
    for entry in json.loads((root / "manifest.json").read_text())["scenes"]:
        if skip_test and entry["split"] == "test":
            continue
        rgb, K, T, depth = load_supervised(root / entry["path"])
        with torch.no_grad():
            tokens = torch.cat([model.jepa.encoder(image[None])[0] for image in rgb])
        cached.append(dict(entry=entry, tokens=tokens, truth=depth, K=K, T=T))
        print(
            json.dumps({"cached": entry["seed"], "split": entry["split"]}), flush=True
        )
    train = [c for c in cached if c["entry"]["split"] == "train"]
    val = [c for c in cached if c["entry"]["split"] == "val"]
    if not train or not val:
        raise ValueError("Need disjoint train/validation scenes")
    tokens = torch.cat([c["tokens"] for c in train])
    truth = torch.cat([c["truth"] for c in train])
    height, width = truth.shape[-2:]
    gh, gw = height // model.config["patch"], width // model.config["patch"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0002, weight_decay=0.0001)

    def evaluate(c):
        prediction = torch.cat(
            [
                model.pose.depths(c["tokens"][i : i + 4], gh, gw, (height, width))
                for i in range(0, len(c["tokens"]), 4)
            ]
        )
        valid = c["truth"] > 0
        return float(
            ((prediction[valid] - c["truth"][valid]).abs() / c["truth"][valid]).mean()
        )

    with torch.no_grad():
        before = {c["entry"]["seed"]: evaluate(c) for c in cached}
    best = sum(before[c["entry"]["seed"]] for c in val) / len(val)
    beststep = 0
    bestweights = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if k.startswith(("pose.depth.", "pose.depth_refiner."))
    }
    history = []
    for step in range(1, steps + 1):
        if consistency_weight or edge_weight:
            scene = train[int(torch.randint(len(train), (1,)))]
            gap = int((1, 2, 4, 8)[int(torch.randint(4, (1,)))])
            first = int(torch.randint(len(scene["tokens"]) - gap, (1,)))
            ids = [first, first + gap]
            prediction = model.pose.depths(
                scene["tokens"][ids], gh, gw, (height, width)
            )
            loss = depth_loss(prediction, scene["truth"][ids], edge_weight)
            loss = loss + consistency_weight * consistency_loss(
                prediction, scene["truth"][ids], scene["K"][ids], scene["T"][ids]
            )
        else:
            ids = torch.randint(len(tokens), (2,))
            prediction = model.pose.depths(tokens[ids], gh, gw, (height, width))
            loss = depth_loss(prediction, truth[ids], edge_weight)
        loss = loss + multiscale_weight * multiscale_gradient_loss(
            prediction,
            scene["truth"][ids] if consistency_weight or edge_weight else truth[ids],
        )
        if not torch.isfinite(loss):
            raise ValueError("Depth optimization diverged")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                value = sum(evaluate(c) for c in val) / len(val)
            if value < best:
                best, beststep = value, step
                bestweights = {
                    k: v.clone()
                    for k, v in model.state_dict().items()
                    if k.startswith(("pose.depth.", "pose.depth_refiner."))
                }
            row = dict(
                step=step,
                loss=float(loss.detach()),
                validation_abs_rel=value,
                selected_step=beststep,
            )
            history.append(row)
            print(json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(bestweights, strict=False)
    with torch.no_grad():
        rows = [
            dict(
                seed=c["entry"]["seed"],
                split=c["entry"]["split"],
                before_abs_rel=before[c["entry"]["seed"]],
                after_abs_rel=evaluate(c),
            )
            for c in cached
        ]
    protocol = dict(
        steps=steps,
        selected_step=beststep,
        seed=seed,
        consistency_weight=consistency_weight,
        edge_weight=edge_weight,
        multiscale_weight=multiscale_weight,
        geometry_resolution=[height, width],
        target_view_used=False,
        test_loaded=not skip_test,
        train_seeds=[c["entry"]["seed"] for c in train],
        validation_seeds=[c["entry"]["seed"] for c in val],
        source_checkpoint_sha256=hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
        trainable=["pose.depth", "pose.depth_refiner"],
        elapsed_seconds=time.perf_counter() - start,
    )
    saved = copy.deepcopy(state)
    saved.update(model=model.state_dict(), depth_adaptation=protocol)
    torch.save(saved, out / "model.pt")
    (out / "report.json").write_text(
        json.dumps(dict(protocol=protocol, rows=rows), indent=2)
    )
    print(json.dumps({"status": "complete", "selected_step": beststep}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--consistency-weight", type=float, default=0)
    p.add_argument("--edge-weight", type=float, default=0)
    p.add_argument("--multiscale-weight", type=float, default=0)
    p.add_argument("--skip-test", action="store_true")
    run(**vars(p.parse_args()))
