"""Multiscene orthographic field training, with frozen predicted source geometry.

Orthographic RGB/depth GT is explicitly training supervision on train scenes.
Validation chooses weights. Test orthographic labels are never loaded here.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import load_supervised
from .geometry import pixels
from .inference import load_model
from .model import Pipeline
from .optimization import predict_cameras
from .overfit import make_cache, cache_prediction
from .render import opaque_weights
from .multiview import indices as sequence_indices


def ray_prediction(field, cache, ids):
    points = cache["points"][ids]
    direction = cache["directions"][ids]
    features = tuple(v[ids].reshape(-1, v.shape[-1]) for v in cache["features"])
    features = (*features[:4], features[4][:, 0])
    sigma, color, support = field.evaluate(
        points.reshape(-1, 3), direction.reshape(-1, 3), features
    )
    steps = cache["steps"][ids]
    sigma = sigma.reshape_as(steps)
    spacing = torch.cat(
        (steps[:, 1:] - steps[:, :-1], steps[:, -1:] - steps[:, -2:-1]), -1
    )
    alpha = -torch.expm1(
        -sigma * spacing * cache["ray_dirs"][ids].norm(dim=-1, keepdim=True)
    )
    trans = torch.cumprod(
        torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-8), 1), 1
    )[:, :-1]
    soft = alpha * trans
    weights = opaque_weights(soft)
    rgb = (weights[..., None] * color.reshape(len(ids), -1, 3)).sum(1)
    return rgb, soft, steps, support.reshape_as(steps)


def run(
    data,
    checkpoint,
    output,
    steps=1800,
    seed=20261008,
    rays=2048,
    input_counts=(60,),
    feature_center_mapping=False,
    local_sources=None,
    source_geometry="predicted",
    warm_field=False,
):
    if source_geometry not in ("predicted", "oracle"):
        raise ValueError("Unknown training source geometry")
    if local_sources is not None and local_sources < 2:
        raise ValueError("Need >=2 local sources")
    if not input_counts or any(n not in (10, 30, 60) for n in input_counts):
        raise ValueError("Unsupported training input counts")
    if min(steps, rays) < 1:
        raise ValueError("Invalid rendering budget")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    root = Path(data)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        raise ValueError("Use a fresh renderer experiment")
    old, state = load_model(checkpoint)
    config = dict(
        old.config,
        field_arch="fourier",
        field_bands=4,
        color_residual_limit=0.15,
        appearance_geometry=True,
        feature_center_mapping=feature_center_mapping,
    )
    model = Pipeline(**config)
    model.load_state_dict(
        {k: v for k, v in old.state_dict().items() if not k.startswith("field.")},
        strict=False,
    )
    if warm_field:
        model.field.load_state_dict(old.field.state_dict())
    model.eval().requires_grad_(False)
    model.field.requires_grad_(True)
    entries = json.loads((root / "manifest.json").read_text())["scenes"]
    cached = []
    for entry in entries:
        if entry["split"] == "test":
            continue
        path = root / entry["path"]
        rgb, K, truth_T, truth_depth = load_supervised(path)
        with torch.no_grad():
            tokens, structure, appearance, depth = model.features_batched(rgb, 1)
            T, _ = predict_cameras(model, tokens)
            if source_geometry == "oracle":
                T, depth = truth_T, truth_depth
        # GT only defines this training/validation target; source poses/depth are predicted.
        raw_T = np.load(path / "labels/geometry.npz")["T"]
        raw_d = np.load(path / "labels/0030.npy")
        scale = float(np.median(raw_d[raw_d > 0]))
        target = np.load(path / "orthographic/geometry.npz")
        target_T = torch.tensor(target["T"] @ np.linalg.inv(raw_T[30]))
        target_T[:3, 3] /= scale
        target_K = torch.tensor(target["K"])
        target_K[0, 0] *= scale
        target_K[1, 1] *= scale
        gt_depth = torch.tensor(target["depth"]) / scale
        gt = (
            torch.tensor(
                np.array(Image.open(path / "orthographic/rgb.png").convert("RGB"))
            ).float()
            / 255
        )
        h, w = gt_depth.shape
        valid = gt_depth > 0
        xy = pixels(h, w)[..., :2][valid]
        # Fixed training sample cache; no mask based on predicted support.
        take = torch.randperm(len(xy))[:rays]
        xy = xy[take]
        color = gt[valid][take]
        target_depth = gt_depth[valid][take]
        if local_sources is not None:
            order = torch.argsort(xy[:, 0], stable=True)
            xy, color, target_depth = xy[order], color[order], target_depth[order]
        for input_count in input_counts:
            context_ids = sequence_indices(input_count, 60)
            with torch.no_grad():
                context_T, _ = predict_cameras(model, tokens[context_ids])
                if source_geometry == "oracle":
                    context_T = truth_T[context_ids]
            for count in (
                (local_sources,)
                if local_sources is not None
                else ((6, 12) if entry["split"] == "train" else (12,))
            ):
                chosen = (
                    np.linspace(0, input_count - 1, min(count, input_count))
                    .round()
                    .astype(int)
                )
                if local_sources is not None:
                    chosen = np.arange(input_count)
                ids = context_ids[chosen]
                cache = make_cache(
                    model.field,
                    rgb[ids],
                    K[ids],
                    context_T[chosen],
                    structure[ids],
                    appearance[ids],
                    depth[ids],
                    target_K,
                    target_T,
                    xy,
                    samples=32,
                    sampling="hybrid",
                    projection="orthographic",
                    chunk=64 if local_sources is not None else 256,
                    local_sources=local_sources,
                )
                cached.append(
                    dict(
                        entry=entry,
                        cache=cache,
                        color=color,
                        target_depth=target_depth,
                        count=len(chosen),
                        input_count=input_count,
                    )
                )
        print(
            json.dumps({"cached": entry["seed"], "split": entry["split"]}), flush=True
        )
    train = [c for c in cached if c["entry"]["split"] == "train"]
    val = [c for c in cached if c["entry"]["split"] == "val"]
    if not train or not val:
        raise ValueError("Need disjoint rendering train/validation scenes")

    def validation():
        with torch.no_grad():
            return float(
                torch.stack(
                    [
                        (cache_prediction(model.field, c["cache"])[0] - c["color"])
                        .square()
                        .mean()
                        for c in val
                    ]
                ).mean()
            )

    optimizer = torch.optim.AdamW(
        model.field.parameters(), lr=0.0004, weight_decay=0.0001
    )
    initial = validation()
    best = initial
    beststep = 0
    beststate = copy.deepcopy(model.field.state_dict())
    history = []
    for step in range(1, steps + 1):
        c = train[int(torch.randint(len(train), (1,)))]
        ids = torch.randint(len(c["color"]), (96,))
        prediction, raw, ts, support = ray_prediction(model.field, c["cache"], ids)
        distances = (ts - c["target_depth"][ids, None]).abs()
        nearest = distances.argmin(-1)
        probability = raw / raw.sum(-1, keepdim=True).clamp_min(1e-8)
        observable = support.gather(1, nearest[:, None])[:, 0] > 0.001
        classification = (
            -probability.gather(1, nearest[:, None])[:, 0].clamp_min(1e-8).log()
        )
        surface_loss = (
            classification[observable].mean()
            if observable.any()
            else prediction.sum() * 0
        )
        loss = (
            (prediction - c["color"][ids]).square().mean()
            + 0.01 * surface_loss
            + 0.05 * (probability * distances.square()).sum(-1).mean()
            + 0.001 * (1 - raw.sum(-1)).square().mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("Rendering optimization diverged")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.field.parameters(), 5)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            v = validation()
            if v < best:
                best, beststep, beststate = (
                    v,
                    step,
                    copy.deepcopy(model.field.state_dict()),
                )
            row = dict(
                step=step,
                loss=float(loss.detach()),
                validation_psnr=-10 * np.log10(v),
                selected_step=beststep,
            )
            history.append(row)
            print(json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
    model.field.load_state_dict(beststate)
    protocol = dict(
        seed=seed,
        steps=steps,
        selected_step=beststep,
        train_seeds=[e["seed"] for e in entries if e["split"] == "train"],
        validation_seeds=[e["seed"] for e in entries if e["split"] == "val"],
        target_gt_used_for_training=True,
        test_target_loaded=False,
        source_geometry=source_geometry,
        warm_field=warm_field,
        cached_ray_count=rays,
        training_input_counts=list(input_counts),
        source_counts=[local_sources] if local_sources is not None else [6, 12],
        local_sources=local_sources,
        cache_order=(
            "sorted target columns" if local_sources is not None else "random pixels"
        ),
        field_bands=4,
        feature_center_mapping=feature_center_mapping,
        initial_validation_psnr=-10 * np.log10(initial),
        selected_validation_psnr=-10 * np.log10(best),
        source_checkpoint_sha256=hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
        elapsed_seconds=time.perf_counter() - start,
    )
    saved = copy.deepcopy(state)
    saved.update(
        model=model.state_dict(), config=model.config, renderer_adaptation=protocol
    )
    torch.save(saved, out / "model.pt")
    (out / "report.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps({"status": "complete", "selected_step": beststep}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--rays", type=int, default=2048)
    p.add_argument("--input-counts", nargs="+", type=int, default=[60])
    p.add_argument("--feature-center-mapping", action="store_true")
    p.add_argument("--local-sources", type=int)
    p.add_argument(
        "--source-geometry", choices=["predicted", "oracle"], default="predicted"
    )
    p.add_argument("--warm-field", action="store_true")
    run(**vars(p.parse_args()))
