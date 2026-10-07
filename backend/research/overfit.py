"""Single-scene falsification experiment. Held-out RGB is read only after optimization.

Uses supervised cross-view correspondence as a declared auxiliary objective. This is
an overfitting/capacity probe, not evidence of pure JEPA or unseen-scene generalization.
"""

import argparse
import copy
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import generate, load_supervised
from .geometry import (
    correspondence,
    pixels,
    project,
    backproject,
    sample,
    rotation_error,
)
from .model import Pipeline, Field, FourierField
from .optimization import predict_cameras
from .render import rays, depth_guided_steps
from .train import jepa_loss


def supported_mask(target_depth, target_K, target_T, source_depth, K, T):
    mask = torch.zeros_like(target_depth, dtype=torch.bool)
    for i in range(len(source_depth)):
        _, valid = correspondence(
            target_depth,
            source_depth[i],
            target_K,
            K[i],
            target_T,
            T[i],
            tolerance=0.015,
        )
        mask |= valid
    return mask & (target_depth > 0.15) & (target_depth < 3.5)


def integrate(sigma, color, support, directions, steps, opaque=False):
    count = steps.shape[-1]
    sigma = sigma.reshape(-1, count)
    color = color.reshape(-1, count, 3)
    support = support.reshape(-1, count)
    if steps.ndim == 1:
        steps = steps[None].expand(len(directions), -1)
    spacing = torch.cat(
        (steps[:, 1:] - steps[:, :-1], steps[:, -1:] - steps[:, -2:-1]), -1
    )
    delta = spacing * directions.norm(dim=-1, keepdim=True)
    alpha = -torch.expm1(-sigma * delta)
    trans = torch.cumprod(
        torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-8), 1), 1
    )[:, :-1]
    weights = alpha * trans
    if opaque:
        from .render import opaque_weights

        weights = opaque_weights(weights)
    return (
        (weights[..., None] * color).sum(1),
        weights.sum(1),
        (weights * support).sum(1),
    )


@torch.no_grad()
def make_cache(
    field,
    rgb,
    K,
    T,
    structure,
    appearance,
    depth,
    target_K,
    target_T,
    xy,
    samples=64,
    chunk=128,
    sampling="uniform",
    projection="perspective",
    local_sources=None,
):
    if local_sources is not None and (
        projection != "orthographic" or local_sources < 2
    ):
        raise ValueError("Invalid local source cache")
    restore = None
    if local_sources is not None:
        # Local camera selection assumes a narrow lateral column, not a raster row.
        order = (xy[:, 0] * (xy[:, 1].max() + 1) + xy[:, 1]).argsort(stable=True)
        restore = order.argsort()
        xy = xy[order]
    steps = torch.linspace(0.15, 3.5, samples, device=rgb.device)
    positions = []
    all_steps = []
    directions = []
    ray_dirs = []
    parts = [[] for _ in range(5)]
    for start in range(0, len(xy), chunk):
        ids = torch.arange(len(rgb), device=rgb.device)
        if local_sources is not None:
            from .render import select_local_sources

            ids = select_local_sources(
                xy[start : start + chunk], target_K, target_T, T, local_sources
            )
        o, d = rays(
            target_K, target_T, *rgb.shape[-2:], xy[start : start + chunk], projection
        )
        local_steps = (
            depth_guided_steps(o, d, K[ids], T[ids], depth[ids], samples)
            if sampling == "depth_guided"
            else steps[None].expand(len(o), -1)
        )
        if sampling == "hybrid":
            from .render import hybrid_steps

            local_steps = hybrid_steps(o, d, K[ids], T[ids], depth[ids], samples)
        all_steps.append(local_steps)
        p = o[:, None] + d[:, None] * local_steps[..., None]
        view = F.normalize(d, dim=-1)[:, None].expand_as(p)
        features = field.query(
            p.reshape(-1, 3),
            rgb[ids],
            K[ids],
            T[ids],
            structure[ids],
            appearance[ids],
            depth[ids],
        )
        positions.append(p)
        directions.append(view)
        ray_dirs.append(d)
        for group, item in zip(parts, features):
            group.append(item.reshape(len(p), samples, -1))
    result = {
        "points": torch.cat(positions),
        "directions": torch.cat(directions),
        "ray_dirs": torch.cat(ray_dirs),
        "features": tuple(torch.cat(p) for p in parts),
        "steps": torch.cat(all_steps),
    }
    if restore is not None:
        for name in ("points", "directions", "ray_dirs", "steps"):
            result[name] = result[name][restore]
        result["features"] = tuple(item[restore] for item in result["features"])
    return result


def cached_render(field, cache, indices=None):
    if indices is None:
        indices = slice(None)
    p = cache["points"][indices]
    d = cache["directions"][indices]
    features = tuple(
        item[indices].reshape(-1, item.shape[-1]) for item in cache["features"]
    )
    features = (*features[:4], features[4][:, 0])
    sigma, color, support = field.evaluate(p.reshape(-1, 3), d.reshape(-1, 3), features)
    return integrate(
        sigma,
        color,
        support,
        cache["ray_dirs"][indices],
        cache["steps"][indices],
        getattr(field, "opaque_surface", False),
    )


@torch.no_grad()
def cache_prediction(field, cache, chunk=256):
    colors = []
    alpha = []
    support = []
    for start in range(0, len(cache["points"]), chunk):
        c, a, s = cached_render(field, cache, slice(start, start + chunk))
        colors.append(c)
        alpha.append(a)
        support.append(s)
    return torch.cat(colors), torch.cat(alpha), torch.cat(support)


def psnr(prediction, target, mask=None):
    error = (prediction - target).square()
    if mask is not None:
        error = error[mask]
    return float(-10 * torch.log10(error.mean().clamp_min(1e-12)))


def save_comparison(path, prediction, truth, mask):
    prediction = prediction.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    rgb = (prediction.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255))).save(
        path.with_name(path.stem + "-rgba.png")
    )
    error = np.abs(prediction - truth).mean(-1)
    heat = np.stack(
        (np.minimum(error * 5, 1), np.zeros_like(error), np.zeros_like(error)), -1
    )
    heat = heat * mask[..., None]
    strip = np.concatenate((prediction, truth, heat), 1)
    Image.fromarray((strip.clip(0, 1) * 255).astype(np.uint8)).save(path)


def matching_examples(rgb, K, T, depth, patch):
    _, _, h, w = rgb.shape
    centers = (
        pixels(h // patch, w // patch, rgb.device)[..., :2] * patch + (patch - 1) / 2
    )
    examples = []
    pairs = [pair for i in range(len(rgb) - 1) for pair in [(i, i + 1), (i + 1, i)]]
    for a, b in pairs:
        xy, valid = correspondence(
            depth[a], depth[b], K[a], K[b], T[a], T[b], tolerance=0.015
        )
        xy = sample(xy.permute(2, 0, 1), centers).reshape(-1, 2)
        good = sample(valid.float()[None], centers).reshape(-1) > 0.99
        target = (xy + 0.5) / patch - 0.5
        nearest = target.round().long()
        good &= (
            (nearest[:, 0] >= 0)
            & (nearest[:, 0] < w // patch)
            & (nearest[:, 1] >= 0)
            & (nearest[:, 1] < h // patch)
        )
        ids = torch.where(good)[0]
        if not len(ids):
            continue
        labels = nearest[ids, 1] * (w // patch) + nearest[ids, 0]
        examples.append((a, b, ids, labels, target[ids]))
    if not examples:
        raise ValueError("No overlapping visible patch correspondences")
    return examples


def matching_loss(model, rgb, examples):
    tokens, _, _ = model.jepa.encoder(rgb)
    tokens = F.normalize(tokens, dim=-1)
    terms = []
    for a, b, ids, labels, _ in examples:
        terms.append(F.cross_entropy(tokens[a, ids] @ tokens[b].T / 0.07, labels))
    return torch.stack(terms).mean()


@torch.no_grad()
def matching_metrics(model, rgb, examples):
    tokens, h, w = model.jepa.encoder(rgb)
    tokens = F.normalize(tokens, dim=-1)
    correct = []
    distances = []
    for a, b, ids, labels, xy in examples:
        nearest = (tokens[a, ids] @ tokens[b].T).argmax(-1)
        recovered = torch.stack((nearest % w, nearest // w), -1).float()
        correct.append((nearest == labels).float())
        distances.append((recovered - xy).norm(dim=-1) * model.config["patch"])
    d = torch.cat(distances)
    return {
        "descriptor_top1": float(torch.cat(correct).mean()),
        "correspondence_median_px": float(d.median()),
        "correspondence_within_patch": float(
            (d <= model.config["patch"]).float().mean()
        ),
    }


@torch.no_grad()
def geometry_metrics(model, tokens, rgb, T, truth_depth):
    _, h, w = model.jepa.encoder(rgb)
    prediction = model.pose.depths(tokens, h, w, rgb.shape[-2:])
    cameras, _ = predict_cameras(model, tokens)
    valid = truth_depth > 0
    return (
        {
            "rotation_degrees": float(
                rotation_error(cameras[:, :3, :3], T[:, :3, :3]).mean() * 180 / math.pi
            ),
            "translation_canonical_l1": float(
                (cameras[:, :3, 3] - T[:, :3, 3]).abs().mean()
            ),
            "depth_abs_rel": float(
                ((prediction - truth_depth).abs() / truth_depth.clamp_min(0.05))[
                    valid
                ].mean()
            ),
        },
        cameras,
        prediction,
    )


def fit_field(field, caches, targets, steps, log, stage, batch=384, persist=None):
    if persist:
        persist = Path(persist)
        persist.mkdir(parents=True, exist_ok=True)
    field.requires_grad_(True)
    optimizer = torch.optim.Adam(field.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, steps, eta_min=0.00005
    )
    initial = []
    with torch.no_grad():
        for cache, target in zip(caches, targets):
            initial.append(psnr(cache_prediction(field, cache)[0], target))
    best_loss = float("inf")
    best = None
    for step in range(steps):
        index = step % len(caches)
        cache = caches[index]
        truth = targets[index]
        ids = torch.randint(len(truth), (min(batch, len(truth)),), device=truth.device)
        color, alpha, _ = cached_render(field, cache, ids)
        loss = F.mse_loss(color, truth[ids]) + 0.001 * (1 - alpha).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {stage}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 5)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 100 == 0 or step == 0 or step == steps - 1:
            with torch.no_grad():
                # Fixed probe drawn once, including every training view; do not tune on held-out RGB.
                values = []
                for c, t in zip(caches, targets):
                    ids = torch.arange(
                        0, len(t), max(1, len(t) // 512), device=t.device
                    )
                    values.append(F.mse_loss(cached_render(field, c, ids)[0], t[ids]))
                probe = float(torch.stack(values).mean())
            row = {
                "stage": stage,
                "step": step + 1,
                "loss": float(loss.detach()),
                "fixed_probe_psnr": -10 * math.log10(max(probe, 1e-12)),
            }
            log.append(row)
            print(json.dumps(row), flush=True)
            if probe < best_loss:
                best_loss = probe
                best = copy.deepcopy(field.state_dict())
            if persist:
                torch.save(
                    {
                        "field": best,
                        "last_field": field.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step + 1,
                        "history": log,
                        "torch_rng": torch.get_rng_state(),
                    },
                    persist / "field.tmp",
                )
                (persist / "field.tmp").replace(persist / "field.pt")
                (persist / "progress.json").write_text(json.dumps(row, indent=2))
    field.load_state_dict(best)
    field.requires_grad_(False)
    final = []
    with torch.no_grad():
        for cache, target in zip(caches, targets):
            final.append(psnr(cache_prediction(field, cache)[0], target))
    return {"initial_train_psnr": initial, "final_train_psnr": final, "steps": steps}


def overfit(
    scene,
    output,
    device="cpu",
    encoder_steps=400,
    geometry_steps=1200,
    vae_steps=1000,
    render_steps=1500,
    samples=64,
    warm_start=None,
    sampling="uniform",
    skip_legacy=False,
):
    if (
        render_steps < 1
        or samples < 4
        or (not warm_start and min(encoder_steps, geometry_steps, vae_steps) < 1)
    ):
        raise ValueError("Positive training budgets and >=4 samples required")
    torch.manual_seed(41)
    np.random.seed(41)
    torch.set_num_threads(4)
    path = Path(scene)
    out = Path(output)
    if (out / "report.json").exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log = []
    rgb, K, T, truth_depth = load_supervised(path, device)
    model = Pipeline(64, 4, 1, depth_refinement=True, field_arch="fourier").to(device)
    if warm_start:
        state = torch.load(warm_start, map_location=device, weights_only=False)
        if {
            k: v for k, v in state["config"].items() if k != "ray_sampling"
        } != model.config or Path(state["scene"]).resolve() != path.resolve():
            raise ValueError("Warm-start scene/model mismatch")
        if state.get("heldout_used_for_training") is not False:
            raise ValueError("Warm-start must exclude held-out training")
        model.load_state_dict(state["model"])
        encoder_steps = geometry_steps = vae_steps = 0
    if sampling != "uniform":
        model.config["ray_sampling"] = sampling
    examples = matching_examples(rgb, K, T, truth_depth, 4)
    before_matching = matching_metrics(model, rgb, examples)
    optimizer = torch.optim.AdamW(
        [p for p in model.jepa.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.0001,
    )
    for step in range(encoder_steps):
        prediction, metrics = jepa_loss(model, rgb, K, T, truth_depth)
        correspondence_loss = matching_loss(model, rgb, examples)
        loss = prediction + 0.2 * correspondence_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.jepa.parameters(), 1)
        optimizer.step()
        model.jepa.ema()
        if (step + 1) % 100 == 0 or step == 0:
            row = {
                "stage": "jepa_auxiliary",
                "step": step + 1,
                "loss": float(loss.detach()),
                "jepa_loss": float(prediction.detach()),
                **matching_metrics(model, rgb, examples),
            }
            log.append(row)
            print(json.dumps(row), flush=True)
    model.jepa.requires_grad_(False)
    with torch.no_grad():
        tokens, gh, gw = model.jepa.encoder(rgb)
    before_geometry, _, _ = geometry_metrics(model, tokens, rgb, T, truth_depth)
    optimizer = torch.optim.Adam(model.pose.parameters(), lr=0.001)
    for step in range(geometry_steps):
        depth = model.pose.depths(tokens, gh, gw, rgb.shape[-2:])
        valid = truth_depth > 0
        loss = F.l1_loss(depth[valid].log(), truth_depth[valid].log())
        for i in range(len(rgb)):
            for j in range(i + 1, min(len(rgb), i + 3)):
                R, d, length, c = model.pose.pair(tokens[i : i + 1], tokens[j : j + 1])
                gt = T[j] @ torch.linalg.inv(T[i])
                loss = (
                    loss
                    + 10 * F.mse_loss(R, gt[None, :3, :3])
                    + 5 * F.mse_loss(d * length[:, None], gt[None, :3, 3])
                    + 0.01 * F.binary_cross_entropy(c, torch.ones_like(c))
                )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.pose.parameters(), 5)
        optimizer.step()
        if (step + 1) % 200 == 0 or step == 0:
            row = {
                "stage": "pose_depth",
                "step": step + 1,
                "loss": float(loss.detach()),
                **geometry_metrics(model, tokens, rgb, T, truth_depth)[0],
            }
            log.append(row)
            print(json.dumps(row), flush=True)
    model.pose.requires_grad_(False)
    optimizer = torch.optim.Adam(model.vae.parameters(), lr=0.002)
    with torch.no_grad():
        vae_initial = psnr(model.vae(rgb, False)[0], rgb)
    for step in range(vae_steps):
        reconstruction, mu, lv = model.vae(rgb, stochastic=True)
        loss = (
            F.mse_loss(reconstruction, rgb)
            + 1e-6 * (-0.5 * (1 + lv - mu.square() - lv.exp())).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0 or step == 0:
            with torch.no_grad():
                score = psnr(model.vae(rgb, False)[0], rgb)
            row = {
                "stage": "vae",
                "step": step + 1,
                "loss": float(loss.detach()),
                "reconstruction_psnr": score,
            }
            log.append(row)
            print(json.dumps(row), flush=True)
    model.vae.requires_grad_(False)
    with torch.no_grad():
        _, structure, appearance, predicted_depth = model.features(rgb)
        after_geometry, predicted_T, predicted_depth = geometry_metrics(
            model, tokens, rgb, T, truth_depth
        )
        masks = []
        coordinates = []
        targets = []
        for target in range(len(rgb)):
            ids = [i for i in range(len(rgb)) if i != target]
            mask = supported_mask(
                truth_depth[target],
                K[target],
                T[target],
                truth_depth[ids],
                K[ids],
                T[ids],
            )
            masks.append(mask)
            xy = pixels(*mask.shape, device)[..., :2][mask]
            coordinates.append(xy)
            targets.append(rgb[target, :, mask].T)
        caches = {}
        for name, cameras, depths in [
            ("oracle", T, truth_depth),
            ("predicted", predicted_T, predicted_depth),
        ]:
            caches[name] = []
            for target in range(len(rgb)):
                ids = [i for i in range(len(rgb)) if i != target]
                caches[name].append(
                    make_cache(
                        model.field,
                        rgb[ids],
                        K[ids],
                        cameras[ids],
                        structure[ids],
                        appearance[ids],
                        depths[ids],
                        K[target],
                        cameras[target],
                        coordinates[target],
                        samples,
                        sampling=sampling,
                    )
                )
    fields = {
        "legacy_oracle": Field(64).to(device),
        "fourier_oracle": copy.deepcopy(model.field),
        "fourier_predicted": copy.deepcopy(model.field),
    }
    if skip_legacy:
        fields.pop("legacy_oracle")
    fits = {}
    torch.save(
        {
            "version": 1,
            "config": model.config,
            "model": model.state_dict(),
            "scene": str(path),
            "heldout_used_for_training": False,
            "completed": {
                "jepa": encoder_steps or state["completed"]["jepa"],
                "pose": geometry_steps or state["completed"]["pose"],
                "vae": vae_steps or state["completed"]["vae"],
                "render_gt": 0,
                "render_pred": 0,
                "joint": 0,
            },
        },
        out / "encoders.pt",
    )
    for name, field in fields.items():
        source = "predicted" if name == "fourier_predicted" else "oracle"
        fits[name] = fit_field(
            field, caches[source], targets, render_steps, log, name, persist=out / name
        )
        dest = out / name
        dest.mkdir(exist_ok=True)
        for target in range(len(rgb)):
            color, alpha, support = cache_prediction(field, caches[source][target])
            frame = torch.zeros_like(rgb[target].permute(1, 2, 0))
            frame[masks[target]] = color
            save_comparison(
                dest / f"train-{target}.png",
                frame,
                rgb[target].permute(1, 2, 0),
                masks[target],
            )
            fits[name].setdefault("train_supported_fraction", []).append(
                float(((alpha > 0.4) & (support > 0.3)).float().mean())
            )
    # Checkpoint contains only scene input-derived model parameters and training metadata.
    model.field.load_state_dict(fields["fourier_predicted"].state_dict())
    torch.save(
        {
            "version": 1,
            "config": model.config,
            "model": model.state_dict(),
            "completed": {
                "jepa": encoder_steps or state["completed"]["jepa"],
                "pose": geometry_steps or state["completed"]["pose"],
                "vae": vae_steps or state["completed"]["vae"],
                "render_gt": render_steps,
                "render_pred": render_steps,
                "joint": 0,
            },
            "protocol": "single-scene-overfit-v1",
            "auxiliary_correspondence_weight": 0.2,
            "scene": str(path),
            "sampling": sampling,
            "warm_start": warm_start,
            "heldout_used_for_training": False,
        },
        out / "model.pt",
    )
    # The first read of held-out image/geometry occurs here, after all fitting and checkpoint selection.
    held = np.load(path / "heldout" / "geometry.npz")
    original = np.load(path / "labels" / "geometry.npz")["T"]
    anchor = len(rgb) // 2
    raw = np.load(path / "labels" / f"{anchor:04}.npy")
    scale = np.median(raw[raw > 0])
    held_T = torch.tensor(held["T"] @ np.linalg.inv(original[anchor]), device=device)
    held_T[:3, 3] /= scale
    held_K = torch.tensor(held["K"], device=device)
    held_depth = torch.tensor(held["depth"] / scale, device=device)
    held_rgb = (
        torch.tensor(
            np.asarray(Image.open(path / "heldout" / "rgb.png")).copy(), device=device
        ).float()
        / 255
    )
    held_mask = supported_mask(held_depth, held_K, held_T, truth_depth, K, T)
    held_xy = pixels(*held_depth.shape, device)[..., :2][held_mask]
    for name, field in fields.items():
        source = "predicted" if name == "fourier_predicted" else "oracle"
        cameras = predicted_T if source == "predicted" else T
        depths = predicted_depth if source == "predicted" else truth_depth
        cache = make_cache(
            field,
            rgb,
            K,
            cameras,
            structure,
            appearance,
            depths,
            held_K,
            held_T,
            held_xy,
            samples,
            sampling=sampling,
        )
        color, alpha, support = cache_prediction(field, cache)
        frame = torch.zeros_like(held_rgb)
        frame[held_mask] = color
        fits[name]["heldout_psnr_fixed_mask"] = psnr(color, held_rgb[held_mask])
        fits[name]["heldout_mask_fraction"] = float(held_mask.float().mean())
        fits[name]["heldout_supported_fraction"] = float(
            ((alpha > 0.4) & (support > 0.3)).float().mean()
        )
        save_comparison(out / name / "heldout.png", frame, held_rgb, held_mask)
    with torch.no_grad():
        vae_final = psnr(model.vae(rgb, False)[0], rgb)
    criterion = {
        "train_psnr_db": 30,
        "heldout_psnr_db": 25,
        "supported_fraction": 0.95,
        "correspondence_within_patch": 0.9,
        "depth_abs_rel": 0.03,
        "rotation_degrees": 0.2,
    }
    result = {
        "protocol": "single-scene-overfit-v1",
        "scene": str(path),
        "input_sha256": hashlib.sha256(
            rgb.cpu().numpy().tobytes() + K.cpu().numpy().tobytes()
        ).hexdigest(),
        "shape": list(rgb.shape),
        "config": model.config,
        "device": device,
        "sampling": sampling,
        "warm_start": warm_start,
        "seed": 41,
        "heldout_used_for_training": False,
        "mask": "fixed ground-truth visibility, same masks for all compared fields; target view excluded from conditioning",
        "auxiliary_correspondence_weight": 0.2,
        "before_matching": before_matching,
        "after_matching": matching_metrics(model, rgb, examples),
        "before_geometry": before_geometry,
        "after_geometry": after_geometry,
        "vae": {"initial_psnr": vae_initial, "final_psnr": vae_final},
        "fields": fits,
        "criteria": criterion,
        "elapsed_seconds": time.time() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    result["component_passes"] = {
        "correspondence": result["after_matching"]["correspondence_within_patch"]
        >= criterion["correspondence_within_patch"],
        "pose": after_geometry["rotation_degrees"] <= criterion["rotation_degrees"],
        "depth": after_geometry["depth_abs_rel"] <= criterion["depth_abs_rel"],
    }
    result["passes"] = {
        name: {
            "train_fit": min(row["final_train_psnr"]) >= criterion["train_psnr_db"],
            "heldout": row["heldout_psnr_fixed_mask"] >= criterion["heldout_psnr_db"]
            and row["heldout_supported_fraction"] >= criterion["supported_fraction"],
        }
        for name, row in fits.items()
    }
    (out / "history.json").write_text(json.dumps(log, indent=2))
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--encoder-steps", type=int, default=400)
    p.add_argument("--geometry-steps", type=int, default=1200)
    p.add_argument("--vae-steps", type=int, default=1000)
    p.add_argument("--render-steps", type=int, default=1500)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--warm-start")
    p.add_argument("--sampling", choices=["uniform", "depth_guided"], default="uniform")
    p.add_argument("--skip-legacy", action="store_true")
    overfit(**vars(p.parse_args()))


if __name__ == "__main__":
    main()


@torch.no_grad()
def cache_surface_depth(field, cache, chunk=256):
    """Depth of the opaque selected surface, not GT-guided sampling."""
    result = []
    for start in range(0, len(cache["points"]), chunk):
        ids = slice(start, start + chunk)
        p = cache["points"][ids]
        d = cache["directions"][ids]
        features = tuple(
            item[ids].reshape(-1, item.shape[-1]) for item in cache["features"]
        )
        features = (*features[:4], features[4][:, 0])
        if hasattr(field, "geometry_sigma"):
            sigma = field.geometry_sigma(p.reshape(-1, 3), features)
        else:
            sigma, _, _ = field.evaluate(p.reshape(-1, 3), d.reshape(-1, 3), features)
        t = cache["steps"][ids]
        sigma = sigma.reshape_as(t)
        spacing = torch.cat(
            (t[:, 1:] - t[:, :-1], t[:, -1:] - t[:, -2:-1]), -1
        ) * cache["ray_dirs"][ids].norm(dim=-1, keepdim=True)
        alpha = -torch.expm1(-sigma * spacing)
        trans = torch.cumprod(
            torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-8), 1), 1
        )[:, :-1]
        weights = alpha * trans
        z = t.gather(1, weights.argmax(-1)[:, None])[:, 0]
        result.append(torch.where(weights.sum(-1) > 1e-12, z, 0))
    return torch.cat(result)
