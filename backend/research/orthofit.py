"""Explicit orthographic-GT capacity experiment; not held-out NVS evaluation."""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import orthographic_gt, load_input
from .inference import load_model
from .optimization import predict_cameras
from .geometry import pixels, project, sample
from .overfit import make_cache, cache_prediction, cached_render, psnr, save_comparison


def run(
    scene,
    checkpoint,
    output,
    steps=2000,
    width=256,
    device="cpu",
    source_geometry="predicted",
    adapt_steps=0,
    surface_weight=0.0,
    occlusion_aware=False,
    edge_fraction=0.0,
    freeze_density=False,
    deepen=False,
    local_sources=None,
):
    if (
        steps < 1
        or width < 16
        or not 0 <= edge_fraction <= 1
        or surface_weight < 0
        or adapt_steps < 0
    ):
        raise ValueError("Invalid budget")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.time()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # GT is intentionally training supervision in this experiment.
    manifest = Path(scene).parent / "manifest.json"
    material = (
        json.loads(manifest.read_text()).get("material_style", "legacy_checker")
        if manifest.exists()
        else "legacy_checker"
    )
    scene_meta = json.loads((Path(scene) / "scene.json").read_text())
    image, target_K, raw_T, raw_depth = orthographic_gt(
        int(Path(scene).name),
        out / "gt",
        width,
        material_style=material,
        shelf_width=scene_meta.get("shelf_width"),
        layout=scene_meta.get("layout"),
    )
    model, state = load_model(checkpoint, device)
    if deepen and model.config.get("field_arch") == "fourier":
        from .model import DeepFourierField

        field = DeepFourierField(model.config["dim"]).to(device)
        missing, unexpected = field.load_state_dict(
            model.field.state_dict(), strict=False
        )
        if unexpected or any(not k.startswith("color_refinement.") for k in missing):
            raise ValueError("Incompatible field warm start")
        model.field = field
        model.config["field_arch"] = "deep_fourier"
    elif deepen and model.config.get("field_arch") != "deep_fourier":
        raise ValueError("Deepening requires Fourier checkpoint")
    model.config["ray_sampling"] = "hybrid"
    model.config["opaque_surface"] = True
    model.field.opaque_surface = True
    if occlusion_aware:
        model.config["occlusion_aware"] = True
        model.field.occlusion_aware = True
    state["config"] = model.config
    rgb, K = load_input(scene, device)
    adaptation = []
    if adapt_steps:
        from .data import load_supervised
        from .overfit import geometry_metrics
        import torch.nn.functional as F

        _, _, truth_T, truth_depth = load_supervised(scene, device)
        with torch.no_grad():
            tokens, gh, gw = model.jepa.encoder(rgb)
        optimizer = torch.optim.Adam(model.pose.parameters(), lr=0.0005)
        valid = truth_depth > 0
        for step in range(adapt_steps):
            depth = model.pose.depths(tokens, gh, gw, rgb.shape[-2:])
            loss = F.l1_loss(depth[valid].log(), truth_depth[valid].log())
            for i in range(len(rgb)):
                for j in range(i + 1, len(rgb)):
                    R, d, length, c = model.pose.pair(
                        tokens[i : i + 1], tokens[j : j + 1]
                    )
                    gt = truth_T[j] @ torch.linalg.inv(truth_T[i])
                    loss = (
                        loss
                        + 10 * F.mse_loss(R, gt[None, :3, :3])
                        + 5 * F.mse_loss(d * length[:, None], gt[None, :3, 3])
                    )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.pose.parameters(), 5)
            optimizer.step()
        adaptation.append(
            {
                "stage": "source_geometry_adaptation",
                "steps": adapt_steps,
                **geometry_metrics(model, tokens, rgb, truth_T, truth_depth)[0],
            }
        )
        optimizer = torch.optim.Adam(model.vae.parameters(), lr=0.0005)
        for step in range(adapt_steps):
            reconstruction, mu, lv = model.vae(rgb)
            loss = (
                F.mse_loss(reconstruction, rgb)
                + 1e-6 * (-0.5 * (1 + lv - mu.square() - lv.exp())).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        adaptation.append(
            {
                "stage": "appearance_adaptation",
                "steps": adapt_steps,
                "loss": float(loss.detach()),
            }
        )
        print(json.dumps(adaptation), flush=True)
    with torch.no_grad():
        tokens, structure, appearance, depth = model.features_batched(rgb, 1)
        T, _ = predict_cameras(model, tokens)
    if source_geometry == "oracle":
        from .data import load_supervised

        _, _, T, depth = load_supervised(scene, device)
    elif source_geometry != "predicted":
        raise ValueError("Unknown source geometry")
    anchor = len(rgb) // 2
    original = np.load(Path(scene) / "labels" / "geometry.npz")["T"][anchor]
    raw_source_depth = np.load(Path(scene) / "labels" / f"{anchor:04}.npy")
    scale = float(np.median(raw_source_depth[raw_source_depth > 0]))
    target_T = torch.tensor(raw_T @ np.linalg.inv(original), device=device)
    target_T[:3, 3] /= scale
    target_K = torch.tensor(
        target_K * np.array([[scale, 1, 1], [1, scale, 1], [1, 1, 1]]),
        dtype=torch.float32,
        device=device,
    )
    gt_depth = torch.tensor(raw_depth / scale, device=device)
    h, w = raw_depth.shape
    xy = pixels(h, w, device)[..., :2]
    # Diagnostic source-support mask only; supervision uses all GT valid pixels.
    from .render import rays

    origins, directions = rays(target_K, target_T, h, w, projection="orthographic")
    points = origins + directions * gt_depth.reshape(-1, 1)
    visible = torch.zeros(h * w, device=device, dtype=torch.bool)
    for i in range(len(rgb)):
        uv, z = project(points, K[i], T[i])
        observed = sample(depth[i, None], uv)[:, 0]
        visible |= (
            (uv[:, 0] >= 0)
            & (uv[:, 0] <= rgb.shape[-1] - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= rgb.shape[-2] - 1)
            & (z > 0)
            & ((z - observed).abs() < 0.025)
        )
    source_supported = (gt_depth > 0) & visible.reshape(h, w)
    mask = gt_depth > 0
    cache = make_cache(
        model.field,
        rgb,
        K,
        T,
        structure,
        appearance,
        depth,
        target_K,
        target_T,
        xy[mask],
        samples=64,
        sampling="hybrid",
        projection="orthographic",
        local_sources=local_sources,
    )
    truth = torch.tensor(image, device=device).float() / 255
    targets = truth[mask]
    target_depths = gt_depth[mask]
    from .overfit import cache_surface_depth

    edge = torch.zeros_like(mask)
    edge[:, 1:] |= (gt_depth[:, 1:] - gt_depth[:, :-1]).abs() > 0.02
    edge[:, :-1] |= (gt_depth[:, 1:] - gt_depth[:, :-1]).abs() > 0.02
    edge[1:] |= (gt_depth[1:] - gt_depth[:-1]).abs() > 0.02
    edge[:-1] |= (gt_depth[1:] - gt_depth[:-1]).abs() > 0.02

    def metrics(colors):
        selected = cache_surface_depth(model.field, cache)
        relative = (selected - target_depths).abs() / target_depths.clamp_min(0.05)
        edges = edge[mask]
        return {
            "edge_psnr": psnr(colors, targets, edges) if edges.any() else None,
            "interior_psnr": psnr(colors, targets, ~edges) if (~edges).any() else None,
            "target_depth_abs_rel": float(relative.mean()),
            "target_depth_outlier_5percent": float((relative > 0.05).float().mean()),
            "edge_fraction": float(edges.float().mean()),
        }

    before = cache_prediction(model.field, cache)[0]
    initial_metrics = metrics(before)
    initial = psnr(before, targets)

    def save_frame(colors, name):
        frame = torch.zeros_like(truth)
        frame[mask] = colors
        save_comparison(out / f"{name}-comparison.png", frame, truth, mask)
        rgba = np.dstack(
            (
                (frame.cpu().numpy().clip(0, 1) * 255).astype(np.uint8),
                mask.cpu().numpy().astype(np.uint8) * 255,
            )
        )
        Image.fromarray(rgba).save(out / f"{name}.png")

    save_frame(before, "before")
    field = model.field
    field.requires_grad_(True)
    if freeze_density:
        field.density.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in field.parameters() if p.requires_grad], lr=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, steps, eta_min=0.00002
    )
    edge_ids = torch.where(edge[mask])[0]
    selected = None
    if freeze_density:
        selected_depth = cache_surface_depth(field, cache)
        selected = (cache["steps"] - selected_depth[:, None]).abs().argmin(-1)
        selected_valid = selected_depth > 0
    torch.manual_seed(9917)
    log = []
    for step in range(steps):
        count = round(384 * edge_fraction) if len(edge_ids) else 0
        ids = torch.cat(
            (
                torch.randint(len(targets), (384 - count,), device=device),
                (
                    edge_ids[torch.randint(len(edge_ids), (count,), device=device)]
                    if count
                    else torch.empty(0, dtype=torch.long, device=device)
                ),
            )
        )
        p = cache["points"][ids]
        d = cache["directions"][ids]
        features = tuple(
            item[ids].reshape(-1, item.shape[-1]) for item in cache["features"]
        )
        features = (*features[:4], features[4][:, 0])
        if selected is not None:
            chosen = selected[ids]
            rows = torch.arange(len(ids), device=device)
            selected_features = tuple(
                v.reshape(len(ids), 64, -1)[rows, chosen] for v in features[:4]
            )
            selected_features = (
                *selected_features,
                features[4].reshape(len(ids), 64)[rows, chosen],
            )
            _, predicted, _ = field.evaluate(
                p[rows, chosen], d[rows, chosen], selected_features
            )
            geometry = (
                (cache["steps"][ids, chosen] - target_depths[ids]).square().mean()
            )
            surface_loss = predicted.sum() * 0
            opacity_loss = predicted.sum() * 0
        else:
            sigma, color, query_support = field.evaluate(
                p.reshape(-1, 3), d.reshape(-1, 3), features
            )
            sigma = sigma.reshape(len(ids), -1)
            color = color.reshape(len(ids), 64, 3)
            t = cache["steps"][ids]
            spacing = torch.cat((t[:, 1:] - t[:, :-1], t[:, -1:] - t[:, -2:-1]), -1)
            alpha = -torch.expm1(-sigma * spacing)
            trans = torch.cumprod(
                torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-8), 1), 1
            )[:, :-1]
            from .render import opaque_weights

            raw_weights = alpha * trans
            weights = opaque_weights(raw_weights)
            soft = raw_weights / raw_weights.sum(-1, keepdim=True).clamp_min(1e-12)
            distances = (t - target_depths[ids, None]).abs()
            admissible = (query_support.reshape_as(t) > 1e-6) & (
                distances < 0.03 * target_depths[ids, None]
            )
            nearest = distances.masked_fill(~admissible, float("inf")).argmin(-1)
            observed = admissible.any(-1)
            surface_loss = (
                -soft.gather(1, nearest[:, None])[:, 0]
                .clamp_min(1e-8)
                .log()[observed]
                .mean()
                if observed.any()
                else soft.sum() * 0
            )
            predicted = (weights[..., None] * color).sum(1)
            # Supervise the first visible 3D surface, not just a flattened RGB canvas.
            geometry = (weights * (t - target_depths[ids, None]).square()).sum(1).mean()
            opacity_loss = (1 - weights.sum(1)).square().mean()
        loss = (
            (predicted - targets[ids]).square().mean()
            + surface_weight * surface_loss
            + 0.2 * geometry
            + 0.002 * opacity_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Orthographic fitting diverged")
        optimizer.zero_grad()
        loss.backward()
        if freeze_density and field.residual.grad is not None:
            field.residual.grad[0] = 0
        torch.nn.utils.clip_grad_norm_(field.parameters(), 5)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "depth_loss": float(geometry.detach()),
            }
            log.append(row)
            print(json.dumps(row), flush=True)
            (out / "progress.json").write_text(json.dumps(log, indent=2))
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step + 1,
                    "rng": torch.get_rng_state(),
                    "gt_used_for_training": True,
                },
                out / "training-state.pt",
            )
    with torch.no_grad():
        final, alpha, support = cache_prediction(field, cache)
    save_frame(final, "panorama")
    state["scene"] = str(Path(scene).resolve())
    state["material_style"] = material
    state["model"] = model.state_dict()
    state["orthographic_gt_training"] = {
        "steps": steps,
        "seed": int(Path(scene).name),
        "uses_gt_target_camera": True,
        "uses_gt_target_rgb": True,
        "uses_gt_target_depth": True,
        "source_geometry": source_geometry,
        "adaptation": adaptation,
    }
    torch.save(state, out / "model.pt")
    report = {
        "protocol": "orthographic-GT-supervised-capacity-v1",
        "scene": str(Path(scene).resolve()),
        "parent_checkpoint_sha256": hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
        "gt_rgb_sha256": hashlib.sha256(
            (out / "gt" / "rgb.png").read_bytes()
        ).hexdigest(),
        "material_style": material,
        "surface_weight": surface_weight,
        "edge_sampling_fraction": edge_fraction,
        "freeze_density": freeze_density,
        "field_arch": model.config.get("field_arch", "legacy"),
        "field_parameters": sum(p.numel() for p in field.parameters()),
        "ray_sampling_seed": 9917,
        "occlusion_aware": getattr(field, "occlusion_aware", False),
        "initial_psnr": initial,
        "final_psnr": psnr(final, targets),
        "initial_metrics": initial_metrics,
        "final_metrics": metrics(final),
        "fixed_gt_visible_fraction": float(mask.float().mean()),
        "pretraining_source_depth_support_fraction": float(
            source_supported.float().mean()
        ),
        "supported_fraction": float(((alpha > 0.4) & (support > 0.3)).float().mean()),
        "steps": steps,
        "elapsed_seconds": time.time() - started,
        "gt_used_for_training": True,
        "source_geometry": source_geometry,
        "adaptation": adaptation,
        "material_model": "opaque single surface; soft weights only used as gradient surrogate",
        "target_camera": "synthetic GT, canonicalized",
        "not_generalization_evidence": True,
        "mask_protocol": "all positive synthetic GT depths, independent of predicted support",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--device", default="cpu")
    run(**vars(p.parse_args()))
