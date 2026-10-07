import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .data import load_input
from .model import Pipeline
from .geometry import backproject, project, pixels
from .optimization import predict_cameras, optimize_geometry
from .render import render_rays, render_image


def load_model(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    required = ["jepa", "pose", "vae", "render_gt", "render_pred"]
    if state.get("protocol") != "single-scene-overfit-v1":
        required.append("joint")
    if state.get("version") != 1 or any(
        state.get("completed", {}).get(s, 0) < 1 for s in required
    ):
        raise ValueError("Need a trained complete research checkpoint")
    model = Pipeline(**state["config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def reconstruct(
    path,
    checkpoint,
    output,
    device="cpu",
    optimize=20,
    width=1024,
    samples=32,
    use_vae=True,
    global_opt=True,
    target=None,
    projection="orthographic",
    geometry_width=None,
):
    if projection not in ("perspective", "orthographic"):
        raise ValueError("Unknown target projection")
    if optimize < 0 or width < 16 or samples < 4:
        raise ValueError("Invalid rendering budget")
    torch.manual_seed(17)
    start = time.time()
    torch.set_num_threads(4)
    model, state = load_model(checkpoint, device)
    use_vae = use_vae and not state.get("ablations", {}).get("no_vae", False)
    rgb, K = load_input(path, device)
    model.requires_grad_(False)
    if geometry_width is None and rgb.shape[-1] > 256:
        geometry_width = 256
    with torch.no_grad():
        if geometry_width is None:
            tokens, structure, appearance, depth = model.features_batched(
                rgb, 1 if rgb.shape[-1] > 128 else 4
            )
        else:
            height = round(geometry_width * rgb.shape[-2] / rgb.shape[-1])
            if height % model.config["patch"] or geometry_width % model.config["patch"]:
                raise ValueError("Geometry dimensions must be patch-aligned")
            tokens, structure, appearance, depth = model.features_multires(
                rgb, (height, geometry_width), 1
            )
        T, edges = predict_cameras(
            model, tokens, pair_batch=2 if tokens.shape[1] > 768 else 8
        )
    if not torch.isfinite(depth).all() or not torch.isfinite(T).all():
        raise ValueError("Nonfinite predicted geometry")
    graph_history = []
    scene_history = []
    if global_opt:
        T, depth, graph_history = optimize_geometry(
            rgb, K, T, depth, steps=max(1, optimize), edges=edges
        )
    if optimize:
        model.field.residual.requires_grad_(True)
        optimizer = torch.optim.Adam([model.field.residual], lr=0.01)
        for step in range(optimize):
            i = step % len(rgb)
            ids = [j for j in range(len(rgb)) if j != i]
            h, w = rgb.shape[-2:]
            xy = torch.stack(
                (
                    torch.randint(w, (64,), device=device),
                    torch.randint(h, (64,), device=device),
                ),
                -1,
            ).float()
            color, _, _ = render_rays(
                model,
                rgb[ids],
                K[ids],
                T[ids],
                structure[ids],
                appearance[ids],
                depth[ids],
                K[i],
                T[i],
                xy,
                24,
                use_vae,
            )
            loss = F.l1_loss(color, rgb[i, :, xy[:, 1].long(), xy[:, 0].long()].T)
            if not torch.isfinite(loss):
                raise ValueError("Scene optimization diverged")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scene_history.append(float(loss.detach()))
    anchor = len(rgb) // 2
    target_T = T[anchor].clone()
    if target is None and projection == "orthographic":
        target_K, target_T, h, w = orthographic_target(rgb, K, T, depth, width)
    elif target is None:
        # Estimate angular bounds from predicted 3D coverage, capped at 120 degrees.
        points = torch.cat(
            [
                backproject(depth[i], K[i], T[i])[::8, ::8].reshape(-1, 3)
                for i in range(len(rgb))
            ]
        )
        q = points @ target_T[:3, :3].T + target_T[:3, 3]
        good = q[:, 2] > 0.05
        if not good.any():
            raise ValueError("No geometry in front of target camera")
        ratios = q[good, 0] / q[good, 2]
        span = ratios.abs().quantile(0.95).clamp(0.3, 1.732)
        h = max(64, round(width * rgb.shape[-2] / rgb.shape[-1]))
        f = width / (2 * span)
        target_K = K[anchor].clone()
        target_K[0, 0] = target_K[1, 1] = f
        target_K[0, 2] = (width - 1) / 2
        target_K[1, 2] = (h - 1) / 2
        w = width
    else:
        target_K, target_T, h, w = target
    color, mask = render_image(
        model,
        rgb,
        K,
        T,
        structure,
        appearance,
        depth,
        target_K,
        target_T,
        h,
        w,
        samples=samples,
        use_vae=use_vae,
        projection=projection,
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    array = (color.cpu().numpy() * 255).astype(np.uint8)
    valid = mask.cpu().numpy()
    Image.fromarray(np.dstack((array, valid.astype(np.uint8) * 255))).save(
        out / "panorama.png"
    )
    Image.fromarray(valid.astype(np.uint8) * 255).save(out / "mask.png")
    np.save(out / "depth.npy", depth.cpu().numpy())
    np.save(out / "cameras.npy", T.cpu().numpy())
    report = {
        "version": 1,
        "input_count": len(rgb),
        "native_input_resolution": [rgb.shape[-1], rgb.shape[-2]],
        "geometry_working_width": geometry_width or rgb.shape[-1],
        "source_rgb_retained_at_native_resolution": True,
        "input_sha256": hashlib.sha256(
            rgb.cpu().numpy().tobytes() + K.cpu().numpy().tobytes()
        ).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "coverage": float(mask.float().mean()),
        "elapsed_seconds": time.time() - start,
        "confidence": [float(c) for _, _, _, c in edges],
        "low_confidence": any(float(c) < 0.6 for _, _, _, c in edges),
        "global_optimization": graph_history,
        "scene_optimization": scene_history,
        "projection": projection,
        "front_alignment": (
            "robust predicted surface normals"
            if projection == "orthographic"
            else "middle camera"
        ),
        "target_K": target_K.cpu().tolist(),
        "target_T": target_T.cpu().tolist(),
        "scale": "canonical, not metric",
        "trained_steps": state["completed"],
        "field_arch": model.config.get("field_arch", "legacy"),
        "field_bands": model.config.get("field_bands", 8),
        "appearance_geometry": model.config.get("appearance_geometry", False),
        "feature_center_mapping": model.config.get("feature_center_mapping", False),
        "pose_graph_optimization": bool(model.config.get("pose_graph", False)),
        "global_sequence_features": bool(model.config.get("global_pose", False)),
        "global_pose_adaptation": state.get("global_pose_adaptation"),
        "pose_adaptation": state.get("pose_adaptation"),
        "depth_adaptation": state.get("depth_adaptation"),
        "renderer_adaptation": state.get("renderer_adaptation"),
        "orthographic_gt_training": state.get("orthographic_gt_training"),
    }
    if not valid.any():
        report["status"] = "failed_no_coverage"
    else:
        report["status"] = "ok"
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def orthographic_target(rgb, K, T, depth, width):
    """Estimate a frontal plane from predicted geometry; never read scene labels.

    Orthographic K uses pixels per canonical scene unit, rather than focal length.
    Normals perpendicular to the middle camera are excluded (e.g. the floor).
    """
    clouds, normals = [], []
    anchor = T[len(T) // 2]
    forward = anchor[2, :3]
    for i in range(len(rgb)):
        p = backproject(depth[i], K[i], T[i])
        dx = p[1:-1, 2:] - p[1:-1, :-2]
        dy = p[2:, 1:-1] - p[:-2, 1:-1]
        n = F.normalize(torch.linalg.cross(dx, dy), dim=-1)
        n = n * torch.where((n @ forward) < 0, -1.0, 1.0)[..., None]
        valid = ((n @ forward) > 0.9) & (rgb[i, :, 1:-1, 1:-1].mean(0) > 0.06)
        valid &= torch.isfinite(p[1:-1, 1:-1]).all(-1)
        clouds.append(p[1:-1, 1:-1][valid])
        normals.append(n[valid])
    points, n = torch.cat(clouds), torch.cat(normals)
    if len(points) < 32:
        raise ValueError("Insufficient frontal surface geometry")
    z = F.normalize(n.median(0).values, dim=0)
    x = F.normalize(anchor[0, :3] - z * (anchor[0, :3] @ z), dim=0)
    y = torch.linalg.cross(z, x)
    rotation = torch.stack((x, y, z))
    center = -anchor[:3, :3].T @ anchor[:3, 3]
    target = anchor.clone()
    target[:3, :3] = rotation
    target[:3, 3] = -rotation @ center
    q = points @ rotation.T + target[:3, 3]
    low, high = q[:, :2].quantile(0.005, dim=0), q[:, :2].quantile(0.995, dim=0)
    span = (high - low) * 1.06
    if (span <= 0).any():
        raise ValueError("Degenerate orthographic extent")
    scale = (width - 1) / span[0]
    height = max(16, round(float(span[1] * scale)) + 1)
    mid = (low + high) / 2
    intrinsics = K[0].new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    intrinsics[0, 0] = intrinsics[1, 1] = scale
    intrinsics[0, 2] = (width - 1) / 2 - mid[0] * scale
    intrinsics[1, 2] = (height - 1) / 2 - mid[1] * scale
    return intrinsics, target, height, width
