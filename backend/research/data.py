"""Deterministic analytic box renderer: linear camera-z depth and calibrated views."""

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .geometry import canonicalize


def camera(center, look):
    z = look - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0, 1, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack((x, y, z))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    return T


def render(
    boxes,
    K,
    T,
    h,
    w,
    exposure=1,
    projection="perspective",
    material_style="legacy_checker",
    crop_boxes=True,
):
    y, x = np.mgrid[:h, :w]
    directions = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(K).T @ T[:3, :3]
    origin = -T[:3, :3].T @ T[:3, 3]
    if projection == "orthographic":
        q = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(K).T
        q[..., 2] = 0
        origin = q @ T[:3, :3] + origin
        directions = np.broadcast_to(T[2, :3], (h, w, 3))
    elif projection != "perspective":
        raise ValueError("Unknown projection")
    depth = np.full((h, w), np.inf)
    rgb = np.zeros((h, w, 3), np.float32)
    for low, high, color in boxes:
        roi = (slice(None), slice(None))
        if crop_boxes:
            corners = np.array(
                [
                    [x, y, z]
                    for x in (low[0], high[0])
                    for y in (low[1], high[1])
                    for z in (low[2], high[2])
                ]
            )
            q = corners @ T[:3, :3].T + T[:3, 3]
            if projection == "orthographic" or (q[:, 2] > 0.001).all():
                uv = (
                    q[:, :2] * np.array([K[0, 0], K[1, 1]])
                    if projection == "orthographic"
                    else q[:, :2] / q[:, 2, None] * np.array([K[0, 0], K[1, 1]])
                )
                uv += K[:2, 2]
                xmin, ymin = np.floor(uv.min(0) - 2).astype(int)
                xmax, ymax = np.ceil(uv.max(0) + 2).astype(int)
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax + 1), min(h, ymax + 1)
                if xmin >= xmax or ymin >= ymax:
                    continue
                roi = (slice(ymin, ymax), slice(xmin, xmax))
        local_d = directions[roi]
        local_o = origin[roi] if origin.ndim == 3 else origin
        local_depth = depth[roi]
        local_rgb = rgb[roi]
        d = np.where(np.abs(local_d) < 1e-8, 1e-8, local_d)
        a = (low - local_o) / d
        b = (high - local_o) / d
        near = np.minimum(a, b).max(-1)
        far = np.maximum(a, b).min(-1)
        hit = (far >= np.maximum(near, 0)) & (near > 0) & (near < local_depth)
        point = local_o + local_d * near[..., None]
        pattern = (np.floor(point[..., 0] * 24) + np.floor(point[..., 1] * 24)).astype(
            int
        ) % 2
        shade = 0.78 + 0.22 * pattern
        if material_style == "solid_labels":
            # Object-local opaque paint: no world-space grid shared with the background.
            local = (point - low) / (high - low)
            axis = np.minimum(a, b).argmax(-1)
            shade = np.choose(axis, [0.68, 1.0, 0.88])
            paint = np.broadcast_to(np.asarray(color), (*near.shape, 3)).copy()
            if np.ptp(color) > 0.3:  # colored product, not shelf/floor
                label = (
                    (local[..., 0] > 0.15)
                    & (local[..., 0] < 0.85)
                    & (local[..., 1] > 0.30)
                    & (local[..., 1] < 0.66)
                    & (axis == 2)
                )
                paint[label] = np.asarray(color) * 0.25 + 0.72
                stripe = label & (local[..., 1] > 0.44) & (local[..., 1] < 0.50)
                paint[stripe] = np.asarray(color) * 0.45
            local_rgb[hit] = paint[hit] * shade[hit, None]
        elif material_style == "legacy_checker":
            local_rgb[hit] = np.asarray(color) * shade[hit, None]
        else:
            raise ValueError("Unknown material style")
        local_depth[hit] = near[hit]
    depth[~np.isfinite(depth)] = 0
    return (np.clip(rgb * exposure, 0, 1) * 255).astype(np.uint8), depth.astype(
        np.float32
    )


def generate(
    root,
    scenes=12,
    seed=730000,
    h=192,
    w=256,
    views=5,
    exposure_jitter=True,
    material_style="solid_labels",
    capture_step=None,
    target_width=None,
    varied_layout=False,
    capture_tilt=False,
    mixed_bays=False,
):
    if varied_layout and scenes != 24:
        raise ValueError(
            "Varied-layout controlled split requires 24 scenes (12 train, 6 val, 6 test)"
        )
    if capture_step is not None and (
        not np.isfinite(capture_step) or capture_step <= 0
    ):
        raise ValueError("Capture step must be positive")
    if target_width is not None and target_width < 16:
        raise ValueError("Invalid target width")
    if scenes < 6 or views < 2 or h < 16 or w < 16 or h % 8 or w % 8:
        raise ValueError("Need >=6 scenes, >=2 views, image dimensions multiples of 8")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("Output already exists; use a new directory")
    records = []
    for n in range(scenes):
        scene_width = None if capture_step is None else 2.6 + (views - 1) * capture_step
        recipes = [(2, 3), (3, 6), (4, 9), (5, 4), (6, 7), (4, 1000)]
        layers, group_size = recipes[n % 6]
        layout = (
            {
                "layers": layers,
                "group_size": group_size,
                "uneven": True,
                "empty_probability": 0.08 if n % 6 != 5 else 0.0,
            }
            if varied_layout
            else None
        )
        if mixed_bays:
            # Every scene contains multiple adjacent bays with different layer counts.
            counts = [2, 5, 3, 6, 4]
            layout = {
                "bays": [
                    {
                        "layers": counts[(i + n) % 5],
                        "group_size": [3, 1000, 4, 6, 1000][(i + n) % 5],
                        "uneven": True,
                        "empty_probability": 0.04,
                    }
                    for i in range(5)
                ]
            }
        boxes, width, rng = shelf_scene(seed + n, scene_width, layout)
        path = root / str(seed + n)
        (path / "rgb").mkdir(parents=True)
        (path / "labels").mkdir()
        (path / "heldout").mkdir()
        focal = w / (2 * np.tan(np.deg2rad(54) / 2))
        K = np.array(
            [[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]], np.float32
        )
        poses = []
        span = 1.3 if capture_step is None else (views - 1) * capture_step
        for i, cx in enumerate(np.linspace(-span / 2, span / 2, views)):
            T = camera(
                np.array([cx, 1 + rng.uniform(-0.02, 0.02), -2.6]),
                np.array([cx * 0.7 if capture_step is None else cx, 1, 0]),
            )
            if capture_tilt:
                phase = 2 * np.pi * i / max(1, views - 1)
                yaw = np.deg2rad(6 * np.sin(phase + 0.3))
                roll = np.deg2rad(1.5 * np.sin(phase * 1.5))
                center = -T[:3, :3].T @ T[:3, 3]
                T = camera(center, np.array([cx + 2.6 * np.tan(yaw), 1, 0]))
                c, s = np.cos(roll), np.sin(roll)
                T[:3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ T[:3]
            poses.append(T)
            exposure = rng.uniform(0.85, 1.15)
            rgb, depth = render(
                boxes,
                K,
                T,
                h,
                w,
                exposure if exposure_jitter else 1.0,
                material_style=material_style,
            )
            Image.fromarray(rgb).save(path / "rgb" / f"{i:04}.png")
            np.save(path / "labels" / f"{i:04}.npy", depth)
            Image.fromarray((depth > 0).astype(np.uint8) * 255).save(
                path / "labels" / f"{i:04}-valid.png"
            )
        np.savez(path / "labels" / "geometry.npz", T=np.stack(poses))
        (path / "intrinsics.json").write_text(
            json.dumps({"K": [K.tolist()] * views, "width": w, "height": h})
        )
        target = camera(np.array([0.08, 1, -2.6]), np.array([0, 1, 0]))
        Kt = K.copy()
        Kt[0, 0] = Kt[1, 1] = w / (2 * np.tan(np.deg2rad(80) / 2))
        rgb, depth = render(boxes, Kt, target, h, w, material_style=material_style)
        Image.fromarray(rgb).save(path / "heldout" / "rgb.png")
        np.savez(path / "heldout" / "geometry.npz", K=Kt, T=target, depth=depth)
        orthographic_gt(
            seed + n,
            path / "orthographic",
            target_width or w,
            material_style=material_style,
            shelf_width=scene_width,
            layout=layout,
        )
        (path / "scene.json").write_text(
            json.dumps(
                {
                    "shelf_width": width,
                    "layout": layout,
                    "capture_tilt": capture_tilt,
                    "capture_span": span,
                    "varied_layout": varied_layout,
                    "capture_step": capture_step,
                    "views": views,
                    "product_columns": (
                        max(12, round(width / 0.25)) if capture_step is not None else 12
                    ),
                }
            )
        )
        split = (
            ("train" if n < 12 else "val" if n < 18 else "test")
            if varied_layout
            else ("test" if n % 6 == 5 else "val" if n % 6 == 4 else "train")
        )
        records.append({"seed": seed + n, "path": str(seed + n), "split": split})
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "capture_step": capture_step,
                "evaluation_counts": (
                    [views] if capture_step is not None else [10, 30, 60]
                ),
                "renderer": "analytic-ray-box-v1",
                "exposure_jitter": exposure_jitter,
                "material_style": material_style,
                "scenes": records,
            },
            indent=2,
        )
    )


def load_input(path, device="cpu"):
    path = Path(path)
    meta = json.loads((path / "intrinsics.json").read_text())
    files = sorted((path / "rgb").glob("*.png")) + sorted((path / "rgb").glob("*.jpg"))
    if len(files) < 2 or len(files) != len(meta["K"]):
        raise ValueError("Need ordered RGB and one calibrated K per view")
    images = [np.asarray(Image.open(p).convert("RGB")).copy() for p in sorted(files)]
    if any(im.shape != images[0].shape for im in images):
        raise ValueError("All images must have equal dimensions")
    if images[0].shape[:2] != (meta["height"], meta["width"]):
        raise ValueError("Intrinsics dimensions do not match RGB")
    rgb = (
        torch.tensor(np.stack(images), device=device).permute(0, 3, 1, 2).float() / 255
    )
    K = torch.tensor(meta["K"], device=device)
    if (
        K.shape != (len(rgb), 3, 3)
        or not torch.isfinite(K).all()
        or (K[:, [0, 1], [0, 1]] <= 0).any()
        or not torch.allclose(
            K[:, 2], K.new_tensor([0.0, 0.0, 1.0]).expand(len(rgb), 3)
        )
    ):
        raise ValueError("Invalid intrinsics")
    return rgb, K


def load_supervised(path, device="cpu"):
    rgb, K = load_input(path, device)
    p = Path(path)
    T = torch.tensor(np.load(p / "labels" / "geometry.npz")["T"], device=device)
    depth = torch.tensor(
        np.stack([np.load(f) for f in sorted((p / "labels").glob("*.npy"))]),
        device=device,
    )
    T, depth, _ = canonicalize(T, depth)
    return rgb, K, T, depth


def shelf_scene(seed, shelf_width=None, layout=None):
    rng = np.random.default_rng(seed)
    width = rng.uniform(2.5, 3.5)
    if shelf_width is not None:
        if not np.isfinite(shelf_width) or shelf_width <= 0:
            raise ValueError("Invalid shelf width")
        width = float(shelf_width)
    columns = 12 if shelf_width is None else max(12, round(width / 0.25))
    if layout is not None and "bays" in layout:
        bays = layout["bays"]
        if len(bays) < 2:
            raise ValueError("Need multiple bays")
        result = []
        bay_width = width / len(bays)
        for i, bay in enumerate(bays):
            center = -width / 2 + (i + 0.5) * bay_width
            for low, high, color in varied_shelf_boxes(
                rng, bay_width, max(3, round(bay_width / 0.25)), bay
            )[:-1]:
                shift = np.array([center, 0, 0])
                result.append((low + shift, high + shift, color))
            x = -width / 2 + i * bay_width
            result.append(
                (
                    np.array([x, 0, -0.3]),
                    np.array([x + 0.035, 2, 0.3]),
                    [0.7, 0.72, 0.75],
                )
            )
        result.append(
            (
                np.array([-width / 2 - 1, -0.12, -6]),
                np.array([width / 2 + 1, -0.08, 6]),
                [0.65, 0.63, 0.6],
            )
        )
        return result, width, rng
    if layout is not None:
        return varied_shelf_boxes(rng, width, columns, layout), width, rng
    boxes = []
    boxes.append(
        (
            np.array([-width / 2, 0, 0.2]),
            np.array([width / 2, 2, 0.3]),
            [0.45, 0.47, 0.5],
        )
    )
    for layer in range(4):
        boxes.append(
            (
                np.array([-width / 2, 0.1 + layer * 0.46, -0.3]),
                np.array([width / 2, 0.15 + layer * 0.46, 0.3]),
                [0.7, 0.72, 0.75],
            )
        )
        for c in range(columns):
            cx = -width / 2 + (c + 0.5) * width / columns
            front = rng.uniform(-0.34, -0.12)
            boxes.append(
                (
                    np.array(
                        [cx - width / (columns * 2.5), 0.15 + layer * 0.46, front]
                    ),
                    np.array(
                        [
                            cx + width / (columns * 2.5),
                            0.4 + layer * 0.46 + rng.uniform(0, 0.12),
                            0.18,
                        ]
                    ),
                    np.array(
                        [[0.8, 0.25, 0.2], [0.2, 0.65, 0.3], [0.25, 0.3, 0.8]][c % 3]
                    ),
                )
            )
    boxes.append(
        (np.array([-7, -0.12, -6]), np.array([7, -0.08, 6]), [0.65, 0.63, 0.6])
    )
    return boxes, width, rng


def varied_shelf_boxes(rng, width, columns, layout):
    layers, group_size = int(layout["layers"]), int(layout["group_size"])
    if not 2 <= layers <= 8 or group_size < 1:
        raise ValueError("Invalid layer count or product group size")
    gaps = rng.uniform(0.8, 1.2, layers) if layout.get("uneven") else np.ones(layers)
    gaps = gaps / gaps.sum() * 1.84
    heights = 0.1 + np.r_[0.0, gaps.cumsum()[:-1]]
    boxes = [
        (
            np.array([-width / 2, 0, 0.2]),
            np.array([width / 2, 2, 0.3]),
            [0.45, 0.47, 0.5],
        )
    ]
    palette = np.array(
        [
            [0.8, 0.25, 0.2],
            [0.2, 0.65, 0.3],
            [0.25, 0.3, 0.8],
            [0.85, 0.65, 0.15],
            [0.65, 0.2, 0.75],
            [0.15, 0.65, 0.75],
        ]
    )
    for layer, (y, gap) in enumerate(zip(heights, gaps)):
        boxes.append(
            (
                np.array([-width / 2, y, -0.3]),
                np.array([width / 2, y + 0.05, 0.3]),
                [0.7, 0.72, 0.75],
            )
        )
        order = rng.permutation(len(palette))
        sizes = rng.uniform(0.7, 0.85, (len(palette), 2))
        for c in range(columns):
            sku = int(order[(c // group_size) % len(order)])
            if rng.random() < layout.get("empty_probability", 0.0):
                continue
            cx = -width / 2 + (c + 0.5) * width / columns
            half = width / columns * sizes[sku, 0] / 2
            # Items in one SKU block share color and dimensions; small placement/depth variation.
            height = (gap - 0.075) * sizes[sku, 1]
            front = rng.uniform(-0.34, -0.12)
            boxes.append(
                (
                    np.array([cx - half, y + 0.05, front]),
                    np.array([cx + half, y + 0.05 + height, 0.18]),
                    palette[sku],
                )
            )
    boxes.append(
        (
            np.array([-max(7, width / 2 + 1), -0.12, -6]),
            np.array([max(7, width / 2 + 1), -0.08, 6]),
            [0.65, 0.63, 0.6],
        )
    )
    return boxes


def orthographic_gt(
    seed,
    output,
    width=256,
    material_style="legacy_checker",
    shelf_width=None,
    layout=None,
):
    """Independent ray-box GT including physical shelves, occlusion and depth."""
    boxes, shelf_width, _ = shelf_scene(seed, shelf_width, layout)
    h = round(width * 2.12 / (shelf_width * 1.06))
    scale = (width - 1) / (shelf_width * 1.06)
    K = np.array(
        [[scale, 0, (width - 1) / 2], [0, scale, (h - 1) / 2], [0, 0, 1]], np.float32
    )
    T = camera(np.array([0.0, 1.0, -2.6]), np.array([0.0, 1.0, 0.0]))
    rgb, depth = render(
        boxes, K, T, h, width, projection="orthographic", material_style=material_style
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out / "rgb.png")
    Image.fromarray((depth > 0).astype(np.uint8) * 255).save(out / "mask.png")
    np.savez(out / "geometry.npz", K=K, T=T, depth=depth)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "material_style": material_style,
                "projection": "orthographic",
                "width": width,
                "height": h,
                "renderer": "analytic-ray-box-v1",
            },
            indent=2,
        )
    )
    return rgb, K, T, depth
