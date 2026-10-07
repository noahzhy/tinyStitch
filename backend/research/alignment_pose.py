"""Approximate camera lifting from source-only shelf alignment and predicted depth."""

import hashlib, json
from pathlib import Path
import numpy as np
import torch
from .data import load_input


def export(scene, report, output):
    rgb, K = load_input(scene)
    source = json.loads(Path(report).read_text())
    payload = dict(
        version=1,
        input_sha256=hashlib.sha256(
            rgb.numpy().tobytes() + K.numpy().tobytes()
        ).hexdigest(),
        rotations=source["estimated_rotations"],
        front_transforms=source["front_transforms"],
        gt_used_for_fitting=False,
    )
    Path(output).write_text(json.dumps(payload, indent=2))


def lift(path, rgb, K, depth):
    data = json.loads(Path(path).read_text())
    digest = hashlib.sha256(
        rgb.cpu().numpy().tobytes() + K.cpu().numpy().tobytes()
    ).hexdigest()
    if data.get("version") != 1 or data.get("input_sha256") != digest:
        raise ValueError("Alignment does not match input order and intrinsics")
    R = torch.tensor(data["rotations"], device=rgb.device, dtype=K.dtype)
    H = torch.tensor(data["front_transforms"], device=rgb.device, dtype=K.dtype)
    n = len(rgb)
    if (
        R.shape != (n, 3, 3)
        or H.shape != (n, 3, 3)
        or not torch.isfinite(R).all()
        or not torch.isfinite(H).all()
    ):
        raise ValueError("Invalid source alignment")
    if (
        not torch.allclose(
            R @ R.transpose(-1, -2),
            torch.eye(3, device=rgb.device)[None].expand(n, 3, 3),
            atol=1e-4,
        )
        or not (torch.linalg.det(R) > 0.99).all()
    ):
        raise ValueError("Invalid rotations")
    reference = K[n // 2]
    rect = reference[None] @ R @ torch.linalg.inv(K)
    shifts = H @ torch.linalg.inv(rect)
    shifts = shifts / shifts[:, 2:3, 2:3]
    # Foreground scale comes from the predicted source depth, never simulator labels.
    saturation = rgb.max(1).values - rgb.min(1).values
    good = (saturation > 0.2) & (depth > 0) & torch.isfinite(depth)
    scale = (
        depth[good].median()
        if good.any()
        else depth[(depth > 0) & torch.isfinite(depth)].median()
    )
    predicted_scale = float(scale)
    reference_scale = data.get("reference_depth_scale")
    if reference_scale is not None:
        scale = scale.new_tensor(reference_scale)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid predicted depth scale")
    centers = torch.zeros(n, 3, device=rgb.device)
    centers[:, :2] = shifts[:, :2, 2] / reference[[0, 1], [0, 1]] * scale
    T = torch.eye(4, device=rgb.device)[None].repeat(n, 1, 1)
    T[:, :3, :3] = R.transpose(-1, -2)
    T[:, :3, 3] = -(T[:, :3, :3] @ centers[:, :, None])[:, :, 0]
    T = T @ torch.linalg.inv(T[n // 2])
    return T, dict(
        protocol="constant-distance foreground-depth camera lift",
        depth_scale=float(scale),
        predicted_current_depth_scale=predicted_scale,
        frozen_reference_scale=reference_scale is not None,
        alignment_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        uses_gt_geometry=False,
        assumption="approximately parallel translation at fixed shelf distance; foreground saturation prior",
    )
