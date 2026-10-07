"""Opaque source-depth reprojection control, independent of learned field density/color.

This diagnostic is not the neural volume renderer. It helps distinguish geometry
errors from field instability. Pixels without projected source support stay invalid.
"""

import torch
from .geometry import backproject


@torch.no_grad()
def render_surface(rgb, K, T, depth, target_K, target_T, height, width, stride=4):
    if stride < 1 or min(height, width) < 1:
        raise ValueError("Invalid surface control resolution")
    zbuffer = depth.new_full((height * width,), float("inf"))
    output = rgb.new_zeros((height * width, 3))
    for image, intrinsics, pose, d in zip(rgb, K, T, depth):
        # These are selected integer source pixels, not resized pixel centers.
        k = intrinsics.clone()
        k[:2] /= stride
        points = backproject(d[::stride, ::stride], k, pose).reshape(-1, 3)
        q = points @ target_T[:3, :3].T + target_T[:3, 3]
        xy = q[:, :2] * target_K[[0, 1], [0, 1]] + target_K[:2, 2]
        xy = xy.round().long()
        z = q[:, 2]
        valid = (
            (d[::stride, ::stride].reshape(-1) > 0)
            & (z > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        if not valid.any():
            continue
        xy, z = xy[valid], z[valid]
        colors = image[:, ::stride, ::stride].permute(1, 2, 0).reshape(-1, 3)[valid]
        ids = xy[:, 1] * width + xy[:, 0]
        local = z.new_full((height * width,), float("inf"))
        local.scatter_reduce_(0, ids, z, reduce="amin", include_self=True)
        # Deterministically choose a single closest opaque source sample.
        candidate = z == local[ids]
        order = torch.arange(len(z), device=z.device)
        chosen = torch.full(
            (height * width,), len(z), device=z.device, dtype=torch.long
        )
        chosen.scatter_reduce_(
            0, ids[candidate], order[candidate], reduce="amin", include_self=True
        )
        replace = (local < zbuffer) & (chosen < len(z))
        output[replace] = colors[chosen[replace]]
        zbuffer[replace] = local[replace]
    return output.reshape(height, width, 3), torch.isfinite(zbuffer).reshape(
        height, width
    )
