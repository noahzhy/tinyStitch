import torch
import torch.nn.functional as F


def pixels(h, w, device=None):
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    return torch.stack((x, y, torch.ones_like(x)), -1).float()


def backproject(depth, K, T):
    p = pixels(*depth.shape[-2:], depth.device)
    q = (p @ torch.linalg.inv(K).T) * depth[..., None]
    return (q - T[:3, 3]) @ T[:3, :3]


def project(points, K, T):
    q = points @ T[:3, :3].T + T[:3, 3]
    p = q @ K.T
    return p[..., :2] / p[..., 2:].clamp_min(1e-6), q[..., 2]


def sample(image, xy):
    # Explicit bilinear gather supports coordinate gradients on MPS (grid_sample backward does not).
    h, w = image.shape[-2:]
    x, y = xy[..., 0], xy[..., 1]
    x0 = x.floor()
    y0 = y.floor()
    dx = x - x0
    dy = y - y0
    result = torch.zeros(
        (*xy.shape[:-1], image.shape[0]), device=image.device, dtype=image.dtype
    )
    for ox, oy, weight in [
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ]:
        ix = (x0 + ox).long()
        iy = (y0 + oy).long()
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        value = image[:, iy.clamp(0, h - 1), ix.clamp(0, w - 1)].movedim(0, -1)
        result = result + value * (weight * valid)[..., None]
    return result


def correspondence(depth_a, depth_b, Ka, Kb, Ta, Tb, tolerance=0.03):
    xy, z = project(backproject(depth_a, Ka, Ta), Kb, Tb)
    h, w = depth_b.shape
    observed = sample(depth_b[None], xy)[..., 0]
    valid = (
        (depth_a > 0)
        & (z > 0)
        & (xy[..., 0] >= 0)
        & (xy[..., 0] <= w - 1)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= h - 1)
        & (observed > 0)
        & ((observed - z).abs() < tolerance * z.clamp_min(1))
    )
    return xy, valid


def rotation6d(v):
    a = F.normalize(v[..., :3], dim=-1)
    b = v[..., 3:]
    b = F.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), -1)


def rotation_error(a, b):
    relative = a.transpose(-1, -2) @ b
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        -1,
    )
    sine = skew.norm(dim=-1) / 2
    return torch.atan2(sine, cosine.clamp(-1, 1))


def scale_intrinsics(K, sx, sy):
    out = K.clone()
    out[0] *= sx
    out[1] *= sy
    return out


def three_to_opencv(camera_world):
    flip = torch.diag(camera_world.new_tensor([1.0, -1.0, -1.0, 1.0]))
    return flip @ torch.linalg.inv(camera_world)


def canonicalize(T, depth):
    anchor = len(T) // 2
    out = T @ torch.linalg.inv(T[anchor])
    scale = depth[anchor][depth[anchor] > 0].median()
    out = out.clone()
    out[:, :3, 3] /= scale
    return out, depth / scale, scale


def sample_visible_rgb(image, xy, depth, z, tolerance=0.03):
    """Bilinear RGB interpolation restricted to neighbors on the queried surface."""
    h, w = depth.shape
    x, y = xy[:, 0], xy[:, 1]
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    color = image.new_zeros((len(xy), 3))
    den = z.new_zeros(len(xy))
    error = z.new_zeros(len(xy))
    for ox, oy, weight in [
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ]:
        ix, iy = (x0 + ox).long(), (y0 + oy).long()
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ix, iy = ix.clamp(0, w - 1), iy.clamp(0, h - 1)
        observed = depth[iy, ix]
        delta = (z - observed) / observed.clamp_min(0.05)
        keep = valid & (observed > 0) & (delta.abs() < tolerance)
        q = weight * keep
        color = color + image[:, iy, ix].T * q[:, None]
        den = den + q
        error = error + delta * q
    return color / den.clamp_min(1e-8)[:, None], error / den.clamp_min(1e-8), den
