import torch
import torch.nn.functional as F
from .geometry import rotation6d, project, backproject, sample


def predict_cameras(model, tokens, pair_batch=8, graph=None):
    n = len(tokens)
    edges = []
    if pair_batch < 1:
        raise ValueError("Invalid pair batch size")
    if n < 2:
        raise ValueError("Need at least two connected views")
    graph = model.config.get("pose_graph", False) if graph is None else graph
    hops = (1, 2)
    pairs = [(i, i + hop) for i in range(n) for hop in hops if i + hop < n]
    for start in range(0, len(pairs), pair_batch):
        group = pairs[start : start + pair_batch]
        R, d, length, confidence = model.pose.pair(
            tokens[[i for i, j in group]], tokens[[j for i, j in group]]
        )
        for k, (i, j) in enumerate(group):
            edge = torch.eye(4, device=tokens.device)
            edge[:3, :3] = R[k]
            edge[:3, 3] = d[k] * length[k]
            edges.append((i, j, edge, confidence[k]))
    cameras = [torch.eye(4, device=tokens.device)]
    for i in range(n - 1):
        cameras.append(
            next(e for a, b, e, c in edges if a == i and b == i + 1) @ cameras[-1]
        )
    T = torch.stack(cameras)
    T = T @ torch.linalg.inv(T[n // 2])
    if graph:
        T = solve_pose_graph(n, edges, n // 2)
    if model.config.get("global_pose", False):
        T = model.pose.global_sequence(model.pose.global_sequence.statistics(tokens), T)
    return T, edges


def optimize_geometry(rgb, K, T, depth, steps=20, edges=None):
    """Dense photometric/reprojection refinement, with anchor and depth-scale gauge fixed."""
    n = len(rgb)
    anchor = n // 2
    v = torch.cat((T[:, :3, 0], T[:, :3, 1]), -1).detach().clone().requires_grad_()
    translation = T[:, :3, 3].detach().clone().requires_grad_()
    scales = torch.zeros(n, device=rgb.device, requires_grad=True)
    optimizer = torch.optim.Adam([v, translation, scales], lr=0.002)
    history = []
    h, w = depth.shape[-2:]
    base = depth.detach()
    source = rgb.detach()
    fixed = T[anchor].detach()

    def current():
        out = torch.eye(4, device=rgb.device)[None].repeat(n, 1, 1)
        out[:, :3, :3] = rotation6d(v)
        out[:, :3, 3] = translation
        out[anchor] = fixed
        multipliers = torch.exp(scales.clamp(-0.5, 0.5))
        multipliers = multipliers.clone()
        multipliers[anchor] = 1
        return out, base * multipliers[:, None, None]

    for _ in range(steps):
        out, d = current()
        loss = rgb.sum() * 0
        valid_count = 0
        connected = {anchor}
        usable = []
        pairs = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 3))]
        for i, j in pairs:
            points = backproject(d[i], K[i], out[i])[::8, ::8].reshape(-1, 3)
            xy, z = project(points, K[j], out[j])
            target = sample(source[j], xy)
            observed = sample(d[j, None], xy)[:, 0]
            valid = (
                (z > 0)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < w - 1)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < h - 1)
                & ((observed - z).abs() < 0.15)
            )
            if valid.sum() > 4:
                loss = (
                    loss
                    + F.smooth_l1_loss(
                        target[valid],
                        source[i, :, ::8, ::8].permute(1, 2, 0).reshape(-1, 3)[valid],
                    )
                    + 0.1 * F.smooth_l1_loss(z[valid], observed[valid])
                )
                valid_count += int(valid.sum())
                usable.append((i, j))
        for _ in range(n):
            for i, j in usable:
                if i in connected or j in connected:
                    connected.update((i, j))
        if len(connected) != n:
            raise ValueError(
                "Disconnected geometry: some views lack consistent overlap"
            )
        if edges:
            for i, j, edge, confidence in edges:
                relative = out[j] @ torch.linalg.inv(out[i])
                loss = (
                    loss
                    + 0.02
                    * confidence.detach()
                    * (relative[:3] - edge[:3].detach()).square().mean()
                )
        if not valid_count:
            raise ValueError("Disconnected geometry: no consistent projected overlap")
        loss = (
            loss
            + 0.01 * (translation - T[:, :3, 3]).square().mean()
            + 0.01 * scales.square().mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("Geometry optimization diverged")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    out, d = current()
    return out.detach(), d.detach(), history


@torch.no_grad()
def solve_pose_graph(n, edges, anchor):
    """Chordal rotation synchronization and translation least squares, fixed gauge.

    Uses relative poses only. This is pose-graph optimization, not bundle adjustment.
    Uncalibrated network confidence is deliberately not used as a weight.
    """
    if not 0 <= anchor < n or not edges:
        raise ValueError("Invalid pose graph")
    connected = {anchor}
    for _ in range(n):
        for i, j, _, _ in edges:
            if i in connected or j in connected:
                connected.update((i, j))
    if len(connected) != n:
        raise ValueError("Disconnected pose graph")
    sample = edges[0][2]
    free = [i for i in range(n) if i != anchor]
    columns = {i: 3 * k for k, i in enumerate(free)}
    A = sample.new_zeros((3 * len(edges), 3 * (n - 1)))
    rotation_rhs = sample.new_zeros((3 * len(edges), 3))
    translation_rhs = sample.new_zeros((3 * len(edges), 1))
    identity = torch.eye(3, device=sample.device, dtype=sample.dtype)
    for k, (i, j, e, _) in enumerate(edges):
        rows = slice(3 * k, 3 * k + 3)
        R = e[:3, :3]
        if j != anchor:
            A[rows, columns[j] : columns[j] + 3] = identity
        else:
            rotation_rhs[rows] -= identity
        if i != anchor:
            A[rows, columns[i] : columns[i] + 3] = -R
        else:
            rotation_rhs[rows] += R
        translation_rhs[rows, 0] = e[:3, 3]
    # CPU solve also works when the rest of inference runs on MPS.
    rotations = torch.linalg.lstsq(
        A.cpu().double(), rotation_rhs.cpu().double()
    ).solution.to(sample)
    translations = torch.linalg.lstsq(
        A.cpu().double(), translation_rhs.cpu().double()
    ).solution.to(sample)
    T = torch.eye(4, device=sample.device, dtype=sample.dtype)[None].repeat(n, 1, 1)
    for i in free:
        r = rotations[columns[i] : columns[i] + 3]
        u, _, vh = torch.linalg.svd(r.cpu().double())
        correction = torch.eye(3, dtype=torch.float64)
        correction[2, 2] = torch.linalg.det(u @ vh)
        T[i, :3, :3] = (u @ correction @ vh).to(sample)
        T[i, :3, 3] = translations[columns[i] : columns[i] + 3, 0]
    if not torch.isfinite(T).all():
        raise ValueError("Pose graph optimization diverged")
    return T
