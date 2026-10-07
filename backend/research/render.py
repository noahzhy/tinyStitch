import torch
from .geometry import pixels


def rays(K, T, h, w, xy=None, projection="perspective"):
    p = (
        pixels(h, w, K.device).reshape(-1, 3)
        if xy is None
        else torch.cat((xy, torch.ones_like(xy[:, :1])), -1)
    )
    camera_points = p @ torch.linalg.inv(K).T
    o = -T[:3, :3].T @ T[:3, 3]
    if projection == "orthographic":
        camera_points[:, 2] = 0
        origins = camera_points @ T[:3, :3] + o
        directions = T[2, :3].expand_as(origins)
        return origins, directions
    if projection != "perspective":
        raise ValueError("Unknown target projection")
    d = camera_points @ T[:3, :3]
    return o.expand_as(d), d


def render_rays(
    model,
    rgb,
    K,
    T,
    structure,
    appearance,
    depth,
    target_K,
    target_T,
    xy,
    samples=32,
    use_vae=True,
    projection="perspective",
):
    o, d = rays(target_K, target_T, *rgb.shape[-2:], xy, projection)
    mode = model.config.get("ray_sampling", "uniform")
    guided = mode != "uniform"
    if mode == "hybrid":
        steps = hybrid_steps(o, d, K, T, depth, samples)
    elif guided:
        steps = depth_guided_steps(o, d, K, T, depth, samples)
    else:
        steps = torch.linspace(0.15, 3.5, samples, device=rgb.device)[None].expand(
            len(xy), -1
        )
    points = o[:, None] + d[:, None] * steps[..., None]
    directions = torch.nn.functional.normalize(d, dim=-1)[:, None].expand_as(points)
    if (
        not torch.is_grad_enabled()
        and model.config.get("opaque_surface", False)
        and hasattr(model.field, "geometry_sigma")
    ):
        # Select geometry first; evaluate the deep color network at one opaque hit.
        flattened = points.reshape(-1, 3)
        features = model.field.query(flattened, rgb, K, T, structure, appearance, depth)
        sigma = model.field.geometry_sigma(flattened, features).reshape(-1, samples)
        spacing = steps[:, 1:] - steps[:, :-1]
        last = spacing[:, -1:] if guided else steps.new_full((len(xy), 1), 0.1)
        delta = torch.cat((spacing, last), -1) * d.norm(dim=-1, keepdim=True)
        alpha = -torch.expm1(-sigma * delta)
        trans = torch.cumprod(
            torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-8), 1), 1
        )[:, :-1]
        weights = alpha * trans
        chosen = weights.argmax(-1)
        rows = torch.arange(len(xy), device=rgb.device)
        selected_features = tuple(
            v.reshape(len(xy), samples, -1)[rows, chosen] for v in features
        )
        selected_features = (*selected_features[:4], selected_features[4][:, 0])
        _, color, support = model.field.evaluate(
            points[rows, chosen], directions[rows, chosen], selected_features, use_vae
        )
        valid = (weights.sum(-1) > 1e-12).float()
        return color * valid[:, None], valid, support * valid
    sigma, color, support = model.field(
        points.reshape(-1, 3),
        directions.reshape(-1, 3),
        rgb,
        K,
        T,
        structure,
        appearance,
        depth,
        use_vae,
    )
    sigma = sigma.reshape(-1, samples)
    color = color.reshape(-1, samples, 3)
    support = support.reshape(-1, samples)
    spacing = steps[:, 1:] - steps[:, :-1]
    last = spacing[:, -1:] if guided else steps.new_full((len(xy), 1), 0.1)
    delta = torch.cat((spacing, last), -1) * d.norm(dim=-1, keepdim=True)
    alpha = 1 - torch.exp(-sigma * delta)
    trans = torch.cumprod(
        torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-8), 1), 1
    )[:, :-1]
    weights = alpha * trans
    if model.config.get("opaque_surface", False):
        weights = opaque_weights(weights)
    return (
        (weights[..., None] * color).sum(1),
        weights.sum(1),
        (weights * support).sum(1),
    )


@torch.no_grad()
def render_image(
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
    chunk=256,
    samples=32,
    use_vae=True,
    projection="perspective",
    local_sources=None,
):
    if local_sources is not None and (
        local_sources < 2 or projection != "orthographic"
    ):
        raise ValueError(
            "Local source selection requires orthographic rays and >=2 sources"
        )
    xy = pixels(h, w, rgb.device)[..., :2].reshape(-1, 2)
    order = (
        torch.argsort(xy[:, 0], stable=True)
        if local_sources is not None
        else torch.arange(len(xy), device=rgb.device)
    )
    xy = xy[order]
    colors = []
    masks = []
    for start in range(0, len(xy), chunk):
        ids = torch.arange(len(rgb), device=rgb.device)
        if local_sources is not None:
            ids = select_local_sources(
                xy[start : start + chunk], target_K, target_T, T, local_sources
            )

        color, alpha, support = render_rays(
            model,
            rgb[ids],
            K[ids],
            T[ids],
            structure[ids],
            appearance[ids],
            depth[ids],
            target_K,
            target_T,
            xy[start : start + chunk],
            samples,
            use_vae,
            projection,
        )
        colors.append(color)
        masks.append((alpha > 0.4) & (support > 0.3))
    inverse = torch.argsort(order)
    return torch.cat(colors)[inverse].reshape(h, w, 3), torch.cat(masks)[
        inverse
    ].reshape(h, w)


@torch.no_grad()
def depth_guided_steps(
    origins, directions, K, T, depth, samples=64, near=0.15, far=3.5
):
    """Source-depth proposals only; no target depth or target RGB is consulted.

    Search for the nearest supported surface, refine its intersection by fixed-point
    source-depth projection, and sample a narrow volume around it. Gradients through
    sampling locations are detached, as in hierarchical NeRF sampling.
    """
    from .geometry import project, sample

    coarse = torch.linspace(near, far, 32, device=origins.device)
    points = origins[:, None] + directions[:, None] * coarse[None, :, None]
    scores = []
    h, w = depth.shape[-2:]
    for i in range(len(depth)):
        xy, z = project(points, K[i], T[i])
        observed = sample(depth[i, None], xy)[..., 0]
        valid = (
            (z > 0)
            & (observed > 0)
            & (xy[..., 0] >= 0)
            & (xy[..., 0] <= w - 1)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] <= h - 1)
        )
        relative = (z - observed) / observed.clamp_min(0.05)
        scores.append(
            valid.float() * torch.exp(-relative.square() * 400) / (coarse[None] + 0.1)
        )
    score = torch.stack(scores).max(0).values
    center = coarse[score.argmax(-1)]
    for _ in range(8):
        points = origins + directions * center[:, None]
        proposals = []
        weights = []
        for i in range(len(depth)):
            xy, z = project(points, K[i], T[i])
            observed = sample(depth[i, None], xy)[:, 0]
            valid = (
                (z > 0)
                & (observed > 0)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] <= w - 1)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] <= h - 1)
            )
            az = (directions @ T[i, :3, :3].T)[:, 2]
            bz = (origins @ T[i, :3, :3].T + T[i, :3, 3])[:, 2]
            proposal = (observed - bz) / az.clamp_min(1e-6)
            valid &= (az > 0) & (proposal > near) & (proposal < far)
            weight = valid.float() * torch.exp(
                -((z - observed) / observed.clamp_min(0.05)).square() * 400
            )
            proposals.append(proposal.clamp(near, far))
            weights.append(weight)
        weight = torch.stack(weights)
        den = weight.sum(0)
        candidate = (torch.stack(proposals) * weight).sum(0) / den.clamp_min(1e-8)
        center = torch.where(den > 1e-8, center * 0.2 + candidate * 0.8, center)
    offsets = torch.linspace(-0.06, 0.06, samples, device=origins.device)
    return (center[:, None] * (1 + offsets[None])).clamp(near, far)


def opaque_weights(weights):
    """One opaque surface per supported ray, with straight-through gradients.

    The soft distribution is only a gradient surrogate. Forward RGB never mixes
    foreground and background samples or uses partial material transparency.
    """
    mass = weights.sum(-1, keepdim=True)
    soft = weights / mass.clamp_min(1e-12)
    hard = torch.nn.functional.one_hot(weights.argmax(-1), weights.shape[-1]).to(
        weights.dtype
    )
    hard = hard * (mass > 1e-12)
    return hard + soft - soft.detach()


@torch.no_grad()
def hybrid_steps(origins, directions, K, T, depth, samples=64):
    # Cover hidden/recessed surfaces as well as source-depth proposals.
    local = depth_guided_steps(origins, directions, K, T, depth, samples // 2)
    global_steps = torch.linspace(
        0.15, 3.5, samples - samples // 2, device=origins.device
    )[None].expand(len(origins), -1)
    return torch.cat((local, global_steps), -1).sort(-1).values


@torch.no_grad()
def select_local_sources(xy, target_K, target_T, cameras, limit):
    """Nearby source centers for a narrow orthographic column chunk; no GT required.

    Target ray origin fixes lateral location. Depth is not needed for parallel rays.
    Ranking ignores the along-ray distance to avoid bias from canonical scale.
    """
    q = torch.cat((xy.mean(0), xy.new_ones(1))) @ torch.linalg.inv(target_K).T
    q[2] = 0
    origin = (q - target_T[:3, 3]) @ target_T[:3, :3]
    centers = -(cameras[:, :3, :3].transpose(1, 2) @ cameras[:, :3, 3, None])[:, :, 0]
    offset = centers - origin
    direction = target_T[2, :3]
    lateral = offset - (offset @ direction)[:, None] * direction
    return lateral.square().sum(-1).argsort(stable=True)[: min(limit, len(cameras))]
