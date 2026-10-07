import copy
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import pixels, project, sample, rotation6d


class Encoder(nn.Module):
    def __init__(self, dim=192, patch=8, layers=2):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Linear(2, dim)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0),
            layers,
            enable_nested_tensor=False,
        )

    def forward(self, x, mask=None):
        z = self.embed(x)
        h, w = z.shape[-2:]
        xy = pixels(h, w, x.device)[..., :2]
        xy = xy / xy.new_tensor([max(w - 1, 1), max(h - 1, 1)])
        z = z.flatten(2).transpose(1, 2)
        z = z + self.pos(xy.reshape(-1, 2))[None]
        if mask is not None:
            z = z.masked_fill(mask[..., None], 0)
        return self.blocks(z), h, w


class CrossJEPA(nn.Module):
    def __init__(self, dim=192, patch=8, layers=2):
        super().__init__()
        self.encoder = Encoder(dim, patch, layers)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.query = nn.Linear(2, dim)
        self.predictor = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.out = nn.Linear(dim, dim)

    @torch.no_grad()
    def ema(self, m=0.99):
        for a, b in zip(self.target.parameters(), self.encoder.parameters()):
            a.lerp_(b, 1 - m)

    def predict(self, a, b, mask, queries):
        za, _, _ = self.encoder(a, mask)
        zb, _, _ = self.encoder(b, mask)
        context = torch.cat((za, zb), 1)
        q = self.query(queries)
        return self.out(self.predictor(q, context, context, need_weights=False)[0])


class GlobalPoseRefiner(nn.Module):
    """Sequence attention over pooled spatial statistics and anchored local poses."""

    def __init__(self, dim):
        super().__init__()
        self.input = nn.Linear(dim * 2 + 18, 128)
        layer = nn.TransformerEncoderLayer(
            128, 4, 256, dropout=0.0, batch_first=True, norm_first=True
        )
        self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(128, 9)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def statistics(tokens):
        return torch.cat((tokens.mean(1), tokens.std(1, unbiased=False)), -1)

    def forward(self, statistics, base):
        n = len(base)
        rank = torch.linspace(-1, 1, n, device=base.device)
        phases = rank[:, None] * (2 ** torch.arange(4, device=base.device)) * torch.pi
        positional = torch.cat(
            (phases.sin(), phases.cos(), rank.new_full((n, 1), n / 60)), -1
        )
        pose = torch.cat((base[:, :3, 0], base[:, :3, 1], base[:, :3, 3]), -1)
        features = torch.cat((statistics, pose, positional), -1)
        residual = self.output(self.context(self.input(features)[None])[0])
        identity = residual.new_tensor([1.0, 0, 0, 0, 1, 0])
        rotation = (
            rotation6d(identity + 0.15 * residual[:, :6].tanh()) @ base[:, :3, :3]
        )
        translation = base[:, :3, 3] + 0.25 * residual[:, 6:].tanh()
        top = torch.cat((rotation, translation[:, :, None]), -1)
        out = torch.cat((top, base[:, 3:4, :]), 1)
        return out @ torch.linalg.inv(out[n // 2])


class PoseDepth(nn.Module):
    def __init__(
        self, dim, refinement_patch=None, architecture="legacy", max_tokens=None
    ):
        super().__init__()
        self.architecture = architecture
        self.max_tokens = max_tokens
        if architecture not in ("legacy", "relational"):
            raise ValueError("Unknown pose architecture")
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 11)
        )
        if architecture == "relational":
            self.head = nn.Sequential(
                nn.Linear(dim * 4, dim * 3),
                nn.GELU(),
                nn.Linear(dim * 3, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, 11),
            )
        self.depth = nn.Linear(dim, 1)
        self.refinement_patch = refinement_patch
        if refinement_patch:
            self.depth_refiner = nn.Sequential(
                nn.Conv2d(dim, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, refinement_patch**2, 3, padding=1),
                nn.PixelShuffle(refinement_patch),
            )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[:6] = torch.tensor([1.0, 0, 0, 0, 1, 0])
            self.head[-1].bias[6] = 1

    def descriptor(self, a, b):
        if self.max_tokens is not None:
            if self.max_tokens < 1:
                raise ValueError("Invalid pose token budget")

            def reduce(x):
                if x.shape[1] <= self.max_tokens:
                    return x
                ids = (
                    torch.linspace(0, x.shape[1] - 1, self.max_tokens, device=x.device)
                    .round()
                    .long()
                )
                return x[:, ids]

            a, b = reduce(a), reduce(b)
        z = self.cross(a, b, b, need_weights=False)[0]
        if self.architecture == "relational":
            return torch.cat(
                (a.mean(1), b.mean(1), b.mean(1) - a.mean(1), z.mean(1)), -1
            )
        return torch.cat((a.mean(1), z.mean(1)), -1)

    def decode(self, descriptor):
        v = self.head(descriptor)
        R = rotation6d(v[:, :6])
        direction = F.normalize(v[:, 6:9], dim=-1)
        length = F.softplus(v[:, 9]) + 0.001
        return R, direction, length, torch.sigmoid(v[:, 10])

    def pair(self, a, b):
        return self.decode(self.descriptor(a, b))

    def depths(self, tokens, h, w, size):
        logits = self.depth(tokens).transpose(1, 2).reshape(-1, 1, h, w)
        if self.refinement_patch is None:
            # Preserve the numerical behavior of previously delivered checkpoints.
            return F.interpolate(
                F.softplus(logits) + 0.05,
                size=size,
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        logits = F.interpolate(
            logits,
            size=size,
            mode="bilinear",
            align_corners=self.refinement_patch is None,
        )
        if self.refinement_patch:
            refined = self.depth_refiner(
                tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], h, w)
            )
            logits = logits + F.interpolate(
                refined, size=size, mode="bilinear", align_corners=False
            )
        return (F.softplus(logits) + 0.05)[:, 0]


class VAE(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, latent * 2, 4, 2, 1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, stochastic=True):
        mu, logvar = self.enc(x).chunk(2, 1)
        logvar = logvar.clamp(-8, 4)
        z = mu + torch.randn_like(mu) * torch.exp(logvar / 2) if stochastic else mu
        return self.dec(z), mu, logvar


class Field(nn.Module):
    def __init__(self, dim, latent=16):
        super().__init__()
        self.density = nn.Sequential(
            nn.Linear(dim + 4, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.color = nn.Sequential(
            nn.Linear(dim + latent + 6, 64), nn.GELU(), nn.Linear(64, 3)
        )
        self.residual = nn.Parameter(torch.zeros(4))

    def feature_coordinates(self, xy, grid_size, image_size):
        gh, gw = grid_size
        h, w = image_size
        if getattr(self, "feature_center_mapping", False):
            coords = (xy + 0.5) * xy.new_tensor([gw / w, gh / h]) - 0.5
            # Visible source boundary pixels use the nearest patch feature.
            return torch.stack(
                (coords[..., 0].clamp(0, gw - 1), coords[..., 1].clamp(0, gh - 1)), -1
            )
        return xy * xy.new_tensor([(gw - 1) / (w - 1), (gh - 1) / (h - 1)])

    def query(self, points, rgb, K, T, structure, appearance, depth):
        zs = []
        apps = []
        colors = []
        weights = []
        deltas = []
        h, w = rgb.shape[-2:]
        for i in range(len(rgb)):
            xy, z = project(points, K[i], T[i])
            valid = (
                (z > 0)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] <= w - 1)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] <= h - 1)
            )
            observed = sample(depth[i, None], xy)[:, 0]
            delta = (z - observed) / (observed.clamp_min(0.05))
            weight = valid.float() * torch.exp(-delta.square() * 100)
            zs.append(
                sample(
                    structure[i],
                    self.feature_coordinates(xy, structure.shape[-2:], (h, w)),
                )
            )
            apps.append(
                sample(
                    appearance[i],
                    self.feature_coordinates(xy, appearance.shape[-2:], (h, w)),
                )
            )
            if getattr(self, "occlusion_aware", False):
                from .geometry import sample_visible_rgb

                visible_color, _, coverage = sample_visible_rgb(rgb[i], xy, depth[i], z)
                # Color interpolation must not erase density support at imperfect depth edges.
                nearest_rgb = sample(rgb[i], xy.round())
                colors.append(
                    torch.where((coverage > 0)[:, None], visible_color, nearest_rgb)
                )
            else:
                colors.append(sample(rgb[i], xy))
            weights.append(weight)
            deltas.append(delta)
        weight = torch.stack(weights)
        den = weight.sum(0).clamp_min(1e-6)
        aggregate = (
            lambda values: (torch.stack(values) * weight[..., None]).sum(0)
            / den[:, None]
        )
        z = aggregate(zs)
        a = aggregate(apps)
        base = aggregate(colors)
        if getattr(self, "color_fusion", "weighted") == "consensus":
            observed_colors = torch.stack(colors)
            # Weighted channel medians guide selection of one observed RGB.
            ordered, order = observed_colors.sort(dim=0)
            expanded = weight[..., None].expand_as(observed_colors)
            sorted_weights = torch.gather(expanded, 0, order)
            cumulative = sorted_weights.cumsum(0)
            median_ids = (
                (cumulative < den[None, :, None] * 0.5)
                .sum(0)
                .clamp_max(len(colors) - 1)
            )
            median = ordered.gather(0, median_ids[None])[0]
            distance = (observed_colors - median[None]).abs().sum(-1)
            cost = distance + 0.03 * (-weight.clamp_min(1e-8).log())
            cost = cost.masked_fill(weight <= 0, float("inf"))
            selected = cost.argmin(0)
            base = observed_colors[
                selected, torch.arange(len(points), device=points.device)
            ]
            base = torch.where((weight.sum(0) > 0)[:, None], base, base * 0)
        if getattr(self, "color_fusion", "weighted") == "nearest":
            # Near-front shelf capture: closest lateral camera limits parallax
            # sensitivity to uncertain depth. Density and feature support stay unchanged.
            centers = -(T[:, :3, :3].transpose(1, 2) @ T[:, :3, 3, None])[:, :, 0]
            lateral = (points[None, :, :2] - centers[:, None, :2]).square().sum(-1)
            cost = lateral + 0.03 * (-weight.clamp_min(1e-8).log())
            cost = cost.masked_fill(weight < 0.15, float("inf"))
            selected = cost.argmin(0)
            chosen = torch.stack(colors)[
                selected, torch.arange(len(points), device=points.device)
            ]
            base = torch.where(
                torch.isfinite(cost.min(0).values)[:, None], chosen, base
            )
        delta = aggregate([d[:, None] for d in deltas])
        support = weight.max(0).values
        return z, a, base, delta, support

    def evaluate(self, points, direction, features, use_vae=True):
        z, a, base, delta, support = features
        sigma = (
            F.softplus(
                self.density(torch.cat((z, points, delta), -1))[:, 0] + self.residual[0]
            )
            * support
            * 30
        )
        correction = (
            self.color(torch.cat((z, a if use_vae else a * 0, base, direction), -1))
            + self.residual[1:]
        )
        color = (base + 0.15 * torch.tanh(correction)).clamp(0, 1)
        return sigma, color, support

    def forward(
        self, points, direction, rgb, K, T, structure, appearance, depth, use_vae=True
    ):
        return self.evaluate(
            points,
            direction,
            self.query(points, rgb, K, T, structure, appearance, depth),
            use_vae,
        )


class FourierField(Field):
    """Higher-capacity scene field, explicit coordinate bandwidth and unrestricted color residual."""

    def __init__(self, dim, latent=16, bands=8):
        super().__init__(dim, latent)
        self.bands = bands
        coordinates = 3 + 6 * bands
        self.density = nn.Sequential(
            nn.Linear(dim + coordinates + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.color = nn.Sequential(
            nn.Linear(dim + latent + coordinates + 6, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def positional(self, points):
        frequencies = (
            2 ** torch.arange(self.bands, device=points.device, dtype=points.dtype)
            * torch.pi
        )
        phases = points[..., None] * frequencies
        return torch.cat(
            (points, phases.sin().flatten(-2), phases.cos().flatten(-2)), -1
        )

    def corrected_color(self, base, correction):
        limit = getattr(self, "color_residual_limit", None)
        if limit is None:
            return torch.sigmoid(torch.logit(base.clamp(0.001, 0.999)) + correction)
        return (base + limit * torch.tanh(correction)).clamp(0, 1)

    def geometry_sigma(self, points, features):
        z, a, base, delta, support = features
        encoded = self.positional(points)
        sigma = (
            F.softplus(
                self.density(torch.cat((z, encoded, delta), -1))[:, 0]
                + self.residual[0]
            )
            * support
            * 60
        )
        return sigma

    def evaluate(self, points, direction, features, use_vae=True):
        z, a, base, delta, support = features
        encoded = self.positional(points)
        sigma = self.geometry_sigma(points, features)
        correction = (
            self.color(
                torch.cat((z, a if use_vae else a * 0, encoded, base, direction), -1)
            )
            + self.residual[1:]
        )
        color = self.corrected_color(base, correction)
        return sigma, color, support


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.net(x) * 0.1


class DeepFourierField(FourierField):
    """Eight-layer residual color refinement, zero output preserves a warm start."""

    def __init__(self, dim, latent=16, bands=8):
        super().__init__(dim, latent, bands)
        inputs = dim + latent + 3 + 6 * bands + 6
        self.color_refinement = nn.Sequential(
            nn.Linear(inputs, 192),
            nn.GELU(),
            *[ResidualBlock(192) for _ in range(3)],
            nn.LayerNorm(192),
            nn.Linear(192, 3),
        )
        nn.init.zeros_(self.color_refinement[-1].weight)
        nn.init.zeros_(self.color_refinement[-1].bias)

    def evaluate(self, points, direction, features, use_vae=True):
        z, a, base, delta, support = features
        inputs = torch.cat(
            (z, a if use_vae else a * 0, self.positional(points), base, direction), -1
        )
        correction = (
            self.color(inputs) + self.color_refinement(inputs) + self.residual[1:]
        )
        color = self.corrected_color(base, correction)
        return self.geometry_sigma(points, features), color, support


class Pipeline(nn.Module):
    def __init__(
        self,
        dim=192,
        patch=8,
        layers=2,
        depth_refinement=False,
        field_arch="legacy",
        ray_sampling="uniform",
        opaque_surface=False,
        occlusion_aware=False,
        pose_arch="legacy",
        pose_tokens=None,
        pose_graph=False,
        color_residual_limit=None,
        field_bands=8,
        appearance_geometry=False,
        color_fusion="weighted",
        feature_center_mapping=False,
        global_pose=False,
    ):
        super().__init__()
        self.config = dict(dim=dim, patch=patch, layers=layers)
        if depth_refinement:
            self.config["depth_refinement"] = True
        if field_arch != "legacy":
            self.config["field_arch"] = field_arch
        if ray_sampling not in ["uniform", "depth_guided", "hybrid"]:
            raise ValueError("Unknown ray sampling")
        if ray_sampling != "uniform":
            self.config["ray_sampling"] = ray_sampling
        if opaque_surface:
            self.config["opaque_surface"] = True
        self.jepa = CrossJEPA(dim, patch, layers)
        self.pose = PoseDepth(
            dim, patch if depth_refinement else None, pose_arch, pose_tokens
        )
        if global_pose:
            self.config["global_pose"] = True
            self.pose.global_sequence = GlobalPoseRefiner(dim)
        if pose_graph:
            self.config["pose_graph"] = True
        if pose_arch != "legacy":
            self.config["pose_arch"] = pose_arch
        if pose_tokens is not None:
            self.config["pose_tokens"] = pose_tokens
        self.vae = VAE()
        if field_arch not in ["legacy", "fourier", "deep_fourier"]:
            raise ValueError("Unknown field architecture")
        if field_bands < 1 or field_bands > 12:
            raise ValueError("Invalid field bandwidth")
        if field_bands != 8:
            self.config["field_bands"] = field_bands
        if appearance_geometry:
            self.config["appearance_geometry"] = True
        if field_arch == "legacy":
            self.field = Field(dim)
        else:
            self.field = {"fourier": FourierField, "deep_fourier": DeepFourierField}[
                field_arch
            ](dim, bands=field_bands)
        if color_residual_limit is not None:
            if not 0 <= color_residual_limit <= 1:
                raise ValueError("Invalid color residual limit")
            self.config["color_residual_limit"] = color_residual_limit
        if color_fusion not in ("weighted", "consensus"):
            raise ValueError("Unknown color fusion")
        if color_fusion != "weighted":
            self.config["color_fusion"] = color_fusion
        self.field.feature_center_mapping = feature_center_mapping
        if feature_center_mapping:
            self.config["feature_center_mapping"] = True
        self.field.color_fusion = color_fusion
        self.field.color_residual_limit = color_residual_limit
        self.field.opaque_surface = opaque_surface
        self.field.occlusion_aware = occlusion_aware
        if occlusion_aware:
            self.config["occlusion_aware"] = True

    def features_multires(self, rgb, geometry_size=(192, 256), batch_size=1):
        """Keep native RGB for rendering; encode geometry at an explicit working size."""
        if batch_size < 1 or min(geometry_size) < self.config["patch"]:
            raise ValueError("Invalid multires feature budget")
        groups = []
        for i in range(0, len(rgb), batch_size):
            source = rgb[i : i + batch_size]
            geometry = F.interpolate(
                source, size=geometry_size, mode="bilinear", align_corners=False
            )
            tokens, h, w = self.jepa.encoder(geometry)
            depth = self.pose.depths(tokens, h, w, geometry_size)
            depth = F.interpolate(
                depth[:, None],
                size=source.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            # Decode is unnecessary for the appearance prior at inference time.
            appearance = self.vae.enc(
                geometry if self.config.get("appearance_geometry", False) else source
            ).chunk(2, 1)[0]
            structure = tokens.transpose(1, 2).reshape(len(source), -1, h, w)
            groups.append((tokens, structure, appearance, depth))
        return tuple(torch.cat([g[k] for g in groups], 0) for k in range(4))

    def features_batched(self, rgb, batch_size=4):
        if batch_size < 1:
            raise ValueError("Invalid feature batch size")
        groups = [
            self.features(rgb[i : i + batch_size])
            for i in range(0, len(rgb), batch_size)
        ]
        return tuple(torch.cat([g[k] for g in groups], 0) for k in range(4))

    def features(self, rgb):
        tokens, h, w = self.jepa.encoder(rgb)
        depth = self.pose.depths(tokens, h, w, rgb.shape[-2:])
        _, appearance, _ = self.vae(rgb, False)
        structure = tokens.transpose(1, 2).reshape(len(rgb), -1, h, w)
        return tokens, structure, appearance, depth
