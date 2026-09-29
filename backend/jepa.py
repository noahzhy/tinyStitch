"""Small dense cross-view JEPA experiment with EMA targets, not Meta pretrained weights."""

import copy
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "artifacts/models/shelf-jepa.pt"


class DenseEncoder(nn.Module):
    stride = 4
    dim = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1),
        )

    def forward(self, x):
        return F.normalize(self.net(x * 2 - 1), dim=1, eps=1e-6)


class DenseJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DenseEncoder()
        self.target = copy.deepcopy(self.encoder)
        self.target.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 64, 1),
        )

    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for target, context in zip(self.target.parameters(), self.encoder.parameters()):
            target.lerp_(context, 1 - momentum)

    def loss(self, context, target_view, grid, valid, mask, match_weight=0.5):
        # Target sees the complete second view. Stop-gradient prevents target collapse by prediction.
        with torch.no_grad():
            target = F.grid_sample(self.target(target_view), grid, align_corners=True)
        masked = self.encoder(
            context * (1 - F.interpolate(mask, size=context.shape[-2:], mode="nearest"))
        )
        prediction = F.normalize(self.predictor(masked), dim=1)
        weight = (mask * valid).expand_as(target)
        prediction_loss = (
            F.smooth_l1_loss(prediction, target, reduction="none") * weight
        ).sum() / weight.sum().clamp_min(1)
        full = self.encoder(context)
        flat = full.permute(0, 2, 3, 1).reshape(-1, 64)
        std = flat.std(0)
        variance = F.relu(0.075 - std).mean()
        sample = flat[:: max(1, len(flat) // 512)][:512]
        centered = sample - sample.mean(0)
        cov = centered.T @ centered / max(1, len(sample) - 1)
        covariance = (cov.square().sum() - cov.diag().square().sum()) / 64
        # Auxiliary known-augmentation correspondence objective makes dense tokens useful for matching.
        # This is a JEPA-inspired hybrid, not a reproduction of pure I-JEPA or V-JEPA.
        matching = full.sum() * 0
        accuracy = full.sum() * 0
        if match_weight:
            terms = []
            correct = []
            for b in range(context.shape[0]):
                indices = torch.where(valid[b, 0].flatten() > 0.99)[0]
                indices = indices[
                    torch.randperm(len(indices), device=context.device)[:128]
                ]
                a = full[b].flatten(1)[:, indices].T
                z = target[b].flatten(1)[:, indices].T
                z = F.normalize(z, dim=1)
                logits = a @ z.T / 0.07
                labels = torch.arange(len(indices), device=context.device)
                terms.append(F.cross_entropy(logits, labels))
                correct.append((logits.argmax(1) == labels).float().mean())
            matching = torch.stack(terms).mean()
            accuracy = torch.stack(correct).mean()
        loss = (
            prediction_loss + 4 * variance + 0.05 * covariance + match_weight * matching
        )
        return loss, {
            "prediction_loss": prediction_loss,
            "variance_loss": variance,
            "covariance_loss": covariance,
            "matching_loss": matching,
            "matching_accuracy": accuracy,
            "feature_std": std.mean(),
        }


def get_checkpoint(path=DEFAULT_CHECKPOINT):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            "缺少已训练货架 JEPA 权重，请先运行 train_jepa.py；不会用随机网络或 SIFT 代替 JEPA"
        )
    raw = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if (
        checkpoint.get("trained_steps", 0) < 1
        or checkpoint.get("architecture") != "shelf-dense-jepa-v1"
    ):
        raise RuntimeError("权重不是有效的已训练货架 JEPA 模型")
    encoder = DenseEncoder()
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    return encoder, {
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "trained_steps": checkpoint["trained_steps"],
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "validation": checkpoint.get("validation"),
    }


class JEPADescriptor:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="auto"):
        self.model, self.info = get_checkpoint(checkpoint)
        self.device = (
            "mps"
            if device == "auto" and torch.backends.mps.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        self.model.to(self.device)
        self.info["device"] = self.device
        self.detector = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.025)

    @torch.inference_mode()
    def extract(self, image):
        # SIFT only proposes image locations. Its descriptors are never computed or concatenated.
        keypoints = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        if not keypoints:
            return [], None
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)[None]
            .to(self.device)
            .float()
            / 255
        )
        dense = self.model(tensor)
        xy = torch.tensor([k.pt for k in keypoints], device=self.device)
        xy = xy / 4
        xy[:, 0] = xy[:, 0] / (dense.shape[3] - 1) * 2 - 1
        xy[:, 1] = xy[:, 1] / (dense.shape[2] - 1) * 2 - 1
        descriptors = F.grid_sample(dense, xy[None, None], align_corners=True)[
            0, :, 0, :
        ].T
        descriptors = F.normalize(descriptors, dim=1)
        return keypoints, descriptors.cpu().numpy().astype(np.float32)
