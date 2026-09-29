"""Train dense JEPA from RGB-only scene crops and known image augmentations."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.jepa import DenseJEPA


def pair(image, rng, size=192):
    h, w = image.shape[:2]
    y = int(rng.integers(0, max(1, h - size + 1)))
    x = int(rng.integers(0, max(1, w - size + 1)))
    a = image[y : y + size, x : x + size].astype(np.float32) / 255
    corners = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    H = cv2.getPerspectiveTransform(
        corners, corners + rng.uniform(-8, 8, (4, 2)).astype(np.float32)
    )
    b = cv2.warpPerspective(a, H, (size, size))
    # Independent exposure/color changes, no pose/depth/objects/labels.
    a = np.clip(a * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05), 0, 1)
    b = np.clip(
        b * rng.uniform(0.8, 1.2, size=(1, 1, 3)) + rng.uniform(-0.04, 0.04), 0, 1
    )
    n = (size + 3) // 4
    yy, xx = np.mgrid[:n, :n].astype(np.float32) * 4
    target = cv2.perspectiveTransform(
        np.stack([xx, yy], -1).reshape(-1, 1, 2), H
    ).reshape(n, n, 2)
    grid = target / (4 * (n - 1)) * 2 - 1
    valid = (np.abs(grid).max(-1) < 0.94).astype(np.float32)[None]
    mask = np.zeros((1, n, n), np.float32)
    for _ in range(4):
        mx, my = rng.integers(0, n - 8, 2)
        mask[0, my : my + 8, mx : mx + 8] = 1
    return (
        np.transpose(a, (2, 0, 1)).astype(np.float32),
        np.transpose(b, (2, 0, 1)).astype(np.float32),
        grid.astype(np.float32),
        valid,
        mask,
    )


@torch.inference_mode()
def validate(model, images, device):
    rng = np.random.default_rng(123)
    acc = []
    std = []
    for index in np.linspace(0, len(images) - 1, min(16, len(images)), dtype=int):
        image = images[index]
        a, b, grid, valid, _ = pair(image, rng)
        a = torch.tensor(a[None], device=device)
        b = torch.tensor(b[None], device=device)
        grid = torch.tensor(grid[None], device=device)
        za = model.encoder(a)
        zb = F.normalize(
            F.grid_sample(model.encoder(b), grid, align_corners=True), dim=1
        )
        ids = np.where(valid.flatten() > 0.99)[0]
        ids = rng.choice(ids, min(128, len(ids)), replace=False)
        ids = torch.tensor(ids, device=device)
        z1 = za.flatten(2)[0, :, ids].T
        z2 = zb.flatten(2)[0, :, ids].T
        acc.append(
            float(
                ((z1 @ z2.T).argmax(1) == torch.arange(len(ids), device=device))
                .float()
                .mean()
            )
        )
        std.append(float(za.flatten(2).std(2).mean()))
    return {
        "augmentation_matching_accuracy": float(np.mean(acc)),
        "feature_std": float(np.mean(std)),
    }


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/models"))
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(31)
    random.seed(31)
    np.random.seed(31)
    rng = np.random.default_rng(31)
    device = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    folders = sorted([d for d in args.data.iterdir() if (d / "rgb").is_dir()])
    if len(folders) < 10:
        raise ValueError("至少 10 个独立布局，以布局划分训练与验证")
    split = int(len(folders) * 0.85)
    train_dirs = folders[:split]
    val_dirs = folders[split:]

    def read(dirs):
        images = []
        paths = []
        for d in dirs:
            for f in sorted((d / "rgb").glob("*.jpg")):
                rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                # Train at original patch scale; descriptors used at the same pixel scale at inference.
                images.append(rgb)
                paths.append(str(f.relative_to(args.data)))
        return images, paths

    images, train_files = read(train_dirs)
    val_images, val_files = read(val_dirs)
    args.output.mkdir(parents=True, exist_ok=True)
    model = DenseJEPA().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4
    )
    start = 0
    best = -1
    elapsed = 0
    history = []
    latest = args.output / "shelf-jepa-latest.pt"
    if args.resume:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start = ck["step"]
        best = ck["best"]
        elapsed = ck["elapsed"]
        history = ck["history"]
        rng.bit_generator.state = ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if device == "mps" and "mps_rng" in ck:
            if ck["mps_rng"] is not None:
                torch.mps.set_rng_state(ck["mps_rng"])
    initial = validate(model, val_images, device)
    if start == 0:
        (args.output / "training-data.json").write_text(
            json.dumps(
                {
                    "train_layouts": [d.name for d in train_dirs],
                    "val_layouts": [d.name for d in val_dirs],
                    "train_files": train_files,
                    "val_files": val_files,
                    "initial_validation": initial,
                },
                indent=2,
            )
        )
    started = time.perf_counter()
    for step in range(start, args.steps):
        batch = [
            pair(images[int(rng.integers(len(images)))], rng) for _ in range(args.batch)
        ]
        tensors = [
            torch.from_numpy(np.stack([sample[k] for sample in batch])).to(device)
            for k in range(5)
        ]
        # Prediction pretraining followed by descriptor correspondence fine-tuning, same encoder.
        mw = 0.0 if step < args.steps // 4 else 0.5
        loss, metrics = model.loss(*tensors, match_weight=mw)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.update_target(0.99)
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            validation = validate(model, val_images, device)
            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "phase": "jepa" if not mw else "jepa+correspondence",
                "elapsed_seconds": elapsed + time.perf_counter() - started,
                "validation": validation,
                **{k: float(v.detach()) for k, v in metrics.items()},
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            portable = {
                "architecture": "shelf-dense-jepa-v1",
                "encoder": {k: v.cpu() for k, v in model.encoder.state_dict().items()},
                "trained_steps": step + 1,
                "validation": validation,
                "seed": 31,
            }
            if validation["augmentation_matching_accuracy"] > best:
                best = validation["augmentation_matching_accuracy"]
                atomic_save(portable, args.output / "shelf-jepa-best-training.pt")
            atomic_save(
                {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best": best,
                    "elapsed": record["elapsed_seconds"],
                    "history": history,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
                },
                latest,
            )
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            (args.output / "training-state.json").write_text(
                json.dumps(
                    {
                        "step": step + 1,
                        "total_steps": args.steps,
                        "device": device,
                        "best_validation_accuracy": best,
                    }
                )
            )
    # Publish only the complete run's validation-selected deployment checkpoint.
    candidate = args.output / "shelf-jepa-best-training.pt"
    if candidate.exists():
        candidate.replace(args.output / "shelf-jepa.pt")
    print(
        json.dumps(
            {
                "completed_steps": args.steps,
                "seconds": elapsed + time.perf_counter() - started,
                "device": device,
                "best_validation_accuracy": best,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
