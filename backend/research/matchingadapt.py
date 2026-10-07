"""Stage-one dense JEPA adaptation using visible source-source 3D correspondences."""

import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from backend.jepa import DenseJEPA, get_checkpoint, DEFAULT_CHECKPOINT


def correspondence(depth_a, depth_b, K_a, K_b, T_a, T_b):
    h, w = depth_a.shape
    yy, xx = np.mgrid[0:h:4, 0:w:4]
    d = depth_a[yy, xx]
    ray = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(K_a).T
    world = (ray * d[..., None] - T_a[:3, 3]) @ T_a[:3, :3]
    target = world @ T_b[:3, :3].T + T_b[:3, 3]
    uv = target @ K_b.T
    uv = uv[..., :2] / np.maximum(uv[..., 2:], 1e-6)
    px = np.rint(uv).astype(int)
    valid = (
        (d > 0)
        & (target[..., 2] > 0)
        & (uv[..., 0] >= -1e-6)
        & (uv[..., 0] <= w - 4 + 1e-6)
        & (uv[..., 1] >= -1e-6)
        & (uv[..., 1] <= h - 4 + 1e-6)
    )
    sampled = depth_b[px[..., 1].clip(0, h - 1), px[..., 0].clip(0, w - 1)]
    valid &= (sampled > 0) & (np.abs(sampled - target[..., 2]) < 0.03)
    grid = uv / np.array([w - 4, h - 4]) * 2 - 1
    grid[valid] = np.clip(grid[valid], -1, 1)
    return grid.astype(np.float32), valid[None].astype(np.float32)


def run(
    data, output, steps=1200, batch=4, device="mps", checkpoint=str(DEFAULT_CHECKPOINT)
):
    from PIL import Image

    if steps < 1 or batch < 1:
        raise ValueError("Invalid training budget")
    torch.set_num_threads(4)
    torch.manual_seed(20261010)
    rng = np.random.default_rng(20261010)
    root, out = Path(data), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "shelf-jepa.pt").exists():
        raise ValueError("Use fresh output")
    model = DenseJEPA().to(device)
    encoder, info = get_checkpoint(checkpoint)
    model.encoder.load_state_dict(encoder.state_dict())
    model.target.load_state_dict(encoder.state_dict())
    pools = {"train": [], "val": []}
    scene_lists = {"train": [], "val": []}
    skipped = 0
    for entry in json.loads((root / "manifest.json").read_text())["scenes"]:
        if entry["split"] == "test":
            continue
        split = entry["split"]
        scene_lists[split].append(entry["seed"])
        path = root / entry["path"]
        meta = json.loads((path / "intrinsics.json").read_text())
        K = np.asarray(meta["K"])
        T = np.load(path / "labels/geometry.npz")["T"]
        images = [
            np.asarray(Image.open(f).convert("RGB")).copy()
            for f in sorted((path / "rgb").glob("*.png"))
        ]
        depths = [np.load(path / "labels" / f"{i:04}.npy") for i in range(len(images))]
        for gap in (1, 2):
            for i in range(len(images) - gap):
                for a, b in [(i, i + gap), (i + gap, i)]:
                    grid, valid = correspondence(
                        depths[a], depths[b], K[a], K[b], T[a], T[b]
                    )
                    if valid.sum() < 128:
                        skipped += 1
                        continue
                    pools[split].append((images[a], images[b], grid, valid))
    if not pools["train"] or not pools["val"]:
        raise ValueError("Missing train/val correspondences")
    # Fixed validation pairs, no test scenes opened.
    validation_pairs = [
        pools["val"][i] for i in np.linspace(0, len(pools["val"]) - 1, 24, dtype=int)
    ]

    def tensors(samples, masking):
        result = []
        for a, b, g, v in samples:
            mask = np.zeros_like(v)
            h, w = v.shape[-2:]
            if masking:
                for _ in range(4):
                    y = int(rng.integers(0, h - 8))
                    x = int(rng.integers(0, w - 8))
                    mask[:, y : y + 8, x : x + 8] = 1
            result.append(
                (
                    a.transpose(2, 0, 1).astype(np.float32) / 255,
                    b.transpose(2, 0, 1).astype(np.float32) / 255,
                    g,
                    v,
                    mask,
                )
            )
        return [
            torch.from_numpy(np.stack([s[k] for s in result])).to(device)
            for k in range(5)
        ]

    @torch.no_grad()
    def validation():
        accuracies = []
        std = []
        vrng = np.random.default_rng(19)
        for sample in validation_pairs:
            a, b, g, v, _ = tensors([sample], False)
            za = model.encoder(a)
            zb = F.normalize(
                F.grid_sample(model.encoder(b), g, align_corners=True), dim=1
            )
            ids = np.flatnonzero(sample[3].flatten() > 0.99)
            ids = vrng.choice(ids, 128, replace=False)
            ids = torch.tensor(ids, device=device)
            left = za.flatten(2)[0, :, ids].T
            right = zb.flatten(2)[0, :, ids].T
            accuracies.append(
                float(
                    ((left @ right.T).argmax(1) == torch.arange(128, device=device))
                    .float()
                    .mean()
                )
            )
            std.append(float(za.flatten(2).std(2).mean()))
        return dict(
            visible_crossview_retrieval_accuracy=float(np.mean(accuracies)),
            feature_std=float(np.mean(std)),
        )

    initial = validation()
    best = -1.0
    beststep = 0
    bestweights = None
    history = []
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0001,
        weight_decay=0.0001,
    )
    for step in range(1, steps + 1):
        samples = [
            pools["train"][int(rng.integers(len(pools["train"])))] for _ in range(batch)
        ]
        args = tensors(samples, True)
        loss, metrics = model.loss(*args, match_weight=0.0 if step <= 100 else 0.5)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite JEPA loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        model.update_target(0.99)
        if step % 100 == 0 or step == steps:
            val = validation()
            row = dict(
                step=step,
                loss=float(loss.detach()),
                validation=val,
                elapsed_seconds=time.perf_counter() - start,
                **{k: float(v.detach()) for k, v in metrics.items()},
            )
            history.append(row)
            print(json.dumps(row), flush=True)
            if val["visible_crossview_retrieval_accuracy"] > best:
                best = val["visible_crossview_retrieval_accuracy"]
                beststep = step
                bestweights = copy.deepcopy(
                    {k: v.cpu() for k, v in model.encoder.state_dict().items()}
                )
            (out / "history.json").write_text(json.dumps(history, indent=2))
    protocol = dict(
        train_scenes=scene_lists["train"],
        validation_scenes=scene_lists["val"],
        test_loaded=False,
        selected_step=beststep,
        steps=steps,
        seed=20261010,
        batch=batch,
        device=device,
        source_resolution=[256, 192],
        train_pairs=len(pools["train"]),
        val_pairs=len(pools["val"]),
        skipped_pairs=skipped,
        initial_validation=initial,
        selected_validation_accuracy=best,
        source_checkpoint_sha256=info["checkpoint_sha256"],
        supervision="source-source GT depth/pose visible correspondence; no target panorama",
        model="existing DenseJEPA hybrid prediction+matching loss, EMA stop-gradient",
        elapsed_seconds=time.perf_counter() - start,
    )
    torch.save(
        dict(
            architecture="shelf-dense-jepa-v1",
            encoder=bestweights,
            trained_steps=beststep,
            validation={"visible_crossview_retrieval_accuracy": best},
            training_protocol=protocol,
        ),
        out / "shelf-jepa.pt",
    )
    torch.save(
        dict(
            model={k: v.cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),
            step=steps,
            numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
        ),
        out / "training-state.pt",
    )
    (out / "report.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps(dict(status="complete", selected_step=beststep)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="mps")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    run(**vars(p.parse_args()))
