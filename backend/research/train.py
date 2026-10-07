import json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import load_supervised
from .geometry import correspondence, sample, pixels, rotation_error
from .optimization import predict_cameras
from .render import render_rays
from .model import Pipeline

STAGES = ["jepa", "pose", "vae", "render_gt", "render_pred", "joint"]


def jepa_loss(model, rgb, K, T, depth, same_image=False):
    patch = model.config["patch"]
    h, w = rgb.shape[-2:]
    gh, gw = h // patch, w // patch
    terms = []
    accuracies = []
    skips = 0
    for a, b in [(0, 1), (1, 0)]:
        if same_image:
            b = a
        xy, valid = correspondence(depth[a], depth[b], K[a], K[b], T[a], T[b])
        centers = pixels(gh, gw, rgb.device)[..., :2] * patch + (patch - 1) / 2
        target_xy = sample(xy.permute(2, 0, 1), centers)
        good = sample(valid.float()[None], centers)[..., 0] > 0.99
        select = good.flatten() & (torch.rand(gh * gw, device=rgb.device) < 0.5)
        if not select.any():
            skips += 1
            continue
        # Hide both source and corresponding reference positions, avoiding direct target copying.
        mask_a = select[None]
        mask_b = torch.zeros_like(mask_a)
        coords = (target_xy.reshape(-1, 2)[select] / patch).long()
        ids = coords[:, 1].clamp(0, gh - 1) * gw + coords[:, 0].clamp(0, gw - 1)
        mask_b[0, ids] = True
        za, _, _ = model.jepa.encoder(rgb[a : a + 1], mask_a)
        zb, _, _ = model.jepa.encoder(rgb[b : b + 1], mask_b)
        queries = target_xy.reshape(-1, 2)[select] / target_xy.new_tensor(
            [w - 1, h - 1]
        )
        q = model.jepa.query(queries[None])
        context = torch.cat((za, zb), 1)
        prediction = model.jepa.out(
            model.jepa.predictor(q, context, context, need_weights=False)[0]
        )[0]
        with torch.no_grad():
            target, _, _ = model.jepa.target(rgb[b : b + 1])
            target = target[0].T.reshape(-1, gh, gw)
            goal = sample(target, target_xy.reshape(-1, 2)[select] / patch - 0.5)
        prediction = F.normalize(prediction, dim=-1)
        goal = F.normalize(goal, dim=-1)
        full = F.normalize(za[0], dim=-1)
        variance = F.relu(0.04 - full.std(0)).mean()
        terms.append(F.mse_loss(prediction, goal) + variance * 2)
        accuracies.append(
            float(
                (
                    (prediction @ goal.T).argmax(-1)
                    == torch.arange(len(goal), device=rgb.device)
                )
                .float()
                .mean()
            )
        )
    if not terms:
        raise ValueError("No valid JEPA correspondence")
    return torch.stack(terms).mean(), {
        "retrieval_accuracy": sum(accuracies) / len(accuracies),
        "feature_std": float(full.std(0).mean().detach()),
        "skipped_pairs": skips,
    }


def pose_loss(model, rgb, K, T, depth):
    tokens, h, w = model.jepa.encoder(rgb)
    pred = model.pose.depths(tokens, h, w, rgb.shape[-2:])
    valid = depth > 0
    loss = F.l1_loss(torch.log(pred[valid]), torch.log(depth[valid].clamp_min(0.05)))
    for i in range(len(rgb) - 1):
        R, d, length, c = model.pose.pair(tokens[i : i + 1], tokens[i + 1 : i + 2])
        gt = T[i + 1] @ torch.linalg.inv(T[i])
        norm = gt[:3, 3].norm().clamp_min(1e-5)
        loss = (
            loss
            + rotation_error(R, gt[None, :3, :3]).mean()
            + (1 - (d * gt[:3, 3] / norm).sum())
            + 0.2 * (length.log() - norm.log()).abs().mean()
            + 0.05 * F.binary_cross_entropy(c, torch.ones_like(c))
        )
    return loss


def rendering_loss(model, rgb, K, T, depth, predicted=False, use_vae=True):
    tokens, structure, appearance, pred_depth = model.features(rgb)
    if predicted:
        T, _ = predict_cameras(model, tokens)
        depth = pred_depth
    # Each supervised view is excluded from conditioning; no self-image pixel copying.
    target = int(torch.randint(len(rgb), (1,)))
    ids = [i for i in range(len(rgb)) if i != target]
    h, w = rgb.shape[-2:]
    xy = torch.stack(
        (
            torch.randint(w, (48,), device=rgb.device),
            torch.randint(h, (48,), device=rgb.device),
        ),
        -1,
    ).float()
    color, _, _ = render_rays(
        model,
        rgb[ids],
        K[ids],
        T[ids],
        structure[ids],
        appearance[ids],
        depth[ids],
        K[target],
        T[target],
        xy,
        samples=24,
        use_vae=use_vae,
    )
    truth = rgb[target, :, xy[:, 1].long(), xy[:, 0].long()].T
    return F.l1_loss(color, truth)


def train(
    data,
    output,
    steps=20,
    device="cpu",
    dim=192,
    patch=8,
    layers=2,
    resume=None,
    same_image=False,
    no_vae=False,
    single_scene=False,
):
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    torch.set_num_threads(4)
    root = Path(data)
    manifest = json.loads((root / "manifest.json").read_text())
    records = [s for s in manifest["scenes"] if s["split"] == "train"]
    validation = [s for s in manifest["scenes"] if s["split"] == "val"]
    if single_scene:
        records = records[:1]
    if not records or not validation:
        raise ValueError("Need independent train and validation scenes")
    if steps < 1:
        raise ValueError("steps must be positive")
    model = Pipeline(dim, patch, layers).to(device)
    history = []
    completed = {}
    state = None
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["config"] != model.config or state["manifest"] != manifest:
            raise ValueError("Resume config or dataset mismatch")
        expected = {
            "same_image": same_image,
            "no_vae": no_vae,
            "single_scene": single_scene,
        }
        if any(
            state.get("ablations", {}).get(k, False) != v for k, v in expected.items()
        ):
            raise ValueError("Resume ablation mismatch")
        model.load_state_dict(state["model"])
        history = state["history"]
        completed = state["completed"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        if device == "mps" and state.get("rng_mps") is not None:
            torch.mps.set_rng_state(state["rng_mps"].cpu())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    checkpoint = state
    for stage in STAGES:
        model.requires_grad_(False)
        modules = {
            "jepa": [
                model.jepa.encoder,
                model.jepa.predictor,
                model.jepa.query,
                model.jepa.out,
            ],
            "pose": [model.pose],
            "vae": [model.vae],
            "render_gt": [model.field],
            "render_pred": [model.field],
            "joint": [model.jepa.encoder, model.pose, model.vae, model.field],
        }[stage]
        for module in modules:
            module.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4 if stage == "joint" else 1e-3,
        )
        if state and state.get("stage") == stage:
            optimizer.load_state_dict(state["optimizer"])
        for step in range(completed.get(stage, 0), steps):
            scene = records[step % len(records)]
            rgb, K, T, depth = load_supervised(root / scene["path"], device)
            metrics = {}
            if stage == "jepa":
                loss, metrics = jepa_loss(model, rgb, K, T, depth, same_image)
            elif stage == "pose":
                loss = pose_loss(model, rgb, K, T, depth)
            elif stage == "vae":
                recon, mu, lv = model.vae(rgb)
                loss = (
                    F.l1_loss(recon, rgb)
                    + 1e-4 * (-0.5 * (1 + lv - mu.square() - lv.exp())).mean()
                )
            elif stage.startswith("render"):
                loss = rendering_loss(
                    model, rgb, K, T, depth, stage == "render_pred", not no_vae
                )
            else:
                jl, metrics = jepa_loss(model, rgb, K, T, depth, same_image)
                recon, mu, lv = model.vae(rgb)
                loss = (
                    jl
                    + 0.1 * pose_loss(model, rgb, K, T, depth)
                    + 0.2 * F.l1_loss(recon, rgb)
                    + 1e-4 * (-0.5 * (1 + lv - mu.square() - lv.exp())).mean()
                    + rendering_loss(model, rgb, K, T, depth, True, not no_vae)
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {stage}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            model.jepa.ema()
            completed[stage] = step + 1
            row = {
                "stage": stage,
                "step": step + 1,
                "loss": float(loss.detach()),
                **metrics,
            }
            history.append(row)
            checkpoint = {
                "version": 1,
                "config": model.config,
                "model": model.state_dict(),
                "completed": completed.copy(),
                "history": history,
                "stage": stage,
                "optimizer": optimizer.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_mps": torch.mps.get_rng_state() if device == "mps" else None,
                "ablations": {
                    "same_image": same_image,
                    "no_vae": no_vae,
                    "single_scene": single_scene,
                },
                "manifest": manifest,
            }
            torch.save(checkpoint, output / "latest.tmp")
            (output / "latest.tmp").replace(output / "latest.pt")
            if (step + 1) % 5 == 0 or step == 0:
                print(json.dumps(row), flush=True)
        with torch.no_grad():
            validation_losses = []
            for item in validation:
                rgb, K, T, depth = load_supervised(root / item["path"], device)
                validation_losses.append(float(pose_loss(model, rgb, K, T, depth)))
            history.append(
                {
                    "stage": stage,
                    "validation_pose_depth_loss": sum(validation_losses)
                    / len(validation_losses),
                    "validation_scenes": len(validation_losses),
                }
            )
    checkpoint["history"] = history
    checkpoint["elapsed_seconds"] = time.time() - started
    import resource

    checkpoint["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (output / "training-report.json").write_text(
        json.dumps(
            {
                "config": model.config,
                "completed": completed,
                "elapsed_seconds": checkpoint["elapsed_seconds"],
                "peak_rss_bytes": checkpoint["peak_rss_bytes"],
                "ablations": checkpoint["ablations"],
                "loss_weights": {
                    "jepa": 1,
                    "pose": 0.1,
                    "vae_reconstruction": 0.2,
                    "kl": 1e-4,
                    "render": 1,
                },
            },
            indent=2,
        )
    )
    torch.save(checkpoint, output / "model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    return output / "model.pt"
