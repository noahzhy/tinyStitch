"""Global feature pose refinement, train/val only; no target RGB or test fit."""

import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from .data import load_supervised
from .inference import load_model
from .model import Pipeline
from .optimization import predict_cameras
from .poseadapt import metrics


def run(data, checkpoint, output, steps=1600, seed=20261009):
    if steps < 1:
        raise ValueError("Invalid training budget")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    root, out = Path(data), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        raise ValueError("Use a fresh output directory")
    start = time.perf_counter()
    old, state = load_model(checkpoint)
    if old.config.get("global_pose"):
        raise ValueError("Expected local pose checkpoint")
    model = Pipeline(**dict(old.config, global_pose=True))
    model.load_state_dict(old.state_dict(), strict=False)
    model.eval().requires_grad_(False)
    head = model.pose.global_sequence
    head.requires_grad_(True)
    cached = []
    entries = json.loads((root / "manifest.json").read_text())["scenes"]
    for entry in entries:
        if entry["split"] == "test":
            continue
        rgb, K, gt, depth = load_supervised(root / entry["path"])
        with torch.no_grad():
            tokens = torch.cat([old.jepa.encoder(image[None])[0] for image in rgb])
            base, _ = predict_cameras(old, tokens)
            stats = head.statistics(tokens)
        cached.append(dict(entry=entry, statistics=stats, base=base, gt=gt))
        print(
            json.dumps({"cached": entry["seed"], "split": entry["split"]}), flush=True
        )
    train = [c for c in cached if c["entry"]["split"] == "train"]
    val = [c for c in cached if c["entry"]["split"] == "val"]
    if not train or not val:
        raise ValueError("Need disjoint train/validation")

    def window(c, begin, count):
        base, gt = c["base"][begin : begin + count], c["gt"][begin : begin + count]
        return (
            c["statistics"][begin : begin + count],
            base @ torch.linalg.inv(base[count // 2]),
            gt @ torch.linalg.inv(gt[count // 2]),
        )

    def validation():
        rows = []
        with torch.no_grad():
            for c in val:
                for count in (10, 30, 60):
                    stats, base, gt = window(c, (len(c["base"]) - count) // 2, count)
                    pred = head(stats, base)
                    rows.append(
                        dict(
                            count=count,
                            before=metrics(base, gt),
                            after=metrics(pred, gt),
                        )
                    )
        score = sum(
            r["after"]["center_l2_mean"] + 0.01 * r["after"]["rotation_max_deg"]
            for r in rows
        ) / len(rows)
        return score, rows

    initial, initial_rows = validation()
    best, best_step, best_weights = initial, 0, copy.deepcopy(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0002, weight_decay=0.001)
    history = []
    for step in range(1, steps + 1):
        c = train[int(torch.randint(len(train), (1,)))]
        count = (10, 30, 60)[int(torch.randint(3, (1,)))]
        begin = int(torch.randint(len(c["base"]) - count + 1, (1,)))
        stats, base, gt = window(c, begin, count)
        pred = head(stats, base)
        centers = lambda t: -(t[:, :3, :3].transpose(1, 2) @ t[:, :3, 3, None])[:, :, 0]
        error = centers(pred) - centers(gt)
        delta = centers(pred) - centers(base)
        loss = (
            100 * error.square().mean()
            + 100 * (pred[:, :3, :3] - gt[:, :3, :3]).square().mean()
            + 0.1 * delta.square().mean()
            + 0.1 * delta.diff(dim=0).square().mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("Global refinement diverged")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 2)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            score, rows = validation()
            if score < best:
                best, best_step, best_weights = (
                    score,
                    step,
                    copy.deepcopy(head.state_dict()),
                )
            row = dict(
                step=step,
                loss=float(loss.detach()),
                validation_score=score,
                selected_step=best_step,
            )
            history.append(row)
            print(json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
    head.load_state_dict(best_weights)
    score, rows = validation()
    protocol = dict(
        seed=seed,
        steps=steps,
        selected_step=best_step,
        initial_validation_score=initial,
        selected_validation_score=score,
        initial_validation=initial_rows,
        selected_validation=rows,
        train_seeds=[c["entry"]["seed"] for c in train],
        validation_seeds=[c["entry"]["seed"] for c in val],
        test_loaded=False,
        target_view_loaded=False,
        trainable="pose.global_sequence",
        representation="per-view JEPA mean/std + local camera pose + sequence position, two-layer global attention",
        training_windows=[10, 30, 60],
        window_policy="contiguous sections; not sparse full-span subsets",
        loss="100 center MSE + 100 rotation matrix MSE + .1 correction magnitude + .1 correction differences",
        selection="mean validation center error + .01 max rotation degrees",
        source_checkpoint_sha256=hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
        elapsed_seconds=time.perf_counter() - start,
    )
    saved = copy.deepcopy(state)
    saved.update(
        model=model.state_dict(),
        config=model.config,
        global_pose_adaptation=protocol,
        global_pose_rng_state=torch.get_rng_state(),
    )
    torch.save(saved, out / "model.pt")
    (out / "report.json").write_text(json.dumps(protocol, indent=2))
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_step": best_step,
                "validation_score": score,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--seed", type=int, default=20261009)
    run(**vars(p.parse_args()))
