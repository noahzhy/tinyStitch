"""Stage-one evaluation on different shelf lengths; no target views loaded."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .data import load_input, load_supervised
from .inference import load_model
from .optimization import predict_cameras
from .poseadapt import metrics


def run(data_roots, checkpoints, output):
    torch.set_num_threads(4)
    rows = []
    for checkpoint in checkpoints:
        model, state = load_model(checkpoint)
        model.requires_grad_(False)
        for root in data_roots:
            root = Path(root)
            manifest = json.loads((root / "manifest.json").read_text())
            for entry in manifest["scenes"]:
                if entry["split"] != "test":
                    continue
                path = root / entry["path"]
                start = time.perf_counter()
                rgb, K = load_input(path)
                with torch.no_grad():
                    tokens, structure, appearance, depth = model.features_batched(
                        rgb, 4
                    )
                    predicted, edges = predict_cameras(model, tokens)
                # Source labels first enter scoring after RGB-only inference.
                _, _, truth, gt_depth = load_supervised(path)
                valid = gt_depth > 0
                row = dict(
                    checkpoint=checkpoint,
                    checkpoint_sha256=hashlib.sha256(
                        Path(checkpoint).read_bytes()
                    ).hexdigest(),
                    scene=str(path),
                    seed=entry["seed"],
                    input_count=len(rgb),
                    source_resolution=list(rgb.shape[-2:][::-1]),
                    scene_settings=json.loads((path / "scene.json").read_text()),
                    pose=metrics(predicted, truth),
                    depth_abs_rel=float(
                        (
                            (depth[valid] - gt_depth[valid]).abs() / gt_depth[valid]
                        ).mean()
                    ),
                    pose_graph=model.config.get("pose_graph", False),
                    gt_used_for_fitting=False,
                    target_view_loaded=False,
                    elapsed_seconds=time.perf_counter() - start,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))
