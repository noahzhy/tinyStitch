"""Frozen-checkpoint 10–60 view scaling probe, with two-stage diagnostics."""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import load_input
from .geometry import canonicalize, rotation_error
from .inference import load_model
from .optimization import predict_cameras
from .render import render_image


def indices(count, total):
    if total != 60 or count not in (10, 30, 60):
        raise ValueError(
            "This controlled probe requires 60 source views and counts 10,30,60"
        )
    if count == 60:
        return np.arange(60)
    if count == 30:
        return np.arange(0, 60, 2)
    return np.array([0, 8, 14, 20, 28, 30, 36, 44, 52, 58])


def single(
    scene,
    checkpoint,
    output,
    count,
    device="cpu",
    width=128,
    samples=32,
    geometry_width=None,
    color_residual_limit=None,
    source_limit=None,
    color_fusion=None,
    local_sources=None,
    oracle_control=False,
    alignment=None,
):
    torch.set_num_threads(4)
    start = time.perf_counter()
    out = Path(output) / str(count)
    out.mkdir(parents=True, exist_ok=True)
    rgb, K = load_input(scene, device)
    if len(rgb) == 60:
        selected = indices(count, len(rgb))
    elif len(rgb) == count:
        selected = np.arange(count)
    else:
        raise ValueError("Unsupported source count")
    rgb, K = rgb[selected], K[selected]
    model, state = load_model(checkpoint, device)
    model.requires_grad_(False)
    if color_fusion is not None:
        if color_fusion not in ("weighted", "consensus", "nearest"):
            raise ValueError("Unknown fusion mode")
        model.field.color_fusion = color_fusion
    if color_residual_limit is not None:
        if not 0 <= color_residual_limit <= 1:
            raise ValueError("Invalid color residual limit")
        model.field.color_residual_limit = color_residual_limit

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    sync()
    feature_start = time.perf_counter()
    with torch.no_grad():
        if geometry_width:
            height = round(geometry_width * rgb.shape[-2] / rgb.shape[-1])
            tokens, structure, appearance, depth = model.features_multires(
                rgb, (height, geometry_width), 1
            )
        else:
            tokens, structure, appearance, depth = model.features_batched(rgb, 4)
    sync()
    feature_seconds = time.perf_counter() - feature_start
    pose_start = time.perf_counter()
    with torch.no_grad():
        pred, edges = predict_cameras(
            model, tokens, pair_batch=2 if tokens.shape[1] > 768 else 8
        )
    sync()
    pose_seconds = time.perf_counter() - pose_start
    alignment_diagnostics = None
    if alignment:
        from .alignment_pose import lift

        if count != 60:
            raise ValueError(
                "Alignment override currently requires all 60 ordered inputs"
            )
        calibration = state.get("reference_depth_calibration")
        camera_depth = depth
        if calibration:
            import copy

            head = copy.deepcopy(model.pose).eval().requires_grad_(False)
            head.load_state_dict(calibration["weights"], strict=False)
            gh, gw = structure.shape[-2:]
            with torch.no_grad():
                camera_depth = torch.cat(
                    [
                        head.depths(tokens[i : i + 4], gh, gw, depth.shape[-2:])
                        for i in range(0, len(tokens), 4)
                    ]
                )
        pred, alignment_diagnostics = lift(alignment, rgb, K, camera_depth)
        if calibration:
            alignment_diagnostics["frozen_reference_scale"] = True
            alignment_diagnostics["calibration_checkpoint_sha256"] = calibration[
                "checkpoint_sha256"
            ]
            alignment_diagnostics["calibration_protocol"] = calibration["protocol"]
        edges = [(i, j, pred[j] @ torch.linalg.inv(pred[i]), c) for i, j, e, c in edges]
    # GT labels enter evaluation only, after predictions have been computed.
    raw = np.load(Path(scene) / "labels/geometry.npz")["T"][selected]
    raw_depth = np.stack(
        [np.load(Path(scene) / "labels" / f"{i:04}.npy") for i in selected]
    )
    gt, gt_depth, scale = canonicalize(
        torch.tensor(raw, device=device), torch.tensor(raw_depth, device=device)
    )
    relrot = []
    reldir = []
    relbaseline = []
    for i, j, e, c in edges:
        actual = gt[j] @ torch.linalg.inv(gt[i])
        a, b = e[:3, 3], actual[:3, 3]
        relrot.append(float(rotation_error(e[:3, :3], actual[:3, :3]) * 180 / torch.pi))
        reldir.append(
            float(
                torch.atan2(torch.linalg.cross(a, b).norm(), torch.dot(a, b))
                * 180
                / torch.pi
            )
        )
        relbaseline.append(
            float((a.norm() - b.norm()).abs() / b.norm().clamp_min(1e-8))
        )
    centers = lambda t: -(t[:, :3, :3].transpose(1, 2) @ t[:, :3, 3, None])[:, :, 0]
    pc, gc = centers(pred), centers(gt)
    ce = (pc - gc).norm(dim=-1)
    valid = gt_depth > 0
    stage1 = {
        "edge_count": len(edges),
        "relative_rotation_mean_deg": float(np.mean(relrot)),
        "relative_translation_direction_mean_deg": float(np.mean(reldir)),
        "relative_baseline_error_mean": float(np.mean(relbaseline)),
        "camera_center_l2_mean_canonical": float(ce.mean()),
        "camera_center_l2_max_canonical": float(ce.max()),
        "sequence_extent_ratio": float(
            (pc[-1] - pc[0]).norm() / (gc[-1] - gc[0]).norm()
        ),
        "global_rotation_max_deg": float(
            rotation_error(pred[:, :3, :3], gt[:, :3, :3]).max() * 180 / torch.pi
        ),
        "source_depth_abs_rel": float(
            ((depth[valid] - gt_depth[valid]).abs() / gt_depth[valid]).mean()
        ),
        "confidence_mean": float(torch.stack([c for i, j, e, c in edges]).mean()),
        "feature_seconds": feature_seconds,
        "pose_seconds": pose_seconds,
    }
    np.save(out / "cameras.npy", pred.cpu().numpy())
    np.save(out / "depth.npy", depth.cpu().numpy())
    interim = {"count": count, "status": "rendering", "stage1": stage1}
    (out / "progress.json").write_text(json.dumps(interim, indent=2))
    print(json.dumps(interim), flush=True)
    target = np.load(Path(scene) / "orthographic/geometry.npz")
    target_T = torch.tensor(target["T"] @ np.linalg.inv(raw[count // 2]), device=device)
    target_T[:3, 3] /= scale
    target_K = torch.tensor(target["K"], device=device)
    target_K[0, 0] *= scale
    target_K[1, 1] *= scale
    truth = np.array(Image.open(Path(scene) / "orthographic/rgb.png").convert("RGB"))
    h, w = truth.shape[:2]
    if width != w:
        raise ValueError("Evaluation output must match synthetic orthographic GT width")
    Image.fromarray(truth).save(out / "gt.png")
    truth_tensor = torch.tensor(truth, device=device).float() / 255
    mask = torch.tensor(target["depth"] > 0, device=device)
    if source_limit is not None and source_limit < 2:
        raise ValueError("Need at least two rendering sources")
    if local_sources is not None and source_limit is not None:
        raise ValueError("Choose global source limit or local sources")
    source_ids = (
        np.arange(count)
        if source_limit is None
        else np.linspace(0, count - 1, min(count, source_limit)).round().astype(int)
    )
    variants = [
        (
            "predicted_all" if source_limit is None else "predicted_selected",
            source_ids,
            pred,
            depth,
        )
    ]
    if count == 60 and source_limit is None and local_sources is None:
        subset = np.linspace(0, count - 1, 8).round().astype(int)
        variants += [
            ("predicted_8", subset, pred, depth),
            ("oracle_geometry_8", subset, gt, gt_depth),
        ]
    if oracle_control:
        variants.append(("oracle_geometry_matched", source_ids, gt, gt_depth))
    rows = []
    for name, ids, T, d in variants:
        sync()
        t0 = time.perf_counter()
        color, support = render_image(
            model,
            rgb[ids],
            K[ids],
            T[ids],
            structure[ids],
            appearance[ids],
            d[ids],
            target_K,
            target_T,
            h,
            w,
            samples=samples,
            projection="orthographic",
            local_sources=local_sources,
        )
        sync()
        elapsed = time.perf_counter() - t0
        mse = (color[mask] - truth_tensor[mask]).square().mean()
        row = {
            "mode": name,
            "source_count": len(ids),
            "psnr_gt_valid": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "supported_fraction_gt_valid": float(support[mask].float().mean()),
            "coverage_whole_image": float(support.float().mean()),
            "render_seconds": elapsed,
        }
        rows.append(row)
        color_np = (color.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(color_np).save(out / f"{name}-rgb.png")
        Image.fromarray(
            np.dstack((color_np, support.cpu().numpy().astype(np.uint8) * 255))
        ).save(out / f"{name}.png")
        print(json.dumps({"count": count, "stage2": row}), flush=True)
        (out / "progress.json").write_text(
            json.dumps({"count": count, "stage1": stage1, "stage2": rows}, indent=2)
        )
    report = {
        "count": count,
        "status": "ok",
        "input_indices": selected.tolist(),
        "anchor_source_index": int(selected[count // 2]),
        "scene_seed": int(Path(scene).name),
        "source_resolution": [rgb.shape[-1], rgb.shape[-2]],
        "geometry_working_width": geometry_width or rgb.shape[-1],
        "native_rgb_used_for_rendering": True,
        "output_resolution": [w, h],
        "alignment_pose": alignment_diagnostics,
        "stage1": stage1,
        "stage2": rows,
        "device": device,
        "samples": samples,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(
            rgb.cpu().numpy().tobytes() + K.cpu().numpy().tobytes()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "weights_frozen": True,
        "training_input_count": 60 if state.get("pose_adaptation") else 3,
        "pose_training_views_per_scene": 60 if state.get("pose_adaptation") else 3,
        "pose_adaptation": state.get("pose_adaptation"),
        "depth_adaptation": state.get("depth_adaptation"),
        "renderer_adaptation": state.get("renderer_adaptation"),
        "appearance_geometry": model.config.get("appearance_geometry", False),
        "feature_center_mapping": model.config.get("feature_center_mapping", False),
        "render_source_limit": source_limit,
        "local_sources_per_column_chunk": local_sources,
        "color_fusion": getattr(model.field, "color_fusion", "weighted"),
        "renderer_training_input_count": (
            None if state.get("renderer_adaptation") else 3
        ),
        "renderer_training_source_counts": state.get("renderer_adaptation", {}).get(
            "source_counts", [3]
        ),
        "renderer_training_scene_seeds": state.get("renderer_adaptation", {}).get(
            "train_seeds", [731000]
        ),
        "test_type": "matched-target synthetic evaluation; module training scenes and source counts are recorded separately",
        "gt_used_for_test_fitting": False,
        "target_camera": "GT supplied for matched evaluation only",
        "scale": "GT-normalized evaluation gauge, not metric inference",
        "global_optimization": bool(model.config.get("pose_graph", False)),
        "depth_photometric_optimization": False,
        "global_sequence_features": model.config.get("global_pose", False),
        "global_pose_adaptation": state.get("global_pose_adaptation"),
        "color_residual_limit": getattr(model.field, "color_residual_limit", None),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def run(
    scene,
    checkpoint,
    output,
    device="cpu",
    width=128,
    samples=32,
    counts=(10, 30, 60),
    geometry_width=None,
    color_residual_limit=None,
    source_limit=None,
    color_fusion=None,
    local_sources=None,
    oracle_control=False,
    alignment=None,
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for count in counts:
        command = [
            sys.executable,
            "-m",
            "backend.research.multiview",
            "--scene",
            scene,
            "--checkpoint",
            checkpoint,
            "--output",
            output,
            "--device",
            device,
            "--width",
            str(width),
            "--samples",
            str(samples),
            "--single-count",
            str(count),
        ]
        if alignment:
            command += ["--alignment", alignment]
        if oracle_control:
            command += ["--oracle-control"]
        if geometry_width is not None:
            command += ["--geometry-width", str(geometry_width)]
        if color_fusion is not None:
            command += ["--color-fusion", color_fusion]
        if local_sources is not None:
            command += ["--local-sources", str(local_sources)]
        if source_limit is not None:
            command += ["--source-limit", str(source_limit)]
        if color_residual_limit is not None:
            command += ["--color-residual-limit", str(color_residual_limit)]
        with (out / f"{count}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            row = json.loads((out / str(count) / "report.json").read_text())
        else:
            row = {
                "count": count,
                "status": "failed",
                "exit_code": result.returncode,
                "log": str(out / f"{count}.log"),
            }
        summary.append(row)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"count": count, "status": row["status"]}), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--geometry-width", type=int)
    p.add_argument("--color-residual-limit", type=float)
    p.add_argument("--source-limit", type=int)
    p.add_argument("--local-sources", type=int)
    p.add_argument("--color-fusion", choices=["weighted", "consensus", "nearest"])
    p.add_argument("--alignment")
    p.add_argument("--oracle-control", action="store_true")
    p.add_argument("--single-count", type=int)
    a = vars(p.parse_args())
    count = a.pop("single_count")
    if count:
        single(**a, count=count)
    else:
        run(**a)
