"""Evaluate real RGB matches on generated scenes; generation labels are not passed to stitch()."""

import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.stitch import stitch, points

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--method", choices=["jepa", "sift"], default="jepa")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
records = []
for directory in sorted(args.input.iterdir()):
    if not (directory / "rgb").is_dir():
        continue
    record = {"seed": directory.name}
    try:
        report = stitch(
            sorted((directory / "rgb").glob("*.jpg")),
            args.output / directory.name,
            method=args.method,
        )
        # Independent simulation check: project shelf FRONT plane into each source camera,
        # then compare the predicted RGB-only mapping for those same physical landmarks.
        scene = json.loads((directory / "generation/scene.json").read_text())
        shelf = scene["shelf"]
        side = scene["side"]
        height = shelf["height"]
        c, s = np.cos(shelf["angle"]), np.sin(shelf["angle"])
        physical = []
        for u in np.linspace(-shelf["width"] * 0.45, shelf["width"] * 0.45, 7):
            for y in np.linspace(height * 0.2, height * 0.8, 4):
                v = side * shelf["depth"] * 0.451
                physical.append(
                    [shelf["x"] + u * c - v * s, y, shelf["z"] + u * s + v * c]
                )
        physical = np.array(physical)
        projected = []
        # Reconstruct projection independently from generator metadata, only after stitching.
        f = 960 / (2 * np.tan(np.deg2rad(54) / 2))
        for pose, H in zip(scene["poses"], report["transforms_work_to_panorama"]):
            center = np.array([pose["x"], pose["y"], pose["z"]])
            forward = np.array([pose["lookX"], pose["lookY"], pose["lookZ"]]) - center
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 1, 0])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            local = physical - center
            depth = local @ forward
            xy = np.c_[
                360 + f * (local @ right) / depth, 480 - f * (local @ up) / depth
            ]
            valid = (
                (xy[:, 0] > 0) & (xy[:, 0] < 720) & (xy[:, 1] > 0) & (xy[:, 1] < 960)
            )
            warped = points(np.array(H), xy)
            warped[~valid] = np.nan
            projected.append(warped)
        stacked = np.array(projected)
        median = np.nanmedian(stacked, axis=0)
        reproj = np.linalg.norm(stacked - median[None, :, :], axis=-1)
        record.update(
            status="completed",
            seconds=report["seconds"],
            size=report["size"],
            median_match_residual_px=report["alignment"]["after_median_px"],
            independent_front_plane_error_px=float(np.nanmedian(reproj)),
            independent_front_plane_error_p95_px=float(np.nanpercentile(reproj, 95)),
            input_digest=report["input_digest"],
        )
    except Exception as e:
        record.update(status="failed", error=str(e))
    records.append(record)
    print(json.dumps(record), flush=True)
report = {
    "hardware": platform.platform(),
    "cpu": "Apple M4 (local test)",
    "rss_at_completion_bytes": psutil.Process().memory_info().rss,
    "measurement_scope": "serial CPU process RSS at completion; not strict peak. Geometry labels used ONLY by evaluation after RGB-only stitching.",
    "method": args.method,
    "records": records,
    "completed": sum(r["status"] == "completed" for r in records),
    "total": len(records),
}
(args.output / "benchmark.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
