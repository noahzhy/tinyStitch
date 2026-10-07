"""Regenerate analytic captures at native resolution without upscaling pixels."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from .data import shelf_scene, render, orthographic_gt


def generate_native(scene, output, width=1280, height=960, target_width=256):
    source = Path(scene)
    out = Path(output)
    if (out / "intrinsics.json").exists():
        raise ValueError("Use a fresh output directory")
    if min(width, height, target_width) < 16:
        raise ValueError("Invalid dimensions")
    manifest = json.loads((source.parent / "manifest.json").read_text())
    if manifest.get("renderer") != "analytic-ray-box-v1":
        raise ValueError("Requires an analytic synthetic source scene")
    if manifest.get("exposure_jitter", True):
        raise ValueError("Native controlled probe requires fixed source exposure")
    meta = json.loads((source / "intrinsics.json").read_text())
    poses = np.load(source / "labels/geometry.npz")["T"]
    K = np.asarray(meta["K"], np.float32).copy()
    sx, sy = width / meta["width"], height / meta["height"]
    K[:, 0, 0] *= sx
    K[:, 1, 1] *= sy
    K[:, 0, 2] = (K[:, 0, 2] + 0.5) * sx - 0.5
    K[:, 1, 2] = (K[:, 1, 2] + 0.5) * sy - 0.5
    scene_meta = (
        json.loads((source / "scene.json").read_text())
        if (source / "scene.json").exists()
        else {}
    )
    scene_width = (
        scene_meta.get("shelf_width")
        if manifest.get("capture_step") is not None
        else None
    )
    boxes, _, _ = shelf_scene(int(source.name), scene_width, scene_meta.get("layout"))
    material = manifest.get("material_style", "legacy_checker")
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    for i, T in enumerate(poses):
        image, depth = render(boxes, K[i], T, height, width, material_style=material)
        Image.fromarray(image).save(out / "rgb" / f"{i:04}.png")
        np.save(out / "labels" / f"{i:04}.npy", depth)
        if i % 10 == 0:
            print(f"Native capture {i+1}/{len(poses)}", flush=True)
    np.savez(out / "labels/geometry.npz", T=poses)
    (out / "intrinsics.json").write_text(
        json.dumps({"K": K.tolist(), "width": width, "height": height})
    )
    orthographic_gt(
        int(source.name),
        out / "orthographic",
        target_width,
        material_style=material,
        shelf_width=scene_width,
        layout=scene_meta.get("layout"),
    )
    (out.parent / "manifest.json").write_text(
        json.dumps(
            {
                "renderer": "analytic-ray-box-v1",
                "material_style": material,
                "exposure_jitter": False,
                "capture_step": manifest.get("capture_step"),
                "native_resolution": [width, height],
                "source_indices": list(range(len(poses))),
                "source_scene": str(source),
                "seed": int(source.name),
            },
            indent=2,
        )
    )
    if scene_meta:
        (out / "scene.json").write_text(json.dumps(scene_meta))
    return {
        "views": len(poses),
        "native_resolution": [width, height],
        "output": str(out),
    }
