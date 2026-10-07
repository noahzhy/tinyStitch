"""Existing planar estimators on the same orthographic evaluation canvas.

Homographies use source RGB only. A declared front-plane adapter uses predicted
anchor depth and the same GT target-camera/gauge used by the NVS benchmark;
it does not fit a transform against target RGB. Native mosaics are also saved.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from backend.stitch import pair_match, graph_transforms, refine, blend
from .data import load_input
from .metrics import image_metrics


def plane_adapter(K, target_K, target_T, depth, height, width):
    xy = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    q = np.c_[xy, np.ones(4)] @ np.linalg.inv(target_K).T
    q[:, 2] = 0
    origins = (q - target_T[:3, 3]) @ target_T[:3, :3]
    direction = target_T[2, :3]
    if abs(direction[2]) < 0.1 or depth <= 0:
        raise ValueError("Invalid front-plane evaluation adapter")
    distance = (depth - origins[:, 2]) / direction[2]
    if (distance <= 0).any():
        raise ValueError("Front plane lies behind target rays")
    world = origins + distance[:, None] * direction
    pixels = world @ K.T
    uv = (pixels[:, :2] / pixels[:, 2:]).astype(np.float32)
    return cv2.getPerspectiveTransform(uv, xy)


def run(scene, reference, output, methods=("sift", "jepa")):
    torch.set_num_threads(4)
    cv2.setRNGSeed(17)
    root, ref, out = Path(scene), Path(reference), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    reference_report = json.loads((ref / "report.json").read_text())
    ids = reference_report["input_indices"]
    rgb, K = load_input(root)
    images = [
        (rgb[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1].copy()
        for i in ids
    ]
    digest = hashlib.sha256(
        rgb[ids].numpy().tobytes() + K[ids].numpy().tobytes()
    ).hexdigest()
    if digest != reference_report["input_sha256"]:
        raise ValueError("Baseline input hash differs from neural reconstruction")
    n, anchor = len(images), len(images) // 2
    reports = []
    for method in methods:
        start = time.perf_counter()
        folder = out / method
        folder.mkdir(exist_ok=True)
        diagnostics = []
        try:
            info = None
            if method == "jepa":
                from backend.jepa import JEPADescriptor

                extractor = JEPADescriptor(device="cpu")
                features = [extractor.extract(im) for im in images]
                info = extractor.info
            else:
                detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.025)
                features = [
                    detector.detectAndCompute(
                        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None
                    )
                    for im in images
                ]
            edges = []
            for i in range(n):
                for j in range(i + 1, min(n, i + 4)):
                    diagnostic, edge = pair_match(
                        i, j, images, features, ratio=0.82 if method == "jepa" else 0.76
                    )
                    diagnostics.append(diagnostic)
                    if edge is not None:
                        edges.append(edge)
            transforms, anchor = graph_transforms(n, edges)
            transforms, alignment = refine(transforms, anchor, edges)
            mosaic, alpha, final_h, exposure = blend(
                images, transforms, lambda *_: None
            )
            cv2.imwrite(str(folder / "native-mosaic.png"), np.dstack((mosaic, alpha)))
            # Labels enter only after source-only homography estimation is complete.
            raw = np.load(root / "labels/geometry.npz")["T"][ids]
            anchor_depth = np.load(root / "labels" / f"{ids[anchor]:04}.npy")
            scale = np.median(anchor_depth[anchor_depth > 0])
            target = np.load(root / "orthographic/geometry.npz")
            target_T = target["T"] @ np.linalg.inv(raw[anchor])
            target_T[:3, 3] /= scale
            target_K = target["K"].copy()
            target_K[0, 0] *= scale
            target_K[1, 1] *= scale
            gt = np.asarray(Image.open(ref / "gt.png").convert("RGB"))
            height, width = gt.shape[:2]
            predicted_depth = np.load(ref / "depth.npy")[anchor]
            plane = float(np.median(predicted_depth[predicted_depth > 0]))
            adapter = plane_adapter(
                K[ids[anchor]].numpy(), target_K, target_T, plane, height, width
            )
            transform = adapter @ np.linalg.inv(final_h[anchor])
            prediction = cv2.warpPerspective(mosaic, transform, (width, height))
            valid = cv2.warpPerspective(
                alpha, transform, (width, height), flags=cv2.INTER_NEAREST
            )
            prediction[valid == 0] = 0
            cv2.imwrite(str(folder / "orthographic-adapted-rgb.png"), prediction)
            cv2.imwrite(
                str(folder / "orthographic-adapted.png"), np.dstack((prediction, valid))
            )
            score = image_metrics(prediction[..., ::-1], gt, target["depth"])
            new_png = ref / (
                "predicted_selected-rgb.png"
                if (ref / "predicted_selected-rgb.png").exists()
                else "predicted_all-rgb.png"
            )
            neural_score = image_metrics(
                np.asarray(Image.open(new_png).convert("RGB")), gt, target["depth"]
            )
            result = dict(
                status="ok",
                metrics=score,
                neural_metrics_same_png_protocol=neural_score,
                supported_fraction_gt_valid=float(
                    (valid[target["depth"] > 0] > 0).mean()
                ),
                accepted_edges=len(edges),
                alignment=alignment,
                plane_depth_predicted=plane,
                mosaic_to_target=transform.tolist(),
                transforms_source_to_mosaic=[h.tolist() for h in final_h],
                exposure=exposure,
            )
        except (ValueError, RuntimeError, cv2.error) as error:
            result = dict(status="failed", error=str(error))
        result.update(
            method=method,
            input_count=n,
            input_sha256=digest,
            neural_reference=str(ref),
            model=info,
            elapsed_seconds=time.perf_counter() - start,
            pairs=diagnostics,
            homography_gt_used_for_fitting=False,
            target_rgb_used_for_alignment=False,
            output_adapter="one anchor-front plane with predicted source depth; shared GT target camera and normalized scoring gauge",
            baseline="existing backend.stitch source matching, pose graph, plane refinement, nearest-center narrow feather; research wrapper accepts 60 sources",
        )
        (folder / "report.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        reports.append(result)
        print(
            json.dumps(
                dict(
                    method=method,
                    status=result["status"],
                    metrics=result.get("metrics"),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
    (out / "summary.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=["sift", "jepa"], default=["sift", "jepa"]
    )
    run(**vars(parser.parse_args()))
