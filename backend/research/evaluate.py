"""Evaluation-only labels, including held-out camera; never passed to reconstruction inputs."""

import argparse, json, time, resource
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from .data import load_supervised, load_input
from .inference import load_model, reconstruct
from .optimization import predict_cameras
from .geometry import rotation_error
from .render import render_image


def evaluate(
    data, checkpoint, output, device="cpu", optimize=5, samples=24, modes=None
):
    torch.set_num_threads(4)
    root = Path(data)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    # LPIPS may require a one-time official pretrained AlexNet download.
    try:
        import lpips

        perceptual = lpips.LPIPS(net="alex").eval()
    except Exception as e:
        perceptual = None
        lpips_error = str(e)
    for item in manifest["scenes"]:
        if item["split"] != "test":
            continue
        path = root / item["path"]
        rgb, K, truth_T, truth_depth = load_supervised(path, device)
        anchor = len(rgb) // 2
        held = np.load(path / "heldout" / "geometry.npz")
        original = np.load(path / "labels" / "geometry.npz")["T"]
        raw_depth = np.load(path / "labels" / f"{anchor:04}.npy")
        scale = np.median(raw_depth[raw_depth > 0])
        target_T = torch.tensor(
            held["T"] @ np.linalg.inv(original[anchor]), device=device
        )
        target_T[:3, 3] /= scale
        target_K = torch.tensor(held["K"], device=device)
        h, w = held["depth"].shape
        for mode in modes or [
            "feedforward",
            "optimized",
            "no_vae",
            "no_global",
            "oracle_cameras",
        ]:
            torch.manual_seed(17)
            dest = out / item["path"] / mode
            started = time.time()
            try:
                if mode == "oracle_cameras":
                    model, _ = load_model(checkpoint, device)
                    with torch.no_grad():
                        _, structure, appearance, predicted_depth = model.features(rgb)
                        color, mask = render_image(
                            model,
                            rgb,
                            K,
                            truth_T,
                            structure,
                            appearance,
                            predicted_depth,
                            target_K,
                            target_T,
                            h,
                            w,
                            samples=samples,
                        )
                    dest.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(
                        np.dstack(
                            (
                                (color.cpu().numpy() * 255).astype(np.uint8),
                                mask.cpu().numpy().astype(np.uint8) * 255,
                            )
                        )
                    ).save(dest / "panorama.png")
                    pred_T = truth_T
                    pred_depth = predicted_depth
                else:
                    report = reconstruct(
                        path,
                        checkpoint,
                        dest,
                        device,
                        optimize=optimize if mode != "feedforward" else 0,
                        width=w,
                        samples=samples,
                        use_vae=mode != "no_vae",
                        global_opt=mode not in ["feedforward", "no_global"],
                        target=(target_K, target_T, h, w),
                        projection="perspective",
                    )
                    pred_T = torch.tensor(np.load(dest / "cameras.npy"), device=device)
                    pred_depth = torch.tensor(
                        np.load(dest / "depth.npy"), device=device
                    )
                image = np.asarray(Image.open(dest / "panorama.png"))
                valid = (image[..., 3] > 0) & (held["depth"] > 0)
                target = (
                    np.asarray(Image.open(path / "heldout" / "rgb.png")).astype(float)
                    / 255
                )
                prediction = image[..., :3].astype(float) / 255
                mse = (
                    np.mean((prediction[valid] - target[valid]) ** 2)
                    if valid.any()
                    else None
                )
                ssim_map = structural_similarity(
                    prediction, target, channel_axis=2, data_range=1, full=True
                )[1]
                ssim = float(ssim_map[valid].mean()) if valid.any() else None
                lp = None
                if perceptual is not None and valid.any():
                    # Full masked frame; coverage reported separately, not claimed as complete-frame quality.
                    a = (
                        torch.tensor(prediction * valid[..., None])
                        .permute(2, 0, 1)[None]
                        .float()
                        * 2
                        - 1
                    )
                    b = (
                        torch.tensor(target * valid[..., None])
                        .permute(2, 0, 1)[None]
                        .float()
                        * 2
                        - 1
                    )
                    with torch.no_grad():
                        lp = float(perceptual(a, b))
                ratios = truth_depth[truth_depth > 0] / pred_depth[truth_depth > 0]
                aligned = pred_depth * ratios.median()
                rot = float(
                    rotation_error(pred_T[:, :3, :3], truth_T[:, :3, :3]).mean()
                    * 180
                    / np.pi
                )
                row = {
                    "seed": item["seed"],
                    "mode": mode,
                    "status": "ok" if valid.any() else "failed_no_coverage",
                    "coverage": float(valid.mean()),
                    "psnr": (
                        float(-10 * np.log10(max(mse, 1e-10)))
                        if mse is not None
                        else None
                    ),
                    "ssim": ssim,
                    "lpips_masked": lp,
                    "rotation_degrees": rot,
                    "translation_canonical_l1": float(
                        (pred_T[:, :3, 3] - truth_T[:, :3, 3]).abs().mean()
                    ),
                    "depth_aligned_l1": float(
                        (aligned[truth_depth > 0] - truth_depth[truth_depth > 0])
                        .abs()
                        .mean()
                    ),
                    "elapsed_seconds": time.time() - started,
                }
                if perceptual is None:
                    row["lpips_unavailable"] = lpips_error
                # Side-by-side and central crop for actual visual review.
                comparison = np.concatenate(
                    (image[..., :3], (target * 255).astype(np.uint8)), 1
                )
                Image.fromarray(comparison).save(dest / "comparison.png")
                Image.fromarray(comparison[h // 4 : 3 * h // 4]).save(dest / "crop.png")
            except Exception as e:
                row = {
                    "seed": item["seed"],
                    "mode": mode,
                    "status": "failed",
                    "error": str(e),
                    "elapsed_seconds": time.time() - started,
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
    # Render planar mosaics into the same target camera using an explicit anchor-depth plane.
    # This deliberately retains the baseline's plane assumption; labels define evaluation only.
    from backend.stitch import stitch
    import cv2

    for item in manifest["scenes"]:
        if item["split"] != "test":
            continue
        path = root / item["path"]
        rgb, K, truth_T, truth_depth = load_supervised(path, "cpu")
        anchor = len(rgb) // 2
        held = np.load(path / "heldout" / "geometry.npz")
        original = np.load(path / "labels" / "geometry.npz")["T"]
        raw_depth = np.load(path / "labels" / f"{anchor:04}.npy")
        scale = np.median(raw_depth[raw_depth > 0])
        target_T = held["T"] @ np.linalg.inv(original[anchor])
        target_T[:3, 3] /= scale
        h, w = held["depth"].shape
        plane_depth = float(truth_depth[anchor][truth_depth[anchor] > 0].median())
        normal = np.array([0.0, 0.0, 1.0])
        projection = (
            held["K"]
            @ (target_T[:3, :3] + np.outer(target_T[:3, 3], normal) / plane_depth)
            @ np.linalg.inv(K[anchor].numpy())
        )
        target = (
            np.asarray(Image.open(path / "heldout" / "rgb.png")).astype(float) / 255
        )
        for method in ["sift", "jepa"]:
            started = time.time()
            dest = out / item["path"] / method
            try:
                result = stitch(
                    sorted((path / "rgb").glob("*.png")),
                    dest,
                    method=method,
                    device="cpu",
                )
                mosaic = cv2.imread(str(dest / "panorama.png"), cv2.IMREAD_UNCHANGED)
                H = np.asarray(result["transforms_work_to_panorama"][anchor])
                warped = cv2.warpPerspective(
                    mosaic, projection @ np.linalg.inv(H), (w, h)
                )
                cv2.imwrite(str(dest / "target.png"), warped)
                prediction = warped[:, :, :3][:, :, ::-1].astype(float) / 255
                valid = (warped[:, :, 3] > 0) & (held["depth"] > 0)
                mse = (
                    np.mean((prediction[valid] - target[valid]) ** 2)
                    if valid.any()
                    else None
                )
                ss = structural_similarity(
                    prediction, target, channel_axis=2, data_range=1, full=True
                )[1]
                lp = None
                if perceptual is not None and valid.any():
                    a = (
                        torch.tensor(prediction * valid[..., None])
                        .permute(2, 0, 1)[None]
                        .float()
                        * 2
                        - 1
                    )
                    b = (
                        torch.tensor(target * valid[..., None])
                        .permute(2, 0, 1)[None]
                        .float()
                        * 2
                        - 1
                    )
                    with torch.no_grad():
                        lp = float(perceptual(a, b))
                rows.append(
                    {
                        "seed": item["seed"],
                        "mode": method,
                        "status": "ok" if valid.any() else "failed_no_coverage",
                        "coverage": float(valid.mean()),
                        "psnr": (
                            float(-10 * np.log10(max(mse, 1e-10)))
                            if mse is not None
                            else None
                        ),
                        "ssim": float(ss[valid].mean()) if valid.any() else None,
                        "lpips_masked": lp,
                        "elapsed_seconds": time.time() - started,
                        "projection": "anchor median-depth plane; evaluation-only ground-truth camera",
                    }
                )
                Image.fromarray(
                    np.concatenate(
                        (
                            (prediction * 255).astype(np.uint8),
                            (target * 255).astype(np.uint8),
                        ),
                        1,
                    )
                ).save(dest / "comparison.png")
            except Exception as e:
                rows.append(
                    {
                        "seed": item["seed"],
                        "mode": method,
                        "status": "failed",
                        "error": str(e),
                    }
                )
    report = {
        "results": rows,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": [
            "Small synthetic research run, not evidence of real-store accuracy",
            "Planar baseline uses a median-depth evaluation plane",
            "Same-image versus cross-image requires separately trained checkpoints",
        ],
        "failure_rate": sum(r["status"] != "ok" for r in rows) / max(1, len(rows)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--optimize", type=int, default=5)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["feedforward", "optimized", "no_vae", "no_global", "oracle_cameras"],
    )
    evaluate(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
