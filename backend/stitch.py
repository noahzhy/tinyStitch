"""Planar RGB mosaic: SIFT, mutual ratio matches, RANSAC, graph alignment, feather blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

cv2.setNumThreads(2)


class StitchError(ValueError):
    pass


def load_image(path: Path, max_side=1280):
    with Image.open(path) as raw:
        if raw.width * raw.height > 50_000_000:
            raise StitchError(f"{path.name} 超过 5000 万像素，请缩小后导入")
        img = ImageOps.exif_transpose(raw).convert("RGB")
        original = img.size
        if min(original) < 100:
            raise StitchError(f"{path.name} 分辨率过低，短边至少 100 像素")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), original


def points(H, xy):
    return cv2.perspectiveTransform(
        np.asarray(xy, np.float32).reshape(-1, 1, 2), H
    ).reshape(-1, 2)


def quad(img):
    h, w = img.shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def pair_match(i, j, images, features, ratio=0.76):
    ki, di = features[i]
    kj, dj = features[j]
    diagnostic = {"i": i, "j": j, "accepted": False, "matches": 0, "inliers": 0}
    if di is None or dj is None or min(len(di), len(dj)) < 12:
        diagnostic["reason"] = "纹理不足：可匹配特征点太少"
        return diagnostic, None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward = {
        m.queryIdx: m.trainIdx
        for row in bf.knnMatch(dj, di, k=2)
        if len(row) == 2
        for m, n in [row]
        if m.distance < ratio * n.distance
    }
    reverse = {
        m.queryIdx: m.trainIdx
        for row in bf.knnMatch(di, dj, k=2)
        if len(row) == 2
        for m, n in [row]
        if m.distance < ratio * n.distance
    }
    matches = [(q, t) for q, t in forward.items() if reverse.get(t) == q]
    diagnostic["matches"] = len(matches)
    if len(matches) < 12:
        diagnostic["reason"] = "重叠不足或商品重复：双向匹配少于 12 点"
        return diagnostic, None
    pj = np.float32([kj[q].pt for q, _ in matches])
    pi = np.float32([ki[t].pt for _, t in matches])
    H, mask = cv2.findHomography(
        pj, pi, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.995
    )
    if H is None or mask is None:
        diagnostic["reason"] = "无法估计平面变换"
        return diagnostic, None
    ok = mask.ravel().astype(bool)
    n = int(ok.sum())
    diagnostic.update(inliers=n, inlier_ratio=n / len(matches))
    if n < 12 or n / len(matches) < 0.35:
        diagnostic["reason"] = "几何一致性不足：RANSAC 内点太少"
        return diagnostic, None
    pi, pj = pi[ok], pj[ok]
    coverage = min(
        cv2.contourArea(cv2.convexHull(p)) / (im.shape[0] * im.shape[1])
        for p, im in [(pi, images[i]), (pj, images[j])]
    )
    warped = points(H, quad(images[j]))
    area_ratio = cv2.contourArea(warped) / (images[j].shape[0] * images[j].shape[1])
    error = np.linalg.norm(points(H, pj) - pi, axis=1)
    diagnostic.update(coverage=float(coverage), median_error_px=float(np.median(error)))
    # Reject local repeated-label matches, folded/projectively singular frames and drastic scale changes.
    denominator = np.c_[quad(images[j]), np.ones(4)] @ H[2]
    if (
        coverage < 0.008
        or not cv2.isContourConvex(warped)
        or cv2.contourArea(warped, oriented=True) <= 0
        or not 0.3 < area_ratio < 3
        or np.min(denominator) * np.max(denominator) <= 0
        or not np.isfinite(warped).all()
    ):
        diagnostic["reason"] = "变换异常或匹配只集中在局部商品，无法可靠拼接"
        return diagnostic, None
    diagnostic["accepted"] = True
    return diagnostic, {
        "i": i,
        "j": j,
        "H": H / H[2, 2],
        "pi": pi,
        "pj": pj,
        "weight": n / (1 + float(np.median(error))),
    }


def graph_transforms(n, edges):
    anchor = n // 2
    transforms = {anchor: np.eye(3)}
    while len(transforms) < n:
        crossing = [
            e for e in edges if (e["i"] in transforms) != (e["j"] in transforms)
        ]
        if not crossing:
            missing = [i + 1 for i in range(n) if i not in transforms]
            raise StitchError(
                f"序列没有连成一张图，第 {missing} 张图片与主体断开。请补拍重叠区域，或按同一货架拆分序列；没有生成伪完整拼图。"
            )
        edge = max(crossing, key=lambda e: e["weight"])
        i, j, H = edge["i"], edge["j"], edge["H"]
        if i in transforms:
            transforms[j] = transforms[i] @ H
        else:
            transforms[i] = transforms[j] @ np.linalg.inv(H)
    return [transforms[i] / transforms[i][2, 2] for i in range(n)], anchor


def refine(transforms, anchor, edges):
    """Optimize one shared plane against all accepted RGB correspondences, anchor fixed."""
    ids = [i for i in range(len(transforms)) if i != anchor]
    offsets = {i: k * 8 for k, i in enumerate(ids)}
    samples = []
    for e in edges:
        step = max(1, len(e["pi"]) // 64)
        samples.append((e["i"], e["j"], e["pi"][::step][:64], e["pj"][::step][:64]))
    x0 = np.concatenate([transforms[i].ravel()[:8] for i in ids])

    def unpack(x):
        out = [np.eye(3) for _ in transforms]
        for i in ids:
            out[i] = np.r_[x[offsets[i] : offsets[i] + 8], 1.0].reshape(3, 3)
        return out

    # Double precision projection is essential for finite-difference optimization.
    def projection(H, xy):
        xyh = np.c_[xy.astype(np.float64), np.ones(len(xy))] @ H.T
        return xyh[:, :2] / np.maximum(np.abs(xyh[:, 2:]), 1e-8) * np.sign(xyh[:, 2:])

    def residual(x):
        hs = unpack(x)
        return np.concatenate(
            [
                (projection(hs[i], pi) - projection(hs[j], pj)).ravel()
                for i, j, pi, pj in samples
            ]
        )

    rows = sum(len(pi) * 2 for _, _, pi, _ in samples)
    sparsity = lil_matrix((rows, len(x0)), dtype=int)
    offset = 0
    for i, j, pi, _ in samples:
        for k in [i, j]:
            if k in offsets:
                sparsity[offset : offset + len(pi) * 2, offsets[k] : offsets[k] + 8] = 1
        offset += len(pi) * 2
    before = float(np.median(np.abs(residual(x0))))
    fit = least_squares(
        residual,
        x0,
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=50,
        x_scale="jac",
        ftol=1e-4,
    )
    after = float(np.median(np.abs(residual(fit.x))))
    if np.isfinite(fit.x).all() and after <= before:
        return unpack(fit.x), {"before_median_px": before, "after_median_px": after}
    return transforms, {"before_median_px": before, "after_median_px": before}


def blend(images, transforms, progress):
    for H, im in zip(transforms, images):
        denominators = np.c_[quad(im), np.ones(4)] @ H[2]
        if np.min(denominators) * np.max(denominators) <= 0 or not cv2.isContourConvex(
            points(H, quad(im))
        ):
            raise StitchError("全序列优化产生不稳定的透视变换，请增加照片重叠")
    all_corners = np.concatenate(
        [points(H, quad(im)) for H, im in zip(transforms, images)]
    )
    if not np.isfinite(all_corners).all():
        raise StitchError("图像变换不稳定，请增加重叠并尽量正对货架拍摄")
    lo = np.floor(all_corners.min(0)).astype(int)
    hi = np.ceil(all_corners.max(0)).astype(int)
    width, height = (hi - lo).tolist()
    if (
        min(width, height) < 100
        or max(width, height) > 16000
        or width * height > 16_000_000
    ):
        raise StitchError(
            "拼图范围异常或超过 1600 万像素，可能存在错误匹配；请拆分序列"
        )
    shift = np.array([[1.0, 0, -lo[0]], [0, 1.0, -lo[1]], [0, 0, 1.0]])
    total = np.zeros((height, width, 3), np.float32)
    weights = np.zeros((height, width), np.float32)
    exposure = []
    # Select the view nearest each pixel's image center, then blend only a narrow seam.
    # Broad averaging would smear raised products when the camera translates.
    owner = np.full((height, width), -1, np.int16)
    best = np.zeros((height, width), np.float32)
    for i, (image, H) in enumerate(zip(images, transforms)):
        progress(0.60 + 0.10 * i / len(images), f"寻找接缝 {i + 1}/{len(images)}")
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        score = np.exp(
            -2 * ((xx - w / 2) / (w / 2)) ** 2 - 0.15 * ((yy - h / 2) / (h / 2)) ** 2
        ).astype(np.float32)
        score[[0, -1], :] = 0
        score[:, [0, -1]] = 0
        warped_score = cv2.warpPerspective(score, shift @ H, (width, height))
        take = warped_score > best
        best[take] = warped_score[take]
        owner[take] = i
    for i, (image, H) in enumerate(zip(images, transforms)):
        progress(0.70 + 0.24 * i / len(images), f"融合第 {i + 1}/{len(images)} 张图片")
        h, w = image.shape[:2]
        valid = np.ones((h, w), np.uint8)
        valid[[0, -1], :] = 0
        valid[:, [0, -1]] = 0
        M = shift @ H
        warped = cv2.warpPerspective(image, M, (width, height)).astype(np.float32)
        coverage = cv2.warpPerspective(valid.astype(np.float32), M, (width, height))
        weight = (
            cv2.GaussianBlur((owner == i).astype(np.float32), (11, 11), 2.0) * coverage
        )
        overlap = (weight > 0.4) & (weights > 0.4)
        gain = np.ones(3)
        if overlap.sum() > 1000:
            existing = total[overlap] / weights[overlap, None]
            sample = warped[overlap]
            gain = np.clip(
                np.median(existing + 5, axis=0) / np.median(sample + 5, axis=0),
                0.8,
                1.25,
            )
        exposure.append(gain.tolist())
        total += np.clip(warped * gain, 0, 255) * weight[:, :, None]
        weights += weight
    mosaic = np.clip(total / np.maximum(weights[:, :, None], 1e-6), 0, 255).astype(
        np.uint8
    )
    alpha = np.uint8(weights > 0.01) * 255
    x, y, w, h = cv2.boundingRect(alpha)
    mosaic, alpha = mosaic[y : y + h, x : x + w], alpha[y : y + h, x : x + w]
    crop = np.array([[1.0, 0, -x], [0, 1.0, -y], [0, 0, 1.0]])
    return mosaic, alpha, [crop @ shift @ H for H in transforms], exposure


def stitch(
    paths: list[Path],
    output: Path,
    progress: Callable = lambda *_: None,
    method="jepa",
    checkpoint=None,
    device="auto",
):
    started = time.perf_counter()
    if not 2 <= len(paths) <= 48:
        raise StitchError("请导入 2–48 张按拍摄顺序排列的同一货架照片")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(17)
    images, metadata, features = [], [], []
    model_info = None
    if method == "jepa":
        from .jepa import JEPADescriptor, DEFAULT_CHECKPOINT

        extractor = JEPADescriptor(checkpoint or DEFAULT_CHECKPOINT, device)
        model_info = extractor.info
    elif method == "sift":
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.025)
    else:
        raise StitchError("未知匹配方法")
    digest = hashlib.sha256()
    for i, path in enumerate(paths):
        progress(0.03 + 0.20 * i / len(paths), f"读取与提取特征 {i + 1}/{len(paths)}")
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        image, original = load_image(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = (
            extractor.extract(image)
            if method == "jepa"
            else sift.detectAndCompute(gray, None)
        )
        features.append((kp, desc))
        images.append(image)
        metadata.append(
            {
                "name": path.name,
                "original_size": list(original),
                "work_size": [image.shape[1], image.shape[0]],
                "features": len(kp),
                "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            }
        )
    edges, diagnostics = [], []
    pairs = [
        (i, j) for i in range(len(paths)) for j in range(i + 1, min(i + 4, len(paths)))
    ]
    for k, (i, j) in enumerate(pairs):
        progress(0.23 + 0.26 * k / len(pairs), f"自动匹配图片 {i + 1} 与 {j + 1}")
        diag, edge = pair_match(
            i, j, images, features, ratio=0.82 if method == "jepa" else 0.76
        )
        diagnostics.append(diag)
        if edge is not None:
            edges.append(edge)
    try:
        transforms, anchor = graph_transforms(len(images), edges)
    except StitchError:
        (output / "diagnostics.json").write_text(
            json.dumps(
                {"images": metadata, "pairs": diagnostics}, ensure_ascii=False, indent=2
            )
        )
        raise
    progress(0.51, "优化整张货架的图像对齐")
    transforms, alignment = refine(transforms, anchor, edges)
    mosaic, alpha, final_h, exposure = blend(images, transforms, progress)
    warnings = [
        "结果只覆盖输入照片实际拍到的区域；没有推断或补画未拍到的商品。",
        "商品凸起、相机前后移动或大幅转动可能产生重影；当前采用单个平面模型。",
    ]
    if alignment["after_median_px"] > 2:
        warnings.append("跨图匹配残差偏大，请检查接缝重影或补拍。")
    if any(im["blur_score"] < 15 for im in metadata):
        warnings.append("有图片清晰度偏低，建议重新拍摄。")
    rgba = cv2.cvtColor(mosaic, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imwrite(str(output / "panorama.png"), rgba)
    jpg = mosaic.copy()
    jpg[alpha == 0] = 255
    cv2.imwrite(str(output / "panorama.jpg"), jpg, [cv2.IMWRITE_JPEG_QUALITY, 95])
    report = {
        "algorithm": f"{method.upper()} descriptors + mutual ratio + RANSAC homography + shared-plane refinement + nearest-center seams + narrow feather",
        "method": method,
        "model": model_info,
        "input_digest": digest.hexdigest(),
        "input_count": len(paths),
        "size": [mosaic.shape[1], mosaic.shape[0]],
        "anchor_image": anchor,
        "seconds": time.perf_counter() - started,
        "coordinate_system": "panorama pixels, no metric scale",
        "images": metadata,
        "pairs": diagnostics,
        "alignment": alignment,
        "transforms_work_to_panorama": [H.tolist() for H in final_h],
        "exposure_gain": exposure,
        "warnings": warnings,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    progress(1.0, "拼图完成")
    return report


def main():
    parser = argparse.ArgumentParser(description="按文件名顺序拼接货架 RGB 图片")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["jepa", "sift"], default="jepa")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    import re

    paths = sorted(
        [
            p
            for p in args.input.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ],
        key=lambda p: [
            int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)
        ],
    )
    report = stitch(
        paths,
        args.output,
        lambda p, text: print(f"{p:.0%} {text}", flush=True),
        method=args.method,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(
        json.dumps(
            {k: report[k] for k in ["size", "seconds", "input_digest"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
