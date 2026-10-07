"""Fixed GT-region image metrics. Predicted masks never remove scoring pixels."""

import numpy as np
import cv2


def image_metrics(prediction, truth, depth):
    prediction = np.asarray(prediction, dtype=np.float64) / 255
    truth = np.asarray(truth, dtype=np.float64) / 255
    valid = np.asarray(depth) > 0
    if (
        prediction.shape != truth.shape
        or truth.shape[:2] != valid.shape
        or not valid.any()
    ):
        raise ValueError("Incompatible images or empty GT scoring region")
    mse = np.mean((prediction[valid] - truth[valid]) ** 2)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mx, my = blur(prediction), blur(truth)
    vx = np.maximum(blur(prediction**2) - mx**2, 0)
    vy = np.maximum(blur(truth**2) - my**2, 0)
    covariance = blur(prediction * truth) - mx * my
    ssim = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mx**2 + my**2 + 0.01**2) * (vx + vy + 0.03**2)
    )
    logd = np.log(np.maximum(depth, 0.001))
    edge = np.zeros_like(valid)
    jump = np.abs(np.diff(logd, axis=1)) > 0.015
    edge[:, 1:] |= jump
    edge[:, :-1] |= jump
    jump = np.abs(np.diff(logd, axis=0)) > 0.015
    edge[1:] |= jump
    edge[:-1] |= jump
    edge = (
        cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        & valid
    )
    score = lambda mask: (
        float(
            -10 * np.log10(max(np.mean((prediction[mask] - truth[mask]) ** 2), 1e-12))
        )
        if mask.any()
        else None
    )
    return dict(
        psnr_png_gt_valid=float(-10 * np.log10(max(mse, 1e-12))),
        ssim_gt_valid=float(ssim[valid].mean()),
        edge_psnr_png=score(edge),
        interior_psnr_png=score(valid & ~edge),
        edge_fraction_gt_valid=float(edge.sum() / valid.sum()),
        ssim_protocol="RGB 11x11 Gaussian sigma1.5, GT-valid window centers; PNG 8-bit colors",
    )
