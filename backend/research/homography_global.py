"""Source-only shelf-line global homography optimization; no neural rendering."""

import argparse, json, time
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from backend.stitch import pair_match, graph_transforms, refine, blend, points


def horizontal_lines(image):
    lines = cv2.createLineSegmentDetector().detect(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )[0]
    if lines is None:
        return np.empty((0, 2, 2))
    lines = lines.reshape(-1, 2, 2)
    delta = lines[:, 1] - lines[:, 0]
    return lines[
        (np.abs(delta[:, 0]) > image.shape[1] * 0.4)
        & (np.abs(delta[:, 1]) < np.abs(delta[:, 0]) * 0.06)
    ]


def optimize(images, transforms, anchor, edges, weight=1.0):
    """Anchor-observed long shelf lines define shared rows; sparse robust global fit."""
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Invalid global constraint weight")
    lines = [horizontal_lines(im) for im in images]
    values = sorted(lines[anchor].mean(1)[:, 1])
    rows = []
    for y in values:
        if rows and abs(y - rows[-1][-1]) < 4:
            rows[-1].append(y)
        else:
            rows.append([y])
    rows = np.array([np.mean(r) for r in rows])
    if len(rows) < 3:
        raise ValueError("Insufficient global shelf rows")
    ids = [i for i in range(len(images)) if i != anchor]
    offsets = {i: k * 8 for k, i in enumerate(ids)}
    sample = [(e["i"], e["j"], e["pi"][::2][:96], e["pj"][::2][:96]) for e in edges]
    # Alternating correspondence indices are withheld from fitting.
    hold = [(e["i"], e["j"], e["pi"][1::2], e["pj"][1::2]) for e in edges]
    x0 = np.concatenate([transforms[i].ravel()[:8] for i in ids])
    constraints = []
    for i, ls in enumerate(lines):
        for line in ls:
            mapped = points(transforms[i], line)
            which = np.abs(rows - mapped[:, 1].mean()).argmin()
            if abs(rows[which] - mapped[:, 1].mean()) < 18:
                constraints.append((i, line, float(rows[which])))

    def unpack(x):
        out = [np.eye(3) for _ in images]
        for i in ids:
            out[i] = np.r_[x[offsets[i] : offsets[i] + 8], 1.0].reshape(3, 3)
        return out

    def projection(H, xy):
        q = np.c_[xy, np.ones(len(xy))] @ H.T
        den = np.where(np.abs(q[:, 2:]) < 1e-8, 1e-8, q[:, 2:])
        return q[:, :2] / den

    def residual(x):
        hs = unpack(x)
        result = [
            (projection(hs[i], a) - projection(hs[j], b)).ravel()
            for i, j, a, b in sample
        ]
        result.extend(
            (projection(hs[i], line)[:, 1] - y) * weight for i, line, y in constraints
        )
        return np.concatenate(result)

    sizes = [(len(a) * 2, [i, j]) for i, j, a, b in sample] + [
        (2, [i]) for i, line, y in constraints
    ]
    sparse = lil_matrix((sum(k for k, _ in sizes), len(x0)), dtype=int)
    r = 0
    for count, views in sizes:
        for i in views:
            if i in offsets:
                sparse[r : r + count, offsets[i] : offsets[i] + 8] = 1
        r += count
    fit = least_squares(
        residual,
        x0,
        jac_sparsity=sparse.tocsr(),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=120,
        x_scale="jac",
        ftol=1e-6,
    )
    hs = unpack(fit.x)
    if not np.isfinite(fit.x).all():
        raise ValueError("Nonfinite optimization")

    def diagnostics(hs):
        errors = np.concatenate(
            [
                np.linalg.norm(projection(hs[i], a) - projection(hs[j], b), axis=1)
                for i, j, a, b in hold
            ]
        )
        line_error = np.concatenate(
            [projection(hs[i], line)[:, 1] - y for i, line, y in constraints]
        )
        return dict(
            withheld_match_median_px=float(np.median(errors)),
            withheld_match_p95_px=float(np.percentile(errors, 95)),
            global_row_rms_px=float(np.sqrt(np.mean(line_error**2))),
        )

    return hs, dict(
        before=diagnostics(transforms),
        after=diagnostics(hs),
        anchor_rows=rows.tolist(),
        line_constraints=len(constraints),
        weight=weight,
        success=bool(fit.success),
        evaluations=fit.nfev,
    )


def translation_global(images, transforms, anchor, edges, weight):
    # Fixed-focal near-front shelf acquisition: one common affine basis, global translations.
    # This constrained candidate deliberately cannot absorb per-frame projective deformation.
    def projection(H, xy):
        q = np.c_[np.asarray(xy, dtype=np.float64), np.ones(len(xy))] @ H.T
        return q[:, :2] / q[:, 2:]

    ids = [i for i in range(len(images)) if i != anchor]
    offset = {i: 2 * k for k, i in enumerate(ids)}
    rows = horizontal_lines(images[anchor]).mean(1)[:, 1]
    if len(rows) < 3:
        raise ValueError("Insufficient global shelf rows")
    constraints = []
    for i, image in enumerate(images):
        for line in horizontal_lines(image):
            initial = line.mean(0)[1]
            if len(rows):
                y = rows[np.abs(rows - initial).argmin()]
                if abs(y - initial) < 3:
                    constraints.append((i, line, float(y)))

    def unpack(x):
        hs = [np.eye(3) for _ in images]
        for i in ids:
            hs[i][:2, 2] = x[offset[i] : offset[i] + 2]
        return hs

    x = np.concatenate([transforms[i][:2, 2] for i in ids])

    def residual(x):
        hs = unpack(x)
        result = [
            (
                projection(hs[e["i"]], e["pi"][::2][:96])
                - projection(hs[e["j"]], e["pj"][::2][:96])
            ).ravel()
            for e in edges
        ]
        result.extend(
            (projection(hs[i], line)[:, 1] - y) * weight for i, line, y in constraints
        )
        return np.concatenate(result)

    fit = least_squares(
        residual, x, loss="soft_l1", f_scale=2.0, max_nfev=60, ftol=1e-7
    )
    hs = unpack(fit.x)

    def diagnostics(hs):
        errors = np.concatenate(
            [
                np.linalg.norm(
                    projection(hs[e["i"]], e["pi"][1::2])
                    - projection(hs[e["j"]], e["pj"][1::2]),
                    axis=1,
                )
                for e in edges
            ]
        )
        lines = np.concatenate(
            [projection(hs[i], line)[:, 1] - y for i, line, y in constraints]
        )
        return dict(
            withheld_match_median_px=float(np.median(errors)),
            withheld_match_p95_px=float(np.percentile(errors, 95)),
            global_row_rms_px=float(np.sqrt(np.mean(lines**2))),
        )

    return hs, dict(
        before=diagnostics(transforms),
        after=diagnostics(hs),
        line_constraints=len(constraints),
        weight=weight,
        success=bool(fit.success),
        evaluations=fit.nfev,
    )


def plane_error(scene, hs, plane_z=0.2):
    # GT enters evaluation only; estimate H without camera labels or target image.
    meta = json.loads((scene / "intrinsics.json").read_text())
    K = np.array(meta["K"])
    T = np.load(scene / "labels/geometry.npz")["T"]
    a = len(hs) // 2
    width = json.loads((scene / "scene.json").read_text())["shelf_width"]
    grid = np.array(
        [
            [x, y, plane_z]
            for x in np.linspace(-width / 2, width / 2, 160)
            for y in (0.3, 0.8, 1.3, 1.8)
        ]
    )

    def project(i):
        q = grid @ T[i, :3, :3].T + T[i, :3, 3]
        uv = q @ K[i].T
        return uv[:, :2] / uv[:, 2:]

    target = project(a)
    errors = []
    frame_errors = []
    for i, H in enumerate(hs):
        uv = project(i)
        valid = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < meta["width"])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < meta["height"])
        )
        e = np.linalg.norm(points(H, uv[valid]) - target[valid], axis=1)
        errors.extend(e)
        frame_errors.append(float(np.mean(e)))
    return dict(
        backplane_mean_px=float(np.mean(errors)),
        backplane_p95_px=float(np.percentile(errors, 95)),
        per_frame_mean_px=frame_errors,
    )


def product_error(scene, hs):
    from .data import shelf_scene

    meta = json.loads((scene / "intrinsics.json").read_text())
    K = np.array(meta["K"])
    T = np.load(scene / "labels/geometry.npz")["T"]
    anchor = len(hs) // 2
    width = json.loads((scene / "scene.json").read_text())["shelf_width"]
    layout = json.loads((scene / "scene.json").read_text()).get("layout")
    boxes, _, _ = shelf_scene(int(scene.name), width, layout)
    cloud = np.array(
        [
            [low[0] + u * (high[0] - low[0]), low[1] + v * (high[1] - low[1]), low[2]]
            for low, high, color in boxes
            if np.ptp(color) > 0.3
            for u in (0.25, 0.5, 0.75)
            for v in (0.25, 0.5, 0.75)
        ]
    )

    def projection(i):
        q = cloud @ T[i, :3, :3].T + T[i, :3, 3]
        uv = q @ K[i].T
        return uv[:, :2] / uv[:, 2:], q[:, 2]

    target, _ = projection(anchor)
    errors = []
    for i, H in enumerate(hs):
        uv, z = projection(i)
        px = np.rint(uv).astype(int)
        valid = (
            (px[:, 0] >= 0)
            & (px[:, 0] < meta["width"])
            & (px[:, 1] >= 0)
            & (px[:, 1] < meta["height"])
        )
        ids = np.flatnonzero(valid)
        d = np.load(scene / "labels" / f"{i:04}.npy")
        ids = ids[np.abs(d[px[ids, 1], px[ids, 0]] - z[ids]) < 0.02]
        errors.extend(np.linalg.norm(points(H, uv[ids]) - target[ids], axis=1))
    if not errors:
        raise ValueError("Empty visible product evaluation")
    return dict(
        mean_px=float(np.mean(errors)),
        p95_px=float(np.percentile(errors, 95)),
        visible_point_count=len(errors),
    )


def run(
    scene,
    output,
    method="sift",
    weight=1.0,
    model="translation",
    evaluate_gt=True,
    checkpoint=None,
):
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Invalid global constraint weight")
    source, out = Path(scene), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if model in (
        "translation",
        "orientation",
        "orientation-consensus",
        "orientation-local",
        "orientation-local-level",
    ):
        meta = json.loads((source / "intrinsics.json").read_text())
        intrinsics = np.asarray(meta["K"])
        if not np.allclose(intrinsics, intrinsics[0]):
            raise ValueError("Shared basis requires fixed camera intrinsics")
    start = time.perf_counter()
    cv2.setRNGSeed(17)
    images = [cv2.imread(str(p)) for p in sorted((source / "rgb").glob("*.png"))]
    if method == "jepa":
        import torch

        torch.set_num_threads(4)
        from backend.jepa import JEPADescriptor

        extractor = (
            JEPADescriptor(checkpoint=checkpoint, device="cpu")
            if checkpoint
            else JEPADescriptor(device="cpu")
        )
        features = [extractor.extract(im) for im in images]
    else:
        detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.025)
        features = [
            detector.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
            for im in images
        ]
    edges = []
    for i in range(len(images)):
        for j in range(i + 1, min(i + 4, len(images))):
            _, edge = pair_match(
                i, j, images, features, ratio=0.82 if method == "jepa" else 0.76
            )
            if edge is not None:
                edges.append(edge)
    base, anchor = graph_transforms(len(images), edges)
    base, _ = refine(base, anchor, edges)
    if model in (
        "orientation",
        "orientation-consensus",
        "orientation-local",
        "orientation-local-level",
    ):
        from .shelf_orientation import optimize as orientation_optimize

        new, report = orientation_optimize(
            images,
            base,
            anchor,
            edges,
            weight,
            intrinsics,
            consensus=model
            in (
                "orientation-consensus",
                "orientation-local",
                "orientation-local-level",
            ),
            local_rows=model in ("orientation-local", "orientation-local-level"),
            level_camera=model == "orientation-local-level",
        )
    else:
        new, report = (translation_global if model == "translation" else optimize)(
            images, base, anchor, edges, weight
        )
    for name, hs in [("baseline", base), ("global", new)]:
        mosaic, alpha, final, exposure = blend(images, hs, lambda *_: None)
        cv2.imwrite(str(out / (name + ".png")), np.dstack((mosaic, alpha)))
        if evaluate_gt:
            report[name + "_gt_evaluation"] = plane_error(source, hs)
            report[name + "_frontplane_gt_evaluation"] = plane_error(source, hs, -0.24)
            report[name + "_products_gt_evaluation"] = product_error(source, hs)
    if model in (
        "orientation",
        "orientation-consensus",
        "orientation-local",
        "orientation-local-level",
    ):
        front = [np.asarray(H) for H in report["front_transforms"]]
        mosaic, alpha, _, _ = blend(images, front, lambda *_: None)
        cv2.imwrite(str(out / "front-facing.png"), np.dstack((mosaic, alpha)))
        if evaluate_gt:
            from scipy.spatial.transform import Rotation

            T = np.load(source / "labels/geometry.npz")["T"]
            front_rotation = np.diag([-1.0, -1.0, 1.0])
            errors = [
                Rotation.from_matrix(
                    np.asarray(R) @ pose[:3, :3] @ front_rotation.T
                ).magnitude()
                * 180
                / np.pi
                for R, pose in zip(report["estimated_rotations"], T)
            ]
            raw_errors = [
                Rotation.from_matrix(
                    np.asarray(R) @ pose[:3, :3] @ front_rotation.T
                ).magnitude()
                * 180
                / np.pi
                for R, pose in zip(report["raw_rotations"], T)
            ]
            report["raw_orientation_error_degrees"] = dict(
                mean=float(np.mean(raw_errors)),
                p95=float(np.percentile(raw_errors, 95)),
            )
            report["orientation_error_degrees"] = dict(
                mean=float(np.mean(errors)), p95=float(np.percentile(errors, 95))
            )
    report.update(
        method=method,
        descriptor_model=extractor.info if method == "jepa" else None,
        alignment_model=model,
        source=str(source),
        input_count=len(images),
        elapsed_seconds=time.perf_counter() - start,
        gt_used_for_fitting=False,
        neural_renderer_used=False,
        groundtruth_evaluation=bool(evaluate_gt),
        match_diagnostic_protocol="Odd correspondences withheld from new optimization; baseline initialization used all matches",
        scope="source homography alignment only; long horizontal shelf structure prior",
        transforms_before=[h.tolist() for h in base],
        transforms_after=[h.tolist() for h in new],
    )
    report["quality_warning_threshold_p95_px"] = 5.0
    report["low_confidence"] = bool(
        not report["success"] or report["after"]["withheld_match_p95_px"] > 5.0
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                before=report["before"],
                after=report["after"],
                method=method,
                descriptor_model=extractor.info if method == "jepa" else None,
                model=model,
                output=str(out),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["sift", "jepa"], default="sift")
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument(
        "--model",
        choices=[
            "projective",
            "translation",
            "orientation",
            "orientation-consensus",
            "orientation-local",
            "orientation-local-level",
        ],
        default="translation",
    )
    p.add_argument("--checkpoint")
    p.add_argument("--no-gt-eval", dest="evaluate_gt", action="store_false")
    run(**vars(p.parse_args()))
