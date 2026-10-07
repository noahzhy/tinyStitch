"""Calibrated shelf-axis orientation, followed by global translation alignment."""

import numpy as np
import cv2
from scipy.optimize import least_squares


def project(H, xy):
    q = np.c_[xy, np.ones(len(xy))] @ H.T
    return q[:, :2] / q[:, 2:]


def direction(lines, K):
    if len(lines) < 3:
        raise ValueError("Insufficient shelf axis segments")
    normals = (
        np.cross(
            np.c_[lines[:, 0], np.ones(len(lines))],
            np.c_[lines[:, 1], np.ones(len(lines))],
        )
        @ K
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    weights = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    for _ in range(5):
        _, _, vt = np.linalg.svd(
            normals * np.sqrt(weights[:, None]), full_matrices=False
        )
        axis = vt[-1]
        weights = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1) / np.maximum(
            np.abs(normals @ axis), 0.005
        )
    return axis


def estimate(image, K):
    raw = cv2.createLineSegmentDetector().detect(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    )[0]
    if raw is None:
        raise ValueError("No shelf lines")
    lines = raw.reshape(-1, 2, 2).astype(float)
    d = np.abs(lines[:, 1] - lines[:, 0])
    horizontal = lines[(d[:, 0] > image.shape[1] * 0.2) & (d[:, 1] < d[:, 0] * 0.25)]
    vertical = lines[(d[:, 1] > image.shape[0] * 0.06) & (d[:, 0] < d[:, 1] * 0.25)]
    u, v = direction(horizontal, K), direction(vertical, K)
    u *= 1 if u[0] > 0 else -1
    v *= 1 if v[1] > 0 else -1
    v -= u * np.dot(u, v)
    v /= np.linalg.norm(v)
    rotation = np.stack([u, v, np.cross(u, v)])
    if rotation[2, 2] < 0.5:
        raise ValueError("Ambiguous shelf orientation")
    return rotation, horizontal


def sequence_consensus(samples, count, require_connected=True):
    """Infer short-baseline motion from training matches; reject repeated-item jumps.

    Constant-speed prior applies to ordered, dense shelf sweeps only. No GT step,
    product width, layer count or withheld correspondence is used.
    """
    velocities = np.array(
        [np.median(a - b, axis=0) / (j - i) for i, j, a, b in samples if len(a)]
    )
    if len(velocities) < 3:
        raise ValueError("Insufficient sequence motion support")
    velocity = np.median(velocities, axis=0)
    mad = np.median(np.abs(velocities - velocity), axis=0)
    tolerance = np.clip(3 * 1.4826 * mad, [8.0, 16.0], [10.0, 20.0])
    selected = []
    rejected = 0
    retained = 0
    for i, j, a, b in samples:
        valid = (np.abs((a - b) - velocity * (j - i)) <= tolerance).all(axis=1)
        rejected += int((~valid).sum())
        if valid.sum() >= 6:
            selected.append((i, j, a[valid], b[valid]))
            retained += int(valid.sum())
    reached = {0}
    while True:
        old = len(reached)
        for i, j, _, _ in selected:
            if i in reached or j in reached:
                reached.update([i, j])
        if old == len(reached):
            break
    if require_connected and len(reached) != count:
        raise ValueError(
            f"Sequence consensus disconnected geometry: missing={sorted(set(range(count))-reached)}, velocity={velocity.tolist()}, tolerance={tolerance.tolist()}, kept_edges={len(selected)}"
        )
    return selected, dict(
        velocity_px_per_frame=velocity.tolist(),
        tolerance_px_per_frame=tolerance.tolist(),
        rejected_correspondences=rejected,
        retained_correspondences=retained,
        retained_edges=len(selected),
        input_edges=len(samples),
        assumption="approximately constant-speed dense ordered sweep",
    )


def level_rotation_prior(rotations):
    """Opt-in near-level capture prior; retains yaw/roll and returns proper SO(3)."""
    corrected = []
    for R in rotations:
        vertical = R[1].copy()
        vertical[2] = 0
        vertical /= np.linalg.norm(vertical)
        horizontal = R[0] - np.dot(R[0], vertical) * vertical
        horizontal /= np.linalg.norm(horizontal)
        corrected.append(
            np.stack([horizontal, vertical, np.cross(horizontal, vertical)])
        )
    return np.stack(corrected)


def optimize(
    images,
    transforms,
    anchor,
    edges,
    weight,
    intrinsics,
    consensus=False,
    local_rows=False,
    level_camera=False,
):
    estimates = [estimate(im, K) for im, K in zip(images, intrinsics)]
    from scipy.spatial.transform import Rotation

    raw_rotations = [R.copy() for R, _ in estimates]
    observed = Rotation.from_matrix(np.stack(raw_rotations)).as_rotvec()

    # Joint sequence fit: retain measured shelf axes, regularize angular acceleration.
    # This does not assume a fixed yaw or force every input camera to one orientation.
    def orientation_residual(x):
        vectors = x.reshape(-1, 3)
        return np.r_[
            (vectors - observed).ravel(), 0.7 * np.diff(vectors, n=2, axis=0).ravel()
        ]

    rotation_fit = least_squares(
        orientation_residual, observed.ravel(), loss="soft_l1", f_scale=0.02
    )
    optimized = Rotation.from_rotvec(rotation_fit.x.reshape(-1, 3)).as_matrix()
    if level_camera:
        optimized = level_rotation_prior(optimized)
    estimates = [(R, item[1]) for R, item in zip(optimized, estimates)]
    rect = [
        intrinsics[anchor] @ R @ np.linalg.inv(K)
        for (R, _), K in zip(estimates, intrinsics)
    ]
    ids = [i for i in range(len(images)) if i != anchor]
    offsets = {i: 2 * k for k, i in enumerate(ids)}
    rows = (
        project(rect[anchor], estimates[anchor][1].reshape(-1, 2))
        .reshape(-1, 2, 2)
        .mean(1)[:, 1]
    )
    constraints = []
    for i, (_, lines) in enumerate(estimates):
        for line in lines:
            xy = project(rect[i], line)
            y = rows[np.abs(rows - xy[:, 1].mean()).argmin()]
            if abs(y - xy[:, 1].mean()) < 5:
                constraints.append((i, xy, y))
    if local_rows:
        constraints = []  # Different bays do not share layer heights.
    samples = [
        (
            e["i"],
            e["j"],
            project(rect[e["i"]], e["pi"][::2][:96]),
            project(rect[e["j"]], e["pj"][::2][:96]),
        )
        for e in edges
    ]

    consensus_report = None
    if consensus:
        samples, consensus_report = sequence_consensus(
            samples, len(images), require_connected=False
        )
        velocity = np.asarray(consensus_report["velocity_px_per_frame"])
        tolerance = np.asarray(consensus_report["tolerance_px_per_frame"])
        optical = []
        h, w = images[0].shape[:2]
        grays = [
            cv2.cvtColor(cv2.warpPerspective(im, H, (w, h)), cv2.COLOR_BGR2GRAY)
            for im, H in zip(images, rect)
        ]
        for i in range(len(images) - 1):
            source = cv2.goodFeaturesToTrack(
                grays[i], maxCorners=240, qualityLevel=0.02, minDistance=6
            )
            if source is None:
                continue
            target = (source - velocity.astype(np.float32).reshape(1, 1, 2)).astype(
                np.float32
            )
            target, status, error = cv2.calcOpticalFlowPyrLK(
                grays[i],
                grays[i + 1],
                source,
                target,
                winSize=(15, 15),
                maxLevel=2,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
                grays[i + 1], grays[i], target, None, winSize=(15, 15), maxLevel=2
            )
            a, b = source[:, 0], target[:, 0]
            valid = (
                (status[:, 0] > 0)
                & (back_status[:, 0] > 0)
                & (np.linalg.norm(backward[:, 0] - a, axis=1) < 1.0)
                & (error[:, 0] < 15)
                & (np.abs(a - b - velocity) <= tolerance).all(axis=1)
                & (b[:, 0] > 8)
                & (b[:, 0] < w - 8)
                & (b[:, 1] > 8)
                & (b[:, 1] < h - 8)
            )
            if valid.sum() >= 6:
                optical.append((i, i + 1, a[valid][:96], b[valid][:96]))
        samples.extend(optical)
        # Enforce connectivity after adding independently checked source-image tracks.
        _, support = sequence_consensus(samples, len(images))
        consensus_report["optical_flow_edges"] = len(optical)
        consensus_report["optical_flow_correspondences"] = sum(
            len(a) for _, _, a, b in optical
        )
        consensus_report["connected_after_tracking"] = True

    def translations(x):
        t = np.zeros((len(images), 2))
        for i in ids:
            t[i] = x[offsets[i] : offsets[i] + 2]
        return t

    initial = np.concatenate(
        [
            (
                project(rect[anchor] @ transforms[i], np.array([[0.0, 0.0]]))
                - project(rect[i], np.array([[0.0, 0.0]]))
            )[0]
            for i in ids
        ]
    )

    def residual(x):
        t = translations(x)
        return np.concatenate(
            [(a + t[i] - b - t[j]).ravel() for i, j, a, b in samples]
            + [((xy + t[i])[:, 1] - y) * weight for i, xy, y in constraints]
        )

    fit = least_squares(residual, initial, loss="soft_l1", f_scale=2, max_nfev=60)
    t = translations(fit.x)
    hs = []
    for i, H in enumerate(rect):
        shift = np.eye(3)
        shift[:2, 2] = t[i]
        hs.append(shift @ H)
    # Return transforms in the original anchor gauge for identical geometry scoring.
    # Front-facing mosaics use front_transforms separately; no claim of orthographic depth correction.
    gauge = np.linalg.inv(rect[anchor])
    result = [gauge @ H for H in hs]

    def diagnostics(transforms):
        errors = np.concatenate(
            [
                np.linalg.norm(
                    project(transforms[e["i"]], e["pi"][1::2])
                    - project(transforms[e["j"]], e["pj"][1::2]),
                    axis=1,
                )
                for e in edges
            ]
        )
        return dict(
            withheld_match_median_px=float(np.median(errors)),
            withheld_match_p95_px=float(np.percentile(errors, 95)),
        )

    return result, dict(
        level_camera_prior=level_camera,
        global_equal_row_prior=not local_rows,
        sequence_consensus=consensus_report,
        before=diagnostics(transforms),
        after=diagnostics(result),
        success=bool(fit.success),
        evaluations=fit.nfev,
        line_constraints=len(constraints),
        raw_rotations=[R.tolist() for R in raw_rotations],
        orientation_optimization_success=bool(rotation_fit.success),
        orientation_regularization="global observed-axis prior plus angular second difference weight 0.7",
        estimated_rotations=[R.tolist() for R, _ in estimates],
        front_transforms=[H.tolist() for H in hs],
        anchor_rectification=rect[anchor].tolist(),
        scope="shelf rotation estimation plus shared calibrated basis and global translations",
    )
