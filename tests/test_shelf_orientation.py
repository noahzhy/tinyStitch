import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from backend.research.shelf_orientation import direction, estimate, project
from backend.research.data import shelf_scene


def test_calibrated_axis_at_infinity_and_rotated():
    K = np.array([[210.0, 0, 128], [0, 210, 96], [0, 0, 1]])
    for angle in (0, 6):
        R = Rotation.from_euler("y", angle, degrees=True).as_matrix()
        axis = R @ np.array([1.0, 0, 0])
        lines = []
        for y in (-0.5, 0, 0.5):
            xyz = np.array([[-1, y, 3], [1, y, 3]]) @ R.T
            q = xyz @ K.T
            lines.append(q[:, :2] / q[:, 2:])
        recovered = direction(np.array(lines), K)
        assert abs(recovered @ axis) > 0.999999


def test_missing_lines_fails():
    with pytest.raises(ValueError):
        estimate(np.zeros((192, 256, 3), np.uint8), np.eye(3))


def test_varied_layout_deterministic_and_grouped():
    layout = dict(layers=2, group_size=1000, uneven=True, empty_probability=0.0)
    a, _, _ = shelf_scene(3, 10.86, layout)
    b, _, _ = shelf_scene(3, 10.86, layout)
    assert len(a) == len(b)
    assert all(
        np.array_equal(x, y) for box, other in zip(a, b) for x, y in zip(box, other)
    )
    products = [box for box in a if np.ptp(box[2]) > 0.3]
    assert len(set(tuple(box[2]) for box in products)) <= 2
    assert len(products) > 50


def test_correspondence_visibility_and_empty_overlap():
    from backend.research.matchingadapt import correspondence

    depth = np.full((16, 20), 3.0, dtype=np.float32)
    K = np.array([[20.0, 0, 10], [0, 20, 8], [0, 0, 1]])
    T = np.eye(4)
    grid, valid = correspondence(depth, depth, K, K, T, T)
    assert valid.sum() == 20
    assert np.allclose(grid[0, 0], [-1, -1])
    _, blocked = correspondence(depth, depth * 0.5, K, K, T, T)
    assert blocked.sum() == 0


def test_sequence_consensus_rejects_repeated_product_jumps():
    from backend.research.shelf_orientation import sequence_consensus

    samples = []
    for i in range(6):
        a = np.zeros((20, 2))
        b = a - np.array([8.0, 0.0])
        b[-4:, 0] -= 35  # wrong repeated item
        samples.append((i, i + 1, a, b))
    selected, report = sequence_consensus(samples, 7)
    assert report["rejected_correspondences"] == 24
    assert report["retained_edges"] == 6
    assert np.allclose(report["velocity_px_per_frame"], [8, 0])
    assert all(len(a) == 16 for _, _, a, b in selected)


def test_sequence_consensus_disconnected_fails():
    from backend.research.shelf_orientation import sequence_consensus

    a = np.zeros((10, 2))
    b = a - 8
    with pytest.raises(ValueError, match="disconnected"):
        sequence_consensus([(0, 1, a, b), (1, 2, a, b), (2, 3, a, b)], 5)


def test_mixed_bays_have_independent_layer_counts():
    layout = {
        "bays": [
            dict(layers=2, group_size=3, uneven=True),
            dict(layers=6, group_size=1000, uneven=True),
        ]
    }
    boxes, _, _ = shelf_scene(7, 6.0, layout)
    shelves = [
        (low, high)
        for low, high, color in boxes
        if np.allclose(color, [0.7, 0.72, 0.75]) and high[0] - low[0] > 2
    ]
    assert len(shelves) == 8
    assert sum(high[0] <= 0 for low, high in shelves) == 2
    assert sum(low[0] >= 0 for low, high in shelves) == 6
    assert len({round(float(low[1]), 3) for low, high in shelves}) > 6


def test_camera_lift_recovers_translation_and_anchor_gauge(tmp_path):
    import json, hashlib, torch
    from backend.research.alignment_pose import lift

    rgb = torch.zeros(3, 3, 8, 8)
    depth = torch.ones(3, 8, 8)
    K = torch.tensor([[20.0, 0, 4], [0, 20.0, 4], [0, 0, 1.0]])[None].repeat(3, 1, 1)
    H = torch.eye(3)[None].repeat(3, 1, 1)
    H[:, 0, 2] = torch.tensor([-4.0, 0.0, 4.0])
    payload = dict(
        version=1,
        input_sha256=hashlib.sha256(
            rgb.numpy().tobytes() + K.numpy().tobytes()
        ).hexdigest(),
        rotations=torch.eye(3)[None].repeat(3, 1, 1).tolist(),
        front_transforms=H.tolist(),
    )
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(payload))
    T, info = lift(path, rgb, K, depth)
    assert torch.allclose(T[1], torch.eye(4))
    assert torch.allclose(T[:, 0, 3], torch.tensor([0.2, 0.0, -0.2]), atol=1e-6)
    assert not info["uses_gt_geometry"]
    with pytest.raises(ValueError, match="does not match"):
        lift(path, rgb + 1, K, depth)


def test_level_prior_reduces_spurious_pitch_preserving_valid_rotation():
    from backend.research.shelf_orientation import level_rotation_prior

    true = (
        Rotation.from_euler("y", 6, degrees=True).as_matrix()
        @ Rotation.from_euler("z", 2, degrees=True).as_matrix()
    )
    biased = Rotation.from_euler("x", 5, degrees=True).as_matrix() @ true
    corrected = level_rotation_prior(np.array([biased]))[0]
    before = Rotation.from_matrix(biased @ true.T).magnitude()
    after = Rotation.from_matrix(corrected @ true.T).magnitude()
    assert after < before * 0.2
    assert np.allclose(corrected @ corrected.T, np.eye(3), atol=1e-8)
    assert np.linalg.det(corrected) > 0.999999


def test_component_merge_preserves_renderer_and_rejects_encoder_changes(tmp_path):
    import torch
    from backend.research.combine import run

    renderer = tmp_path / "renderer.pt"
    depth = tmp_path / "depth.pt"
    a = {
        "model": {
            "encoder.weight": torch.ones(1),
            "field.weight": torch.tensor([2.0]),
            "pose.depth.weight": torch.tensor([3.0]),
        }
    }
    b = {
        "model": {
            "encoder.weight": torch.ones(1),
            "field.weight": torch.tensor([9.0]),
            "pose.depth.weight": torch.tensor([4.0]),
        },
        "depth_adaptation": {"selected_step": 100},
    }
    torch.save(a, renderer)
    torch.save(b, depth)
    run(renderer, depth, tmp_path / "merged")
    result = torch.load(tmp_path / "merged/model.pt", weights_only=False)
    assert result["model"]["field.weight"].item() == 2
    assert result["model"]["pose.depth.weight"].item() == 4
    assert result["reference_depth_calibration"]["weights"]["depth.weight"].item() == 3
    assert len(result["reference_depth_calibration"]["checkpoint_sha256"]) == 64
    b["model"]["encoder.weight"] = torch.zeros(1)
    torch.save(b, depth)
    with pytest.raises(ValueError, match="Frozen"):
        run(renderer, depth, tmp_path / "bad")


def test_frozen_source_scale_keeps_camera_gauge_when_depth_head_changes(tmp_path):
    import json, hashlib, torch
    from backend.research.alignment_pose import lift

    rgb = torch.zeros(3, 3, 8, 8)
    K = torch.tensor([[20.0, 0, 4], [0, 20.0, 4], [0, 0, 1.0]])[None].repeat(3, 1, 1)
    H = torch.eye(3)[None].repeat(3, 1, 1)
    H[:, 0, 2] = torch.tensor([-4.0, 0.0, 4.0])
    data = dict(
        version=1,
        input_sha256=hashlib.sha256(
            rgb.numpy().tobytes() + K.numpy().tobytes()
        ).hexdigest(),
        rotations=torch.eye(3)[None].repeat(3, 1, 1).tolist(),
        front_transforms=H.tolist(),
        reference_depth_scale=1.0,
    )
    p = tmp_path / "fixed.json"
    p.write_text(json.dumps(data))
    a, _ = lift(p, rgb, K, torch.ones(3, 8, 8))
    b, info = lift(p, rgb, K, torch.ones(3, 8, 8) * 1.1)
    assert torch.equal(a, b)
    assert info["frozen_reference_scale"]
    assert info["predicted_current_depth_scale"] > 1.09
    data["reference_depth_scale"] = -1
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid predicted depth scale"):
        lift(p, rgb, K, torch.ones(3, 8, 8))
