import json
import torch
import numpy as np
from backend.research.geometry import *
from backend.research.data import generate, load_supervised, load_input
from backend.research.model import Pipeline
from backend.research.train import jepa_loss, rendering_loss


def test_geometry_roundtrip_and_conversion():
    K = torch.tensor([[30.0, 0, 15], [0, 30, 11], [0, 0, 1]])
    T = torch.eye(4)
    d = torch.ones(24, 32)
    xyz = backproject(d, K, T)
    xy, z = project(xyz, K, T)
    assert torch.allclose(xy, pixels(24, 32)[..., :2], atol=1e-5)
    assert torch.allclose(z, d)
    grid, valid = correspondence(d, d, K, K, T, T)
    assert valid.float().mean() > 0.9
    _, blocked = correspondence(d, d * 2, K, K, T, T)
    assert not blocked.any()
    assert torch.equal(
        three_to_opencv(torch.eye(4)), torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    )
    assert scale_intrinsics(K, 0.5, 0.5)[0, 0] == 15
    R = rotation6d(torch.randn(10, 6))
    assert torch.allclose(
        R.transpose(-1, -2) @ R, torch.eye(3).expand(10, 3, 3), atol=1e-5
    )
    assert (torch.linalg.det(R) > 0.999).all()


def test_real_cross_view_targets_gradients_and_renderer(tmp_path):
    torch.set_num_threads(2)
    generate(tmp_path, 6, h=32, w=48, views=3)
    rgb, K, T, d = load_supervised(tmp_path / "730000")
    model = Pipeline(32, 8, 1)
    loss, metrics = jepa_loss(model, rgb, K, T, d)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.jepa.target.parameters())
    assert model.jepa.query.weight.grad.abs().sum() > 0
    before = next(model.jepa.target.parameters()).clone()
    with torch.no_grad():
        next(model.jepa.encoder.parameters()).add_(0.1)
    model.jepa.ema()
    assert not torch.equal(before, next(model.jepa.target.parameters()))
    model.zero_grad()
    loss = rendering_loss(model, rgb, K, T, d)
    loss.backward()
    assert model.field.density[0].weight.grad is not None
    recon, mu, lv = model.vae(rgb)
    assert recon.shape == rgb.shape
    assert torch.isfinite(lv).all()
    # Inference loader continues to work after all supervision and held-out files vanish.
    import shutil

    shutil.rmtree(tmp_path / "730000" / "labels")
    shutil.rmtree(tmp_path / "730000" / "heldout")
    assert load_input(tmp_path / "730000")[0].shape == rgb.shape
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len({s["seed"] for s in manifest["scenes"]}) == 6


def test_bilinear_coordinate_gradients_and_anchor_gauge(tmp_path):
    from backend.research.optimization import optimize_geometry

    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).requires_grad_()
    xy = torch.tensor([[1.25, 1.5]], requires_grad=True)
    sample(image, xy).sum().backward()
    assert torch.allclose(xy.grad, torch.tensor([[1.0, 4.0]]))
    generate(tmp_path, 6, h=32, w=48, views=3)
    rgb, K, T, d = load_supervised(tmp_path / "730000")
    out, depth, history = optimize_geometry(rgb, K, T, d, steps=2)
    assert torch.allclose(out[1], T[1])
    assert torch.equal(depth[1], d[1])
    assert all(np.isfinite(history))


def test_checkpoint_resume_and_label_free_reconstruction(tmp_path):
    from backend.research.train import train
    from backend.research.inference import reconstruct
    import shutil

    generate(tmp_path / "data", 6, h=32, w=48, views=3)
    checkpoint = train(tmp_path / "data", tmp_path / "model", steps=1, dim=32, layers=1)
    first = torch.load(checkpoint, weights_only=False)
    resumed = train(
        tmp_path / "data",
        tmp_path / "resume",
        steps=1,
        dim=32,
        layers=1,
        resume=checkpoint,
    )
    second = torch.load(resumed, weights_only=False)
    assert first["completed"] == second["completed"]
    assert all(
        torch.equal(first["model"][k], second["model"][k]) for k in first["model"]
    )
    scene = tmp_path / "data" / "730005"
    shutil.rmtree(scene / "labels")
    shutil.rmtree(scene / "heldout")
    report = reconstruct(
        scene,
        checkpoint,
        tmp_path / "result",
        optimize=0,
        global_opt=False,
        width=48,
        samples=8,
    )
    assert report["input_count"] == 3
    assert (tmp_path / "result" / "panorama.png").exists()


def test_empty_overlap_and_invalid_calibration(tmp_path):
    import pytest

    generate(tmp_path, 6, h=32, w=48, views=3)
    rgb, K, T, d = load_supervised(tmp_path / "730000")
    with pytest.raises(ValueError, match="No valid"):
        jepa_loss(Pipeline(32, 8, 1), rgb, K, T, d * 0)
    metadata = tmp_path / "730000" / "intrinsics.json"
    payload = json.loads(metadata.read_text())
    payload["K"][0][0][0] = 0
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid intrinsics"):
        load_input(tmp_path / "730000")


def test_guided_sampling_and_cache_subset_gradients():
    from backend.research.render import depth_guided_steps, rays
    from backend.research.overfit import make_cache, cached_render

    K = torch.tensor([[[20.0, 0, 7.5], [0, 20.0, 7.5], [0, 0, 1.0]]])
    T = torch.eye(4)[None]
    depth = torch.ones(1, 16, 16)
    xy = torch.tensor([[6.0, 7.0], [8.0, 8.0], [9.0, 6.0]])
    origins, directions = rays(K[0], T[0], 16, 16, xy)
    steps = depth_guided_steps(origins, directions, K, T, depth, samples=16)
    assert torch.allclose(steps.mean(-1), torch.ones(3), atol=1e-4)
    assert (steps[:, 1:] >= steps[:, :-1]).all()
    rgb = torch.rand(1, 3, 16, 16)
    model = Pipeline(32, 4, 1, depth_refinement=True, field_arch="fourier")
    with torch.no_grad():
        _, structure, appearance, _ = model.features(rgb)
    cache = make_cache(
        model.field,
        rgb,
        K,
        T,
        structure,
        appearance,
        depth,
        K[0],
        T[0],
        xy,
        16,
        sampling="depth_guided",
    )
    full = cached_render(model.field, cache)[0]
    ids = torch.tensor([2, 0])
    subset = cached_render(model.field, cache, ids)[0]
    assert torch.allclose(subset, full[ids], atol=1e-6)
    subset.sum().backward()
    assert model.field.color[0].weight.grad.abs().sum() > 0


def test_legacy_depth_checkpoint_numerical_compatibility():
    import torch.nn.functional as F

    model = Pipeline(32, 8, 1)
    tokens = torch.randn(2, 4 * 6, 32)
    expected = F.interpolate(
        (F.softplus(model.pose.depth(tokens)) + 0.05)
        .transpose(1, 2)
        .reshape(2, 1, 4, 6),
        size=(32, 48),
        mode="bilinear",
        align_corners=True,
    )[:, 0]
    assert torch.equal(model.pose.depths(tokens, 4, 6, (32, 48)), expected)


def test_overfit_reads_heldout_only_after_all_fitting(tmp_path, monkeypatch):
    import backend.research.overfit as experiment
    from PIL import Image

    generate(tmp_path / "data", 6, h=32, w=48, views=3, exposure_jitter=False)
    completed = []
    fit = experiment.fit_field

    def tracked_fit(*args, **kwargs):
        result = fit(*args, **kwargs)
        completed.append(args[5])
        return result

    read_np = np.load
    read_image = Image.open

    def assert_after_fitting(path):
        if "/heldout/" in str(path):
            assert len(completed) == 3

    def load_np(path, *args, **kwargs):
        assert_after_fitting(path)
        return read_np(path, *args, **kwargs)

    def load_image(path, *args, **kwargs):
        assert_after_fitting(path)
        return read_image(path, *args, **kwargs)

    monkeypatch.setattr(experiment, "fit_field", tracked_fit)
    monkeypatch.setattr(np, "load", load_np)
    monkeypatch.setattr(Image, "open", load_image)
    report = experiment.overfit(
        tmp_path / "data" / "730000",
        tmp_path / "result",
        encoder_steps=1,
        geometry_steps=1,
        vae_steps=1,
        render_steps=1,
        samples=8,
    )
    assert len(completed) == 3
    assert report["heldout_used_for_training"] is False
    assert report["fields"]["fourier_predicted"]["heldout_mask_fraction"] > 0
    from backend.research.inference import load_model

    loaded, _ = load_model(tmp_path / "result" / "model.pt")
    assert loaded.config["depth_refinement"]
    from backend.research.orthofit import run

    adapted = run(
        tmp_path / "data" / "730000",
        tmp_path / "result" / "model.pt",
        tmp_path / "adapted",
        steps=1,
        width=16,
        adapt_steps=1,
        freeze_density=True,
    )
    assert len(adapted["adaptation"]) == 2
    assert adapted["material_style"] == "solid_labels"
    assert adapted["gt_used_for_training"] is True
    assert adapted["freeze_density"] is True
    after = torch.load(tmp_path / "adapted" / "model.pt", weights_only=False)["model"]
    before = loaded.state_dict()
    for key in before:
        if key.startswith("field.density"):
            assert torch.equal(before[key], after[key])
    assert torch.equal(before["field.residual"][0], after["field.residual"][0])


def test_orthographic_parallel_rays_and_depth_independent_scale():
    from backend.research.render import rays

    K = torch.tensor([[100.0, 0.0, 20.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    T = torch.eye(4)
    xy = torch.tensor([[0.0, 0.0], [40.0, 20.0]])
    o, d = rays(K, T, 21, 41, xy, "orthographic")
    assert torch.allclose(d, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    assert not torch.allclose(o[0], o[1])
    for depth in (1.0, 3.0):
        points = o + d * depth
        reprojection = points[:, :2] * K.diag()[:2] + K[:2, 2]
        assert torch.allclose(reprojection, xy, atol=1e-5)
    from backend.research.inference import reconstruct
    import inspect

    assert (
        inspect.signature(reconstruct).parameters["projection"].default
        == "orthographic"
    )


def test_synthetic_orthographic_gt_depth_and_occlusion(tmp_path):
    from backend.research.data import render, orthographic_gt
    import numpy as np

    K = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]], np.float32)
    T = np.eye(4, dtype=np.float32)
    box = (np.array([-0.4, -0.4, 1.0]), np.array([0.4, 0.4, 1.4]), [1.0, 0.0, 0.0])
    near, dn = render([box], K, T, 21, 21, projection="orthographic")
    distant = (box[0] + [0, 0, 2], box[1] + [0, 0, 2], box[2])
    far, df = render([distant], K, T, 21, 21, projection="orthographic")
    assert np.array_equal(dn > 0, df > 0)
    assert np.allclose(df[df > 0] - dn[dn > 0], 2)
    combined, dc = render([distant, box], K, T, 21, 21, projection="orthographic")
    assert np.array_equal(dc, dn)
    assert np.array_equal(combined, near)
    image, k, t, d = orthographic_gt(731000, tmp_path, 64)
    assert image.shape[:2] == d.shape and (d > 0).any()
    assert (tmp_path / "geometry.npz").exists()


def test_opaque_surface_no_foreground_background_blending():
    from backend.research.render import opaque_weights

    weights = torch.tensor([[0.6, 0.3, 0.1], [0.0, 0.0, 0.0]], requires_grad=True)
    opaque = opaque_weights(weights)
    assert torch.allclose(opaque[0], torch.tensor([1.0, 0.0, 0.0]))
    assert opaque[1].sum() == 0
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = opaque @ colors
    assert torch.allclose(result[0], torch.tensor([1.0, 0.0, 0.0]))
    result[:, 0].sum().backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert weights.grad[0].abs().sum() > 0


def test_opaque_checkpoint_config():
    from backend.research.model import Pipeline

    model = Pipeline(dim=32, patch=8, layers=1, opaque_surface=True)
    restored = Pipeline(**model.config)
    restored.load_state_dict(model.state_dict())
    assert restored.field.opaque_surface
    assert restored.config["opaque_surface"] is True


def test_hybrid_sampling_covers_recessed_geometry():
    from backend.research.render import hybrid_steps

    K = torch.eye(3)[None]
    T = torch.eye(4)[None]
    depth = torch.ones(1, 4, 4)
    o = torch.zeros(2, 3)
    d = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]])
    t = hybrid_steps(o, d, K, T, depth, 64)
    assert t.shape == (2, 64)
    assert torch.all(t[:, 1:] >= t[:, :-1])
    assert torch.allclose(t[:, 0], torch.full((2,), 0.15))
    assert torch.allclose(t[:, -1], torch.full((2,), 3.5))
    assert ((t > 0.94) & (t < 1.06)).sum() > 20


def test_solid_gt_is_opaque_and_object_local():
    from backend.research.data import render
    import numpy as np

    K = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]], np.float32)
    T = np.eye(4, dtype=np.float32)
    front = (np.array([-0.4, -0.4, 1.0]), np.array([0.4, 0.4, 1.4]), [0.8, 0.2, 0.1])
    rear = (np.array([-1.0, -1.0, 2.0]), np.array([1.0, 1.0, 2.2]), [0.2, 0.4, 0.9])
    a, depth = render(
        [front], K, T, 41, 41, projection="orthographic", material_style="solid_labels"
    )
    b, _ = render(
        [rear, front],
        K,
        T,
        41,
        41,
        projection="orthographic",
        material_style="solid_labels",
    )
    assert np.array_equal(a[depth > 0], b[depth > 0])
    # Opaque paint on the bottom of the product is spatially constant.
    patch = a[14:17, 15:25]
    assert np.unique(patch.reshape(-1, 3), axis=0).shape[0] == 1


def test_occlusion_aware_rgb_does_not_blend_depth_layers():
    from backend.research.geometry import sample_visible_rgb

    rgb = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    depth = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    xy = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    color, delta, coverage = sample_visible_rgb(
        rgb, xy, depth, torch.tensor([1.0, 2.0, 3.0])
    )
    assert torch.allclose(
        color, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert coverage[2] == 0
    assert torch.allclose(coverage[:2], torch.tensor([0.5, 0.5]))
    (color.sum() + coverage.sum()).backward()
    assert torch.isfinite(xy.grad).all()


def test_color_visibility_preserves_density_support():
    from backend.research.model import Field

    field = Field(8)
    rgb = torch.rand(1, 3, 4, 4)
    K = torch.eye(3)[None]
    T = torch.eye(4)[None]
    structure = torch.rand(1, 8, 2, 2)
    appearance = torch.rand(1, 16, 2, 2)
    depth = torch.ones(1, 4, 4)
    depth[:, :, 2:] = 2
    points = torch.tensor([[0.5, 0.5, 1.5], [1.0, 1.0, 1.0]])
    legacy = field.query(points, rgb, K, T, structure, appearance, depth)
    field.occlusion_aware = True
    visible = field.query(points, rgb, K, T, structure, appearance, depth)
    assert torch.allclose(legacy[-1], visible[-1])
    assert torch.allclose(legacy[-2], visible[-2])


def test_deep_field_warm_start_preserves_output_and_gradients():
    from backend.research.model import FourierField, DeepFourierField, Pipeline

    torch.manual_seed(2)
    baseline = FourierField(8)
    deeper = DeepFourierField(8)
    missing, extra = deeper.load_state_dict(baseline.state_dict(), strict=False)
    assert not extra and all(k.startswith("color_refinement.") for k in missing)
    points = torch.randn(5, 3)
    directions = torch.randn(5, 3)
    features = (
        torch.randn(5, 8),
        torch.randn(5, 16),
        torch.rand(5, 3),
        torch.randn(5, 1),
        torch.ones(5),
    )
    first = baseline.evaluate(points, directions, features)
    second = deeper.evaluate(points, directions, features)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    second[1].sum().backward()
    assert deeper.color_refinement[-1].weight.grad.abs().sum() > 0
    model = Pipeline(dim=8, patch=8, layers=1, field_arch="deep_fourier")
    restored = Pipeline(**model.config)
    restored.load_state_dict(model.state_dict())
    assert isinstance(restored.field, DeepFourierField)


def test_selected_surface_color_matches_full_opaque_render():
    from backend.research.model import DeepFourierField
    from backend.research.overfit import make_cache, cached_render, cache_surface_depth

    field = DeepFourierField(8)
    field.opaque_surface = True
    rgb = torch.rand(2, 3, 4, 4)
    K = torch.tensor([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]).repeat(
        2, 1, 1
    )
    T = torch.eye(4).repeat(2, 1, 1)
    depth = torch.ones(2, 4, 4)
    structure = torch.rand(2, 8, 2, 2)
    appearance = torch.rand(2, 16, 2, 2)
    xy = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    cache = make_cache(
        field,
        rgb,
        K,
        T,
        structure,
        appearance,
        depth,
        K[0],
        T[0],
        xy,
        samples=8,
        sampling="depth_guided",
    )
    full, _, _ = cached_render(field, cache)
    surface = cache_surface_depth(field, cache)
    chosen = (cache["steps"] - surface[:, None]).abs().argmin(-1)
    rows = torch.arange(2)
    features = tuple(v[rows, chosen] for v in cache["features"])
    features = (*features[:4], features[4][:, 0])
    _, selected, _ = field.evaluate(
        cache["points"][rows, chosen], cache["directions"][rows, chosen], features
    )
    assert torch.allclose(full, selected, atol=1e-6)


def test_fast_opaque_inference_matches_differentiable_render():
    from backend.research.model import Pipeline
    from backend.research.render import render_rays

    model = Pipeline(
        dim=8,
        patch=8,
        layers=1,
        field_arch="deep_fourier",
        opaque_surface=True,
        ray_sampling="hybrid",
    )
    rgb = torch.rand(2, 3, 8, 8)
    K = torch.tensor([[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]]).repeat(
        2, 1, 1
    )
    T = torch.eye(4).repeat(2, 1, 1)
    structure = torch.rand(2, 8, 1, 1)
    appearance = torch.rand(2, 16, 1, 1)
    depth = torch.ones(2, 8, 8)
    args = (
        model,
        rgb,
        K,
        T,
        structure,
        appearance,
        depth,
        K[0],
        T[0],
        torch.tensor([[3.0, 3.0], [4.0, 4.0]]),
    )
    expected = render_rays(*args, samples=8)
    with torch.no_grad():
        actual = render_rays(*args, samples=8)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(expected, actual))


def test_multiview_feature_and_pose_batching_matches_unbatched():
    from backend.research.model import Pipeline
    from backend.research.optimization import predict_cameras

    model = Pipeline(dim=8, patch=8, layers=1).eval()
    rgb = torch.rand(7, 3, 32, 32)
    with torch.no_grad():
        full = model.features(rgb)
        batched = model.features_batched(rgb, 2)
        assert all(torch.allclose(a, b, atol=2e-6) for a, b in zip(full, batched))
        serial, serial_edges = predict_cameras(model, full[0], pair_batch=1)
        grouped, grouped_edges = predict_cameras(model, full[0], pair_batch=8)
    assert len(grouped_edges) == 11
    assert torch.allclose(serial, grouped, atol=2e-6)
    assert all(
        (a, b) == (c, d) and torch.allclose(e, f, atol=2e-6)
        for (a, b, e, _), (c, d, f, _) in zip(serial_edges, grouped_edges)
    )


def test_multiview_subsets_share_anchor_and_are_nested():
    from backend.research.multiview import indices

    a, b, c = (indices(n, 60) for n in (10, 30, 60))
    assert set(a) <= set(b) <= set(c)
    assert a[5] == b[15] == c[30] == 30


def test_native_rgb_multires_features_have_separate_spatial_sizes():
    from backend.research.model import Pipeline

    model = Pipeline(dim=8, patch=8, layers=1).eval()
    native = torch.rand(3, 3, 64, 80)
    with torch.no_grad():
        tokens, structure, appearance, depth = model.features_multires(
            native, (32, 40), 1
        )
    assert tokens.shape == (3, 20, 8)
    assert structure.shape[-2:] == (4, 5)
    assert appearance.shape[-2:] == (8, 10)
    assert depth.shape == (3, 64, 80)
    assert torch.isfinite(depth).all() and (depth > 0).all()
    with torch.no_grad():
        a = model.features(native)
        b = model.features_multires(native, (64, 80), 1)
    assert all(torch.allclose(x, y, atol=2e-6) for x, y in zip(a, b))


def test_box_culling_preserves_opaque_gt_pixels_and_depth():
    from backend.research.data import render, shelf_scene, camera
    import numpy as np

    boxes, _, _ = shelf_scene(731000)
    K = np.array([[60.0, 0.0, 31.5], [0.0, 60.0, 23.5], [0.0, 0.0, 1.0]], np.float32)
    T = camera(np.array([0.2, 1.0, -2.6]), np.array([0.0, 1.0, 0.0]))
    for projection in ("perspective", "orthographic"):
        full = render(
            boxes,
            K,
            T,
            48,
            64,
            projection=projection,
            material_style="solid_labels",
            crop_boxes=False,
        )
        fast = render(
            boxes,
            K,
            T,
            48,
            64,
            projection=projection,
            material_style="solid_labels",
            crop_boxes=True,
        )
        assert all(np.array_equal(a, b) for a, b in zip(full, fast))


def test_native_generation_rerenders_and_scales_pixel_centers(tmp_path):
    from backend.research.data import generate, render, shelf_scene
    from backend.research.native_data import generate_native
    import numpy as np, json
    from PIL import Image

    source = tmp_path / "source"
    generate(source, scenes=6, h=32, w=48, views=2, exposure_jitter=False)
    output = tmp_path / "native" / "730000"
    generated = generate_native(
        source / "730000", output, width=96, height=64, target_width=48
    )
    assert generated["native_resolution"] == [96, 64]
    meta = json.loads((output / "intrinsics.json").read_text())
    K = np.array(meta["K"][0])
    assert K[0, 2] == 47.5 and K[1, 2] == 31.5
    assert np.array_equal(
        np.load(output / "labels/geometry.npz")["T"],
        np.load(source / "730000/labels/geometry.npz")["T"],
    )
    boxes, _, _ = shelf_scene(730000)
    T = np.load(output / "labels/geometry.npz")["T"][0]
    expected, _ = render(
        boxes,
        K.astype(np.float32),
        T,
        64,
        96,
        material_style="solid_labels",
        crop_boxes=False,
    )
    assert np.array_equal(expected, np.asarray(Image.open(output / "rgb/0000.png")))


def test_relational_pose_head_gradients_and_checkpoint(tmp_path):
    from backend.research.poseadapt import supervised_loss, pairs_for
    from backend.research.inference import load_model

    model = Pipeline(32, 4, 1, pose_arch="relational", pose_tokens=12)
    a, b = torch.randn(4, 48, 32), torch.randn(4, 48, 32)
    descriptor = model.pose.descriptor(a, b)
    assert descriptor.shape == (4, 128)
    assert not torch.allclose(descriptor, model.pose.descriptor(b, a))
    target = torch.eye(4)[None].repeat(4, 1, 1)
    target[:, 0, 3] = torch.tensor([0.01, 0.03, 0.08, 0.2])
    loss = supervised_loss(model.pose.decode(descriptor), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.pose.head[-1].weight.grad.abs().sum() > 0
    assert (0, 1) in pairs_for(20) and (1, 0) in pairs_for(20)
    assert (0, 16) in pairs_for(20)
    file = tmp_path / "pose.pt"
    torch.save(
        dict(
            version=1,
            protocol="single-scene-overfit-v1",
            completed={
                s: 1 for s in ("jepa", "pose", "vae", "render_gt", "render_pred")
            },
            config=model.config,
            model=model.state_dict(),
        ),
        file,
    )
    restored, _ = load_model(file)
    with torch.no_grad():
        expected = model.pose.pair(a, b)
        actual = restored.pose.pair(a, b)
    for x, y in zip(expected, actual):
        assert torch.allclose(x, y)
    assert torch.allclose(
        actual[0].transpose(-1, -2) @ actual[0], torch.eye(3).expand(4, 3, 3), atol=1e-5
    )


def test_pose_graph_gauge_legal_rotation_and_disconnection():
    import pytest
    from backend.research.optimization import solve_pose_graph

    torch.manual_seed(29)
    truth = torch.eye(4)[None].repeat(6, 1, 1)
    truth[:, :3, :3] = rotation6d(torch.randn(6, 6))
    truth[:, :3, 3] = torch.randn(6, 3)
    edges = [
        (i, j, truth[j] @ torch.linalg.inv(truth[i]), torch.tensor(0.1))
        for i in range(6)
        for j in range(i + 1, min(i + 3, 6))
    ]
    pred = solve_pose_graph(6, edges, 3)
    expected = truth @ torch.linalg.inv(truth[3])
    assert torch.allclose(pred, expected, atol=1e-5)
    assert torch.equal(pred[3], torch.eye(4))
    assert (torch.linalg.det(pred[:, :3, :3]) > 0.999).all()
    with pytest.raises(ValueError, match="Disconnected"):
        solve_pose_graph(6, edges[:1], 3)


def test_color_residual_control_keeps_geometry_and_native_rgb():
    from backend.research.model import DeepFourierField

    field = DeepFourierField(32)
    points = torch.randn(7, 3)
    direction = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1)
    base = torch.rand(7, 3)
    features = (
        torch.randn(7, 32),
        torch.randn(7, 16),
        base,
        torch.randn(7, 1),
        torch.rand(7),
    )
    sigma, _, support = field.evaluate(points, direction, features)
    field.color_residual_limit = 0
    s0, color, support0 = field.evaluate(points, direction, features)
    assert torch.equal(color, base)
    assert torch.equal(sigma, s0) and torch.equal(support, support0)
    field.color_residual_limit = 0.05
    _, color, _ = field.evaluate(points, direction, features)
    assert (color - base).abs().max() <= 0.050001


def test_depth_adaptation_positive_gradient_and_empty_mask():
    import pytest
    from backend.research.depthadapt import depth_loss

    model = Pipeline(32, 4, 1, depth_refinement=True)
    model.requires_grad_(False)
    model.pose.depth.requires_grad_(True)
    model.pose.depth_refiner.requires_grad_(True)
    tokens = torch.randn(2, 48, 32)
    prediction = model.pose.depths(tokens, 6, 8, (24, 32))
    truth = torch.ones_like(prediction)
    truth[:, :, 16:] = 1.4
    truth[:, :2] = 0
    loss = depth_loss(prediction, truth)
    loss.backward()
    assert torch.isfinite(loss) and (prediction > 0).all()
    assert model.pose.depth.weight.grad.abs().sum() > 0
    assert model.pose.depth_refiner[2].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.pose.head.parameters())
    with pytest.raises(ValueError, match="Empty"):
        depth_loss(prediction, torch.zeros_like(truth))


def test_surface_control_opaque_nearest_and_missing_support():
    from backend.research.surface_control import render_surface

    rgb = torch.zeros(2, 3, 4, 4)
    rgb[0, 0] = 1
    rgb[1, 2] = 1
    k = torch.tensor([[2.0, 0, 1.5], [0, 2.0, 1.5], [0, 0, 1.0]])[None].repeat(2, 1, 1)
    t = torch.eye(4)[None].repeat(2, 1, 1)
    d = torch.ones(2, 4, 4)
    d[1] *= 2
    target = torch.tensor([[2.0, 0, 1.5], [0, 2.0, 1.5], [0, 0, 1.0]])
    image, mask = render_surface(rgb, k, t, d, target, torch.eye(4), 4, 4, stride=1)
    assert (image[mask][:, 0] == 1).all()
    assert (image[mask][:, 2] == 0).all()
    empty, valid = render_surface(
        rgb, k, t, d * 0, target, torch.eye(4), 4, 4, stride=1
    )
    assert not valid.any() and not empty.any()


def test_cross_view_depth_consistency_gradients_and_occlusion():
    from backend.research.depthadapt import consistency_loss, depth_loss

    truth = torch.ones(2, 24, 32)
    k = torch.tensor([[30.0, 0, 15.5], [0, 30.0, 11.5], [0, 0, 1.0]])[None].repeat(
        2, 1, 1
    )
    t = torch.eye(4)[None].repeat(2, 1, 1)
    t[1, 0, 3] = 0.05
    pred = truth.clone()
    pred[0] *= 1.15
    pred[1] *= 0.95
    pred.requires_grad_()
    loss = consistency_loss(pred, truth, k, t, stride=4)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert pred.grad[0].abs().sum() > 0 and pred.grad[1].abs().sum() > 0
    blocked = truth.clone()
    blocked[1] *= 2
    assert consistency_loss(pred, blocked, k, t, stride=4) == 0
    assert depth_loss(pred, truth, edge_weight=1).isfinite()


def test_low_band_field_and_geometry_appearance_restore(tmp_path):
    from backend.research.inference import load_model

    m = Pipeline(
        32,
        4,
        1,
        field_arch="fourier",
        field_bands=4,
        appearance_geometry=True,
        color_residual_limit=0.15,
        opaque_surface=True,
    )
    rgb = torch.rand(2, 3, 48, 64)
    with torch.no_grad():
        tokens, structure, appearance, depth = m.features_multires(rgb, (24, 32), 1)
    assert appearance.shape[-2:] == (3, 4) and depth.shape[-2:] == (48, 64)
    path = tmp_path / "band.pt"
    torch.save(
        dict(
            version=1,
            protocol="single-scene-overfit-v1",
            completed={
                s: 1 for s in ("jepa", "pose", "vae", "render_gt", "render_pred")
            },
            config=m.config,
            model=m.state_dict(),
        ),
        path,
    )
    restored, _ = load_model(path)
    assert restored.field.bands == 4 and restored.config["appearance_geometry"]
    points = torch.randn(8, 3)
    directions = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    features = (
        torch.randn(8, 32),
        torch.randn(8, 16),
        torch.rand(8, 3),
        torch.randn(8, 1),
        torch.rand(8),
    )
    sigma, color, _ = restored.field.evaluate(points, directions, features)
    (sigma.mean() + color.mean()).backward()
    assert restored.field.density[0].weight.grad.abs().sum() > 0
    assert restored.field.color[0].weight.grad.abs().sum() > 0
    assert (color - features[2]).abs().max() <= 0.150001


def test_fixed_gt_metrics_perfect_and_corrupted_edges():
    import pytest
    from backend.research.metrics import image_metrics

    truth = np.zeros((24, 32, 3), dtype=np.uint8)
    truth[:, 16:] = 255
    depth = np.ones((24, 32), dtype=np.float32)
    depth[:, 16:] = 1.3
    perfect = image_metrics(truth, truth, depth)
    assert abs(perfect["ssim_gt_valid"] - 1) < 1e-10
    bad = truth.copy()
    bad[:, 14:18] = 127
    measured = image_metrics(bad, truth, depth)
    assert measured["ssim_gt_valid"] < 1
    assert measured["edge_psnr_png"] < measured["interior_psnr_png"]
    with pytest.raises(ValueError, match="empty"):
        image_metrics(truth, truth, depth * 0)


def test_consensus_fusion_observed_color_without_changing_geometry():
    from backend.research.model import FourierField

    field = FourierField(32)
    rgb = torch.zeros(3, 3, 4, 4)
    rgb[:2, 0] = 1
    rgb[2, 2] = 1
    k = torch.tensor([[2.0, 0, 1.5], [0, 2.0, 1.5], [0, 0, 1.0]])[None].repeat(3, 1, 1)
    t = torch.eye(4)[None].repeat(3, 1, 1)
    d = torch.ones(3, 4, 4)
    structure = torch.randn(3, 32, 2, 2)
    appearance = torch.randn(3, 16, 2, 2)
    points = torch.tensor([[0.0, 0.0, 1.0], [100.0, 0.0, 1.0]])
    weighted = field.query(points, rgb, k, t, structure, appearance, d)
    field.color_fusion = "consensus"
    consensus = field.query(points, rgb, k, t, structure, appearance, d)
    assert torch.allclose(weighted[2][0], torch.tensor([2 / 3, 0.0, 1 / 3]))
    assert torch.equal(consensus[2][0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(consensus[2][1], torch.zeros(3))
    for index in (0, 1, 3, 4):
        assert torch.equal(weighted[index], consensus[index])
    assert torch.equal(
        field.geometry_sigma(points, weighted), field.geometry_sigma(points, consensus)
    )


def test_multiscene_field_training_matches_cached_opaque_render_and_gradients():
    from backend.research.renderadapt import ray_prediction
    from backend.research.overfit import cached_render
    from backend.research.model import FourierField

    field = FourierField(32, bands=4)
    field.opaque_surface = True
    field.color_residual_limit = 0.15
    count, steps = 6, 16
    times = torch.linspace(0.2, 2.0, steps)[None].repeat(count, 1)
    cache = dict(
        points=torch.randn(count, steps, 3),
        directions=torch.randn(count, steps, 3),
        ray_dirs=torch.randn(count, 3),
        steps=times,
        features=(
            torch.randn(count, steps, 32),
            torch.randn(count, steps, 16),
            torch.rand(count, steps, 3),
            torch.randn(count, steps, 1),
            torch.rand(count, steps, 1),
        ),
    )
    ids = torch.arange(count)
    actual, raw, t, support = ray_prediction(field, cache, ids)
    expected, _, _ = cached_render(field, cache, ids)
    assert torch.allclose(actual, expected, atol=1e-6)
    probability = raw / raw.sum(-1, keepdim=True).clamp_min(1e-8)
    loss = (
        actual.square().mean()
        - (probability[:, steps // 2].clamp_min(1e-8).log()).mean()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert field.density[0].weight.grad.abs().sum() > 0
    assert field.color[0].weight.grad.abs().sum() > 0


def test_projected_patch_and_vae_feature_pixel_centers():
    from backend.research.model import FourierField

    field = FourierField(32)
    field.feature_center_mapping = True
    xy = torch.tensor([[1.5, 1.5], [5.5, 5.5], [253.5, 189.5]])
    assert torch.allclose(
        field.feature_coordinates(xy, (48, 64), (192, 256)),
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [63.0, 47.0]]),
    )
    xy = torch.tensor([[3.5, 3.5], [11.5, 11.5]])
    assert torch.allclose(
        field.feature_coordinates(xy, (24, 32), (192, 256)),
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    )
    native = torch.tensor([[9.5, 9.5], [29.5, 29.5]])
    assert torch.allclose(
        field.feature_coordinates(native, (48, 64), (960, 1280)),
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    )
    boundary = field.feature_coordinates(
        torch.tensor([[0.0, 0.0], [1279.0, 959.0]]), (48, 64), (960, 1280)
    )
    assert torch.equal(boundary, torch.tensor([[0.0, 0.0], [63.0, 47.0]]))
    xy = torch.tensor([[200.0, 100.0]], requires_grad=True)
    field.feature_coordinates(xy, (48, 64), (960, 1280)).sum().backward()
    assert torch.allclose(xy.grad, torch.full_like(xy, 0.05))


def test_long_shelf_keeps_product_size_and_extends_capture(tmp_path):
    import json
    import numpy as np
    from backend.research.data import generate, shelf_scene
    from backend.research.native_data import generate_native
    from PIL import Image

    short, _, _ = shelf_scene(11, 3.86)
    long, _, _ = shelf_scene(11, 10.86)
    short_products = [b for b in short if np.ptp(b[2]) > 0.3]
    long_products = [b for b in long if np.ptp(b[2]) > 0.3]
    assert len(long_products) > len(short_products) * 2
    assert (
        abs(
            (short_products[0][1][0] - short_products[0][0][0])
            - (long_products[0][1][0] - long_products[0][0][0])
        )
        < 0.01
    )
    generate(
        tmp_path / "source",
        scenes=6,
        seed=11,
        h=24,
        w=32,
        views=4,
        exposure_jitter=False,
        capture_step=0.14,
        target_width=32,
    )
    scene = tmp_path / "source/11"
    poses = np.load(scene / "labels/geometry.npz")["T"]
    centers = -(poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, 3, None])[:, :, 0]
    assert np.allclose(np.diff(centers[:, 0]), 0.14)
    meta = json.loads((scene / "scene.json").read_text())
    assert np.isclose(meta["shelf_width"], 2.6 + 3 * 0.14)
    generate_native(scene, tmp_path / "native/11", width=32, height=24, target_width=32)
    assert np.array_equal(
        np.asarray(Image.open(scene / "rgb/0000.png")),
        np.asarray(Image.open(tmp_path / "native/11/rgb/0000.png")),
    )
    assert np.array_equal(
        np.asarray(Image.open(scene / "orthographic/rgb.png")),
        np.asarray(Image.open(tmp_path / "native/11/orthographic/rgb.png")),
    )


def test_local_column_render_preserves_pixels_and_uses_nearby_sources(monkeypatch):
    import backend.research.render as rendering
    from backend.research.render import select_local_sources, render_image

    K = torch.eye(3)
    T = torch.eye(4)
    cameras = T.repeat(5, 1, 1)
    cameras[:, 0, 3] = -torch.arange(5.0)
    ids = select_local_sources(torch.tensor([[3.0, 0.0], [3.0, 1.0]]), K, T, cameras, 2)
    assert ids[0] == 3

    def fake(
        model,
        rgb,
        intrinsics,
        poses,
        structure,
        appearance,
        depth,
        target_K,
        target_T,
        xy,
        samples,
        use_vae,
        projection,
    ):
        assert len(rgb) == 2
        return (
            torch.cat((xy, xy[:, :1] * 0), -1),
            torch.ones(len(xy)),
            torch.ones(len(xy)),
        )

    monkeypatch.setattr(rendering, "render_rays", fake)
    rgb = torch.zeros(5, 3, 2, 4)
    output, mask = render_image(
        None,
        rgb,
        K.repeat(5, 1, 1),
        cameras,
        rgb,
        rgb,
        torch.ones(5, 2, 4),
        K,
        T,
        2,
        4,
        chunk=2,
        projection="orthographic",
        local_sources=2,
    )
    assert torch.equal(output[..., 0], torch.arange(4.0).repeat(2, 1))
    assert torch.equal(output[..., 1], torch.arange(2.0)[:, None].repeat(1, 4))
    assert mask.all()


def test_homography_front_plane_adapter_requires_no_target_rgb():
    import numpy as np
    import cv2
    from backend.research.homography_benchmark import plane_adapter

    K = np.array([[100, 0, 20], [0, 100, 15], [0, 0, 1]], np.float32)
    Ko = np.array([[50, 0, 20], [0, 50, 15], [0, 0, 1]], np.float32)
    H = plane_adapter(K, Ko, np.eye(4), 2.0, 30, 40)
    # A fronto-parallel plane at z=2 exactly maps these calibrated cameras.
    assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-5)
    xy = np.array([[[3.0, 4.0], [23.0, 12.0]]], np.float32)
    assert np.allclose(cv2.perspectiveTransform(xy, H), xy)
    import pytest

    with pytest.raises(ValueError):
        plane_adapter(K, Ko, np.eye(4), -1.0, 30, 40)


def test_local_training_cache_matches_explicit_source_subset_and_has_gradients():
    from backend.research.overfit import make_cache, cached_render
    from backend.research.render import select_local_sources

    model = Pipeline(32, 4, 1, field_arch="fourier")
    rgb = torch.rand(3, 3, 16, 16)
    K = torch.tensor([[20.0, 0, 7.5], [0, 20.0, 7.5], [0, 0, 1.0]])[None].repeat(
        3, 1, 1
    )
    T = torch.eye(4)[None].repeat(3, 1, 1)
    T[:, 0, 3] = torch.tensor([-0.2, 0.0, 0.2])
    target = torch.eye(4)
    xy = torch.tensor([[7.0, 7.0], [8.0, 8.0]])
    with torch.no_grad():
        _, structure, appearance, _ = model.features(rgb)
        depth = torch.ones(3, 16, 16)
        ids = select_local_sources(xy, K[0], target, T, 2)
        local = make_cache(
            model.field,
            rgb,
            K,
            T,
            structure,
            appearance,
            depth,
            K[0],
            target,
            xy,
            samples=16,
            sampling="hybrid",
            projection="orthographic",
            local_sources=2,
        )
        explicit = make_cache(
            model.field,
            rgb[ids],
            K[ids],
            T[ids],
            structure[ids],
            appearance[ids],
            depth[ids],
            K[0],
            target,
            xy,
            samples=16,
            sampling="hybrid",
            projection="orthographic",
        )
    assert torch.equal(local["steps"], explicit["steps"])
    for a, b in zip(local["features"], explicit["features"]):
        assert torch.equal(a, b)
    cached_render(model.field, local)[0].sum().backward()
    assert model.field.color[0].weight.grad.abs().sum() > 0


def test_global_pose_features_identity_gauge_cross_view_gradient_and_restore():
    from backend.research.model import GlobalPoseRefiner
    from backend.research.geometry import rotation6d

    head = GlobalPoseRefiner(32)
    tokens = torch.randn(5, 12, 32)
    statistics = head.statistics(tokens)
    base = torch.eye(4)[None].repeat(5, 1, 1)
    base[:, 0, 3] = torch.linspace(-0.4, 0.4, 5)
    assert torch.allclose(head(statistics, base), base, atol=1e-6)
    # Nonzero output weights expose the actual global attention dependency.
    with torch.no_grad():
        head.output.weight.normal_(std=0.01)
    statistics.requires_grad_()
    out = head(statistics, base)
    assert torch.allclose(out[2], torch.eye(4), atol=1e-6)
    assert torch.allclose(
        out[:, :3, :3].transpose(1, 2) @ out[:, :3, :3],
        torch.eye(3).expand(5, 3, 3),
        atol=1e-5,
    )
    out[0, :3, 3].sum().backward()
    assert statistics.grad[-1].abs().sum() > 0  # distant view affects first pose
    assert head.input.weight.grad.abs().sum() > 0
    model = Pipeline(32, 4, 1, global_pose=True)
    restored = Pipeline(**model.config)
    restored.load_state_dict(model.state_dict())
    assert restored.config["global_pose"]
    assert torch.allclose(
        model.pose.global_sequence(head.statistics(tokens), base),
        restored.pose.global_sequence(head.statistics(tokens), base),
    )


def test_global_homography_shared_basis_fixes_drift_and_anchor(monkeypatch):
    from backend.research import homography_global as hg

    images = [np.zeros((100, 120, 3), np.uint8) for _ in range(5)]
    lines = np.array([[[0.0, y], [119.0, y]] for y in (20.0, 40.0, 60.0)])
    monkeypatch.setattr(hg, "horizontal_lines", lambda image: lines)
    truth = [np.eye(3) for _ in images]
    initial = [np.eye(3) for _ in images]
    for i in range(5):
        truth[i][0, 2] = (i - 2) * 10
        initial[i][:2, 2] = [(i - 2) * 12, (i - 2) * 2]
    p = np.array(
        [
            [30.0, 20.0],
            [40.0, 40.0],
            [50.0, 60.0],
            [60.0, 30.0],
            [70.0, 50.0],
            [80.0, 70.0],
        ]
    )
    edges = [
        dict(i=i, j=i + 1, pi=p - truth[i][:2, 2], pj=p - truth[i + 1][:2, 2])
        for i in range(4)
    ]
    result, report = hg.translation_global(images, initial, 2, edges, 1.0)
    assert np.allclose(result[2], np.eye(3))
    assert (
        report["after"]["global_row_rms_px"]
        < report["before"]["global_row_rms_px"] * 0.01
    )
    for actual, expected in zip(result, truth):
        assert np.allclose(actual, expected, atol=1e-3)


def test_global_homography_missing_global_structure_fails():
    import pytest
    from backend.research.homography_global import optimize

    images = [np.zeros((100, 120, 3), np.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match="Insufficient"):
        optimize(images, [np.eye(3)] * 3, 1, [])


def test_local_cache_groups_columns_and_restores_raster_order(monkeypatch):
    import backend.research.render as render_module
    from backend.research.overfit import make_cache

    model = Pipeline(32, 4, 1, field_arch="fourier")
    rgb = torch.rand(2, 3, 16, 16)
    K = torch.tensor([[20.0, 0, 7.5], [0, 20.0, 7.5], [0, 0, 1.0]])[None].repeat(
        2, 1, 1
    )
    T = torch.eye(4)[None].repeat(2, 1, 1)
    xy = torch.tensor([[float(x), float(y)] for y in range(4) for x in range(4)])
    spans = []

    def select(points, *args):
        spans.append(float(points[:, 0].max() - points[:, 0].min()))
        return torch.tensor([0, 1])

    monkeypatch.setattr(render_module, "select_local_sources", select)
    with torch.no_grad():
        _, structure, appearance, _ = model.features(rgb)
        depth = torch.ones(2, 16, 16)
        args = (model.field, rgb, K, T, structure, appearance, depth, K[0], T[0], xy)
        cache = make_cache(
            *args, samples=8, chunk=4, projection="orthographic", local_sources=2
        )
        full = make_cache(*args, samples=8, chunk=4, projection="orthographic")
    assert spans == [0.0] * 4
    assert torch.equal(cache["points"], full["points"])
    assert all(
        torch.allclose(a, b, atol=1e-6)
        for a, b in zip(cache["features"], full["features"])
    )


def test_nearest_color_uses_closest_supported_view_without_changing_geometry():
    model = Pipeline(32, 4, 1, field_arch="fourier")
    rgb = torch.zeros(2, 3, 16, 16)
    rgb[0, 0] = 1.0
    rgb[1, 2] = 1.0
    K = torch.tensor([[20.0, 0, 7.5], [0, 20.0, 7.5], [0, 0, 1.0]])[None].repeat(
        2, 1, 1
    )
    T = torch.eye(4)[None].repeat(2, 1, 1)
    T[1, 0, 3] = -0.15
    depth = torch.ones(2, 16, 16)
    with torch.no_grad():
        _, structure, appearance, _ = model.features(rgb)
        point = torch.tensor([[0.0, 0.0, 1.0]])
        args = (point, rgb, K, T, structure, appearance, depth)
        weighted = model.field.query(*args)
        model.field.color_fusion = "nearest"
        nearest = model.field.query(*args)
    assert torch.allclose(nearest[2], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(weighted[2], torch.tensor([[0.5, 0.0, 0.5]]), atol=1e-6)
    for index in (0, 1, 3, 4):
        assert torch.equal(weighted[index], nearest[index])


def test_multiscale_depth_gradient_detects_blurred_step_and_masks_invalid():
    from backend.research.depthadapt import multiscale_gradient_loss
    import torch

    truth = torch.ones(1, 12, 16)
    truth[..., 8:] = 2
    blurred = (
        torch.linspace(1, 2, 16)[None, None].expand(1, 12, 16).clone().requires_grad_()
    )
    assert multiscale_gradient_loss(truth, truth).item() == 0
    loss = multiscale_gradient_loss(blurred, truth)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(blurred.grad).all() and blurred.grad.abs().sum() > 0
    empty = torch.zeros_like(truth)
    assert multiscale_gradient_loss(blurred, empty).item() == 0
