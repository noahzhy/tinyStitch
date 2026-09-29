import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from backend.jepa import DenseJEPA, JEPADescriptor, DEFAULT_CHECKPOINT, get_checkpoint
from backend.stitch import stitch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_jepa import pair


def test_target_stop_gradient_mask_and_ema():
    torch.set_num_threads(2)
    rng = np.random.default_rng(18)
    samples = [
        pair(rng.integers(0, 255, (240, 240, 3), dtype=np.uint8), rng) for _ in range(2)
    ]
    tensors = [torch.tensor(np.stack([s[k] for s in samples])) for k in range(5)]
    model = DenseJEPA()
    before = next(model.target.parameters()).detach().clone()
    loss, metrics = model.loss(*tensors)
    loss.backward()
    assert all(
        p.grad is None and not p.requires_grad for p in model.target.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert torch.isfinite(loss) and metrics["feature_std"] > 0
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer.step()
    model.update_target(0.9)
    assert not torch.equal(before, next(model.target.parameters()))
    assert tensors[4].sum() > 0 and (tensors[4] * tensors[3]).sum() > 0


def test_missing_model_never_falls_back(tmp_path):
    with pytest.raises(RuntimeError, match="缺少已训练"):
        get_checkpoint(tmp_path / "missing.pt")


def test_actual_trained_descriptor_and_rgb_only_stitch(tmp_path):
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("尚未训练；最终验收必须提供真实权重")
    extractor = JEPADescriptor(device="cpu")
    root = Path(__file__).resolve().parents[1] / "examples/simulated/rgb"
    paths = sorted(root.glob("*.jpg"))
    assert len(paths) == 9

    # SIFT descriptor computation is forbidden: detector supplies locations only.
    class DetectOnly:
        def detect(self, *args):
            return cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.025).detect(
                *args
            )

        def detectAndCompute(self, *args):
            raise AssertionError("JEPA flow must not use SIFT descriptors")

    extractor.detector = DetectOnly()
    keys, desc = extractor.extract(cv2.imread(str(paths[0])))
    assert desc.shape == (len(keys), 64) and np.isfinite(desc).all()
    assert desc.std(0).mean() > 0.005
    report = stitch(paths, tmp_path / "learned", method="jepa", device="cpu")
    assert report["method"] == "jepa" and report["model"]["trained_steps"] > 0
    assert report["model"]["checkpoint_sha256"] == extractor.info["checkpoint_sha256"]
    assert (tmp_path / "learned/panorama.png").exists()
    assert report["input_count"] == 9
