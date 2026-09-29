"""Deployment proof: learned descriptors, RGB-only input, model digest and independent data split."""

import json
import shutil
import sys
import tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.jepa import get_checkpoint
from backend.stitch import stitch

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "artifacts/models/training-data.json").read_text())
train = set(manifest["train_layouts"])
val = set(manifest["val_layouts"])
test = {p.name for p in (ROOT / "data/jepa-test").iterdir() if p.is_dir()}
assert not (train & val or train & test or val & test)
model, info = get_checkpoint()
torch.set_num_threads(2)
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    rgb = base / "rgb"
    rgb.mkdir()
    for file in (ROOT / "examples/simulated/rgb").glob("*.jpg"):
        shutil.copyfile(file, rgb / file.name)
    # No generation/scene/pose/truth directory exists in this deployment copy.
    first = stitch(sorted(rgb.glob("*.jpg")), base / "first", device="cpu")
    import cv2

    path = rgb / "0000.jpg"
    image = cv2.imread(str(path))
    cv2.rectangle(image, (200, 100), (300, 200), (0, 0, 0), -1)
    cv2.imwrite(str(path), image)
    second = stitch(sorted(rgb.glob("*.jpg")), base / "second", device="cpu")
    assert first["input_digest"] != second["input_digest"]
    assert first["model"]["checkpoint_sha256"] == info["checkpoint_sha256"]
    assert first["method"] == "jepa" and first["model"]["trained_steps"] > 0
    result = {
        "rgb_only_without_generation_labels": True,
        "changed_input_recomputed": True,
        "train_val_test_layout_disjoint": True,
        "train_layouts": len(train),
        "validation_layouts": len(val),
        "test_layouts": len(test),
        "model": info,
        "cpu_sample_seconds": first["seconds"],
        "sample_size": first["size"],
    }
(ROOT / "artifacts/deployment-validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
