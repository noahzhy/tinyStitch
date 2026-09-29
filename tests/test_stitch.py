import os
import time
import cv2
import numpy as np
import pytest
from PIL import Image
from backend.stitch import StitchError, load_image, points, stitch


def poster(directory):
    random = np.random.default_rng(77)
    canvas = np.full((480, 1500, 3), 235, np.uint8)
    for i in range(450):
        x, y = random.integers([10, 10], [1490, 470])
        color = tuple(int(c) for c in random.integers(0, 200, 3))
        cv2.circle(canvas, (int(x), int(y)), int(random.integers(3, 15)), color, -1)
    for i in range(14):
        cv2.putText(
            canvas,
            f"PRODUCT {i} / {i * i}",
            (i * 105, 100 + i % 3 * 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
        )
    paths = []
    for i, offset in enumerate([0, 300, 600, 800]):
        path = directory / f"{i:04d}.png"
        cv2.imwrite(str(path), canvas[:, offset : offset + 700])
        paths.append(path)
    return canvas, paths


def test_real_alignment_and_pixels(tmp_path):
    canvas, paths = poster(tmp_path)
    report = stitch(paths, tmp_path / "result", method="sift")
    assert report["input_count"] == 4
    assert report["alignment"]["after_median_px"] < 1
    assert 1480 < report["size"][0] < 1520
    hs = [np.array(h) for h in report["transforms_work_to_panorama"]]
    projected = [
        points(h, [[1000 - offset, 240]])[0] for h, offset in zip(hs[2:], [600, 800])
    ]
    assert np.linalg.norm(projected[0] - projected[1]) < 1.5
    rgba = cv2.imread(str(tmp_path / "result/panorama.png"), cv2.IMREAD_UNCHANGED)
    assert rgba.shape[2] == 4 and rgba[:, :, 3].mean() > 240
    center = points(hs[2], [[200, 240]])[0].round().astype(int)
    assert (
        np.linalg.norm(rgba[center[1], center[0], :3].astype(float) - canvas[240, 800])
        < 8
    )
    changed = cv2.imread(str(paths[0]))
    cv2.circle(changed, (100, 150), 15, (0, 0, 0), -1)
    cv2.imwrite(str(paths[0]), changed)
    second = stitch(paths, tmp_path / "second", method="sift")
    assert second["input_digest"] != report["input_digest"]


def test_reject_untextured_and_disconnected(tmp_path):
    _, paths = poster(tmp_path)
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((480, 700, 3), 255, np.uint8))
    with pytest.raises(StitchError, match="断开"):
        stitch([paths[0], blank, paths[1]], tmp_path / "fail", method="sift")
    assert (tmp_path / "fail/diagnostics.json").exists()
    with pytest.raises(StitchError, match="2–48"):
        stitch([paths[0]], tmp_path / "single")


def test_exif_and_resize(tmp_path):
    img = Image.new("RGB", (800, 400), "red")
    exif = img.getexif()
    exif[274] = 6
    path = tmp_path / "phone.jpg"
    img.save(path, exif=exif)
    loaded, original = load_image(path, 300)
    assert original == (400, 800) and loaded.shape[:2] == (300, 150)


def test_api_rgb_only_actual_job(tmp_path):
    os.environ["tinyStitch_DATA"] = str(tmp_path / "service")
    from backend.api import app
    from fastapi.testclient import TestClient

    _, paths = poster(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files=[("files", (p.name, p.read_bytes(), "image/png")) for p in paths],
        )
        assert response.status_code == 200
        rgb = response.json()
        assert (
            client.post(
                "/api/stitch", json={"rgb_id": rgb["id"], "pose": [1, 2]}
            ).status_code
            == 422
        )
        response = client.post(
            "/api/stitch", json={"rgb_id": rgb["id"], "method": "sift"}
        )
        assert response.status_code == 200
        job = response.json()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert job["status"] == "completed", job
        assert client.get(job["result"]["png"]).content[:8] == b"\x89PNG\r\n\x1a\n"
        assert client.get(f"/api/inputs/{rgb['id']}/download").content[:2] == b"PK"
        assert client.get("/api/inputs/../../etc/passwd").status_code != 200
        assert job["result"]["report"]["input_count"] == len(paths)
