"""Matched first-stage test, frozen observed-motion consensus parameters."""

import subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

root = Path("artifacts/research/sequence-consensus-v1")
root.mkdir(parents=True, exist_ok=True)


def run(item):
    method, seed = item
    output = root / f"test-{method}-{seed}"
    cmd = [
        ".venv/bin/python",
        "-m",
        "backend.research.cli",
        "homography-global",
        "--scene",
        f"data/research/grouped-varied-tilt-v1/{seed}",
        "--output",
        str(output),
        "--method",
        method,
        "--model",
        "orientation-consensus",
    ]
    if method == "jepa":
        cmd += ["--checkpoint", "artifacts/research/grouped-matching-v1/shelf-jepa.pt"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
    )
    (root / f"test-{method}-{seed}.log").write_text(result.stdout + result.stderr)
    print(method, seed, result.returncode, flush=True)


with ThreadPoolExecutor(max_workers=2) as pool:
    list(
        pool.map(
            run,
            [
                (method, seed)
                for seed in range(737018, 737024)
                for method in ("sift", "jepa")
            ],
        )
    )
