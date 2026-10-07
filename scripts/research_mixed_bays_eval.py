"""Matched mixed-bay first-stage comparison; inference never reads bay metadata."""

from pathlib import Path
import subprocess, os
from concurrent.futures import ThreadPoolExecutor

root = Path("artifacts/research/mixed-bays-stage1-v1")
root.mkdir(parents=True, exist_ok=True)


def run(task):
    seed, model = task
    name = f'{"local" if model=="orientation-local" else "global"}-{seed}'
    output = root / name
    if (output / "report.json").exists():
        return
    cmd = [
        ".venv/bin/python",
        "-m",
        "backend.research.cli",
        "homography-global",
        "--scene",
        f"data/research/mixed-bays-v1/{seed}",
        "--output",
        str(output),
        "--method",
        "jepa",
        "--model",
        model,
        "--checkpoint",
        "artifacts/research/mixed-bays-matching-v1/shelf-jepa.pt",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
    )
    (root / f"{name}.log").write_text(result.stdout + result.stderr)
    print(name, result.returncode, flush=True)


with ThreadPoolExecutor(max_workers=2) as pool:
    list(
        pool.map(
            run,
            [
                (seed, model)
                for seed in range(738018, 738024)
                for model in ("orientation-local", "orientation-consensus")
            ],
        )
    )

for seed in range(738018, 738024):
    report = root / f"local-{seed}/report.json"
    if report.exists():
        subprocess.run(
            [
                ".venv/bin/python",
                "-m",
                "backend.research.cli",
                "alignment-export",
                "--scene",
                f"data/research/mixed-bays-v1/{seed}",
                "--report",
                str(report),
                "--output",
                str(root / f"alignment-{seed}.json"),
            ],
            check=True,
        )
