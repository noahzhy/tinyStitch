import argparse, json, subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="artifacts/research/grouped-layout-eval-v1")
parser.add_argument(
    "--checkpoint", default="artifacts/research/grouped-matching-v1/shelf-jepa.pt"
)
options = parser.parse_args()
root = Path(options.output)
root.mkdir(parents=True, exist_ok=True)
tasks = []
for tilted, seed in [(False, 736018), (True, 737018)]:
    for n in range(6):
        for method, model in [("sift", "translation"), ("jepa", "translation")] + (
            [("sift", "orientation"), ("jepa", "orientation")] if tilted else []
        ):
            tasks.append((tilted, seed + n, method, model))


def work(t):
    tilted, seed, method, model = t
    name = f'{"tilt" if tilted else "front"}-{method}-{model}-{seed}'
    out = root / name
    if (out / "report.json").exists():
        return name
    cmd = [
        ".venv/bin/python",
        "-m",
        "backend.research.homography_global",
        "--scene",
        f'data/research/grouped-varied{"-tilt" if tilted else ""}-v1/{seed}',
        "--output",
        str(out),
        "--method",
        method,
        "--model",
        model,
    ]
    if method == "jepa":
        cmd += ["--checkpoint", options.checkpoint]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (root / (name + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        out.mkdir(exist_ok=True)
        (out / "failure.json").write_text(
            json.dumps(dict(command=cmd, error=result.stderr))
        )
    print(name, result.returncode, flush=True)
    return name


with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(work, tasks))
