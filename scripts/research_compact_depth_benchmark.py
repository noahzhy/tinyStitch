"""Matched cached-token depth latency, excluding encoder and reference gauge."""

import argparse, json, time, statistics
from pathlib import Path
import torch
from backend.research.inference import load_model


def run(compact, output):
    torch.set_num_threads(4)
    rows = []
    for checkpoint in [
        "artifacts/research/mixed-bays-geometry-depth-v2/model.pt",
        compact,
    ]:
        model, _ = load_model(checkpoint, "mps")
        tokens = torch.randn(4, 48 * 64, model.config["dim"], device="mps")
        with torch.no_grad():
            for _ in range(10):
                model.pose.depths(tokens, 48, 64, (192, 256))
            torch.mps.synchronize()
            times = []
            for _ in range(30):
                start = time.perf_counter()
                model.pose.depths(tokens, 48, 64, (192, 256))
                torch.mps.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        rows.append(
            dict(
                checkpoint=checkpoint,
                refiner_parameters=sum(
                    p.numel() for p in model.pose.depth_refiner.parameters()
                ),
                depth_parameters=sum(
                    p.numel()
                    for n, p in model.pose.named_parameters()
                    if n.startswith(("depth.", "depth_refiner."))
                ),
                median_batch4_ms=statistics.median(times),
                p95_batch4_ms=sorted(times)[28],
            )
        )
    Path(output).write_text(
        json.dumps(
            dict(
                device="mps",
                cached_tokens=True,
                batch=4,
                resolution=[256, 192],
                encoder_and_calibration_excluded=True,
                rows=rows,
            ),
            indent=2,
        )
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compact", required=True)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))
