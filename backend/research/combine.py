"""Combine independently validation-selected field and depth components."""

import argparse, copy, hashlib, json
from pathlib import Path
import torch


def run(renderer, depth, output):
    r = torch.load(renderer, map_location="cpu", weights_only=False)
    d = torch.load(depth, map_location="cpu", weights_only=False)
    selected = lambda k: k.startswith(("pose.depth.", "pose.depth_refiner."))
    for k, value in r["model"].items():
        if not k.startswith("field.") and not selected(k):
            if k not in d["model"] or not torch.equal(value, d["model"][k]):
                raise ValueError(
                    "Frozen encoders/camera/appearance differ; unsafe merge"
                )
    state = copy.deepcopy(r)
    state["reference_depth_calibration"] = dict(
        weights={
            k.removeprefix("pose."): v.clone()
            for k, v in r["model"].items()
            if selected(k)
        },
        checkpoint_sha256=hashlib.sha256(Path(renderer).read_bytes()).hexdigest(),
        protocol="frozen pre-adaptation depth head for source camera gauge",
    )
    keys = [k for k in d["model"] if selected(k)]
    if not keys:
        raise ValueError("Missing depth component")
    for k in keys:
        if k not in r["model"] or r["model"][k].shape != d["model"][k].shape:
            raise ValueError("Incompatible depth component")
        state["model"][k] = d["model"][k].clone()
    state["depth_adaptation"] = d["depth_adaptation"]
    protocol = dict(
        renderer_sha256=hashlib.sha256(Path(renderer).read_bytes()).hexdigest(),
        depth_sha256=hashlib.sha256(Path(depth).read_bytes()).hexdigest(),
        selection="independent training/validation selection; no additional fit",
        copied_keys=keys,
    )
    state["component_combination"] = protocol
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        raise ValueError("Use fresh output")
    torch.save(state, out / "model.pt")
    (out / "report.json").write_text(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--renderer", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))
