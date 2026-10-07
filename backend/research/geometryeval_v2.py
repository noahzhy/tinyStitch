import json, torch
from pathlib import Path
from backend.research.data import load_supervised
from backend.research.inference import load_model
from backend.research.depthadapt import consistency_loss
from backend.research.geometry import correspondence, backproject, project, sample


def main():
    root = Path("artifacts/research/staged-optimization-v2")
    root.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    rgb, K, T, d = load_supervised("data/research/multiview-geometry256-v1/731005")
    valid = d > 0
    edge = torch.zeros_like(valid)
    for axis in (-1, -2):
        diff = d.clamp_min(0.001).log().diff(dim=axis).abs() > 0.015
        if axis == -1:
            edge[..., 1:] |= diff
            edge[..., :-1] |= diff
        else:
            edge[..., 1:, :] |= diff
            edge[..., :-1, :] |= diff
    edge = (
        torch.nn.functional.max_pool2d(edge.float()[:, None], 3, 1, 1)[:, 0].bool()
        & valid
    )
    rows = []
    for name in ["depth-multiscene-v1", "depth-edge-control-v2", "depth-consistent-v2"]:
        model, _ = load_model(Path("artifacts/research") / name / "model.pt")
        with torch.no_grad():
            tokens = torch.cat([model.jepa.encoder(image[None])[0] for image in rgb])
            pred = model.pose.depths(tokens, 48, 64, (192, 256))
            residual = (pred - d).abs() / d.clamp_min(0.001)
            terms = []
            errors = []
            for i in range(0, 56, 4):
                ids = [i, i + 4]
                terms.append(
                    float(consistency_loss(pred[ids], d[ids], K[ids], T[ids], stride=8))
                )
                xy, mask = correspondence(
                    d[i], d[i + 4], K[i], K[i + 4], T[i], T[i + 4]
                )
                q, z = project(
                    backproject(pred[i], K[i], T[i])[::8, ::8], K[i + 4], T[i + 4]
                )
                mask = mask[::8, ::8]
                errors.append((q - xy[::8, ::8])[mask].norm(dim=-1))
            errors = torch.cat(errors)
            row = dict(
                checkpoint=name,
                abs_rel=float(residual[valid].mean()),
                edge_abs_rel=float(residual[edge].mean()),
                interior_abs_rel=float(residual[valid & ~edge].mean()),
                source_consistency_loss=sum(terms) / len(terms),
                depth_only_reprojection_median_px=float(errors.median()),
                depth_only_reprojection_p95_px=float(torch.quantile(errors, 0.95)),
                protocol="GT source poses isolate depth effects; no target-view labels",
            )
            rows.append(row)
            print(row, flush=True)
    (root / "stage1-depth-test.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
