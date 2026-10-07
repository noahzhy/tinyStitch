import argparse, json
from .data import generate
from .train import train
from .inference import reconstruct


def main():
    p = argparse.ArgumentParser(description="Calibrated cross-view JEPA research CLI")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--output", required=True)
    g.add_argument("--scenes", type=int, default=12)
    g.add_argument("--seed", type=int, default=730000)
    g.add_argument("--height", type=int, default=192)
    g.add_argument("--width", type=int, default=256)
    g.add_argument("--views", type=int, default=5)
    g.add_argument("--no-exposure-jitter", action="store_true")
    g.add_argument(
        "--capture-step",
        type=float,
        help="fixed spacing; shelf length grows with view count",
    )
    g.add_argument("--target-width", type=int)
    g.add_argument("--varied-layout", action="store_true")
    g.add_argument("--capture-tilt", action="store_true")
    g.add_argument("--mixed-bays", action="store_true")
    pa = sub.add_parser(
        "pose-adapt", help="scene-disjoint variable-baseline pose adaptation"
    )
    pa.add_argument("--data", required=True)
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--output", required=True)
    pa.add_argument("--steps", type=int, default=6000)
    pa.add_argument("--seed", type=int, default=20261007)
    pa.add_argument("--max-pair-gap", type=int)
    gp = sub.add_parser(
        "global-pose-adapt",
        help="global sequence feature refinement of local camera poses",
    )
    gp.add_argument("--data", required=True)
    gp.add_argument("--checkpoint", required=True)
    gp.add_argument("--output", required=True)
    gp.add_argument("--steps", type=int, default=1600)
    gp.add_argument("--seed", type=int, default=20261009)
    da = sub.add_parser("depth-adapt", help="source-only multiscene depth adaptation")
    da.add_argument("--data", required=True)
    da.add_argument("--checkpoint", required=True)
    da.add_argument("--output", required=True)
    da.add_argument("--steps", type=int, default=1200)
    da.add_argument("--seed", type=int, default=20261007)
    da.add_argument("--consistency-weight", type=float, default=0)
    da.add_argument("--edge-weight", type=float, default=0)
    da.add_argument("--multiscale-weight", type=float, default=0)
    da.add_argument("--skip-test", action="store_true")
    ra = sub.add_parser(
        "render-adapt", help="frozen-geometry multiscene orthographic field training"
    )
    ra.add_argument("--data", required=True)
    ra.add_argument("--checkpoint", required=True)
    ra.add_argument("--output", required=True)
    ra.add_argument("--steps", type=int, default=1800)
    ra.add_argument("--seed", type=int, default=20261008)
    ra.add_argument("--rays", type=int, default=2048)
    ra.add_argument("--input-counts", nargs="+", type=int, default=[60])
    ra.add_argument("--feature-center-mapping", action="store_true")
    ra.add_argument("--local-sources", type=int)
    ra.add_argument(
        "--source-geometry", choices=["predicted", "oracle"], default="predicted"
    )
    ra.add_argument("--warm-field", action="store_true")
    ma = sub.add_parser(
        "matching-adapt", help="train cross-view matching on varied shelf scenes"
    )
    ma.add_argument("--data", required=True)
    ma.add_argument("--output", required=True)
    ma.add_argument("--steps", type=int, default=1200)
    ma.add_argument("--batch", type=int, default=4)
    ma.add_argument("--device", default="mps")
    ae = sub.add_parser(
        "alignment-export",
        help="export source-only alignment for approximate camera lifting",
    )
    ae.add_argument("--scene", required=True)
    ae.add_argument("--report", required=True)
    ae.add_argument("--output", required=True)
    hg = sub.add_parser(
        "homography-global", help="source-only global shelf-line homography alignment"
    )
    hg.add_argument("--scene", required=True)
    hg.add_argument("--output", required=True)
    hg.add_argument("--method", choices=["sift", "jepa"], default="sift")
    hg.add_argument("--weight", type=float, default=1.0)
    hg.add_argument(
        "--model",
        choices=[
            "projective",
            "translation",
            "orientation",
            "orientation-consensus",
            "orientation-local",
            "orientation-local-level",
        ],
        default="translation",
    )
    hg.add_argument("--checkpoint")
    hg.add_argument("--no-gt-eval", dest="evaluate_gt", action="store_false")
    hb = sub.add_parser(
        "homography-benchmark",
        help="existing estimated homographies on the same NVS scoring canvas",
    )
    hb.add_argument("--scene", required=True)
    hb.add_argument("--reference", required=True)
    hb.add_argument("--output", required=True)
    hb.add_argument(
        "--methods", nargs="+", choices=["sift", "jepa"], default=["sift", "jepa"]
    )
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--device", default="cpu")
    t.add_argument("--dim", type=int, default=192)
    t.add_argument("--patch", type=int, default=8)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--resume")
    t.add_argument("--same-image", action="store_true")
    t.add_argument("--no-vae", action="store_true")
    t.add_argument("--single-scene", action="store_true")
    r = sub.add_parser("reconstruct")
    r.add_argument("--input", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--optimize", type=int, default=20)
    r.add_argument("--width", type=int, default=1024)
    r.add_argument(
        "--projection", choices=["perspective", "orthographic"], default="orthographic"
    )
    r.add_argument("--geometry-width", type=int)
    r.add_argument("--samples", type=int, default=32)
    r.add_argument("--no-global-opt", action="store_true")
    r.add_argument("--no-vae", action="store_true")
    b = sub.add_parser("evaluate")
    b.add_argument("--data", required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--device", default="cpu")
    b.add_argument("--optimize", type=int, default=5)
    b.add_argument("--samples", type=int, default=24)
    b.add_argument(
        "--modes",
        nargs="+",
        choices=["feedforward", "optimized", "no_vae", "no_global", "oracle_cameras"],
    )
    native = sub.add_parser(
        "generate-native",
        help="rerender calibrated synthetic captures at native resolution",
    )
    native.add_argument("--scene", required=True)
    native.add_argument("--output", required=True)
    native.add_argument("--width", type=int, default=1280)
    native.add_argument("--height", type=int, default=960)
    native.add_argument("--target-width", type=int, default=256)
    m = sub.add_parser(
        "multiview", help="frozen-checkpoint 10/30/60 view scaling evaluation"
    )
    m.add_argument("--scene", required=True)
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--device", default="cpu")
    m.add_argument("--width", type=int, default=128)
    m.add_argument("--samples", type=int, default=32)
    m.add_argument("--geometry-width", type=int)
    m.add_argument("--color-residual-limit", type=float)
    m.add_argument("--source-limit", type=int)
    m.add_argument("--local-sources", type=int)
    m.add_argument("--color-fusion", choices=["weighted", "consensus", "nearest"])
    m.add_argument("--alignment")
    m.add_argument("--oracle-control", action="store_true")
    m.add_argument("--counts", type=int, nargs="+", default=[10, 30, 60])
    o = sub.add_parser(
        "orthofit", help="explicit synthetic orthographic RGB/depth supervision"
    )
    o.add_argument(
        "--source-geometry", choices=["predicted", "oracle"], default="predicted"
    )
    o.add_argument("--deepen", action="store_true")
    o.add_argument("--local-sources", type=int)
    o.add_argument("--freeze-density", action="store_true")
    o.add_argument("--edge-fraction", type=float, default=0.0)
    o.add_argument("--surface-weight", type=float, default=0.0)
    o.add_argument("--occlusion-aware", action="store_true")
    o.add_argument("--adapt-steps", type=int, default=0)
    o.add_argument("--scene", required=True)
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--output", required=True)
    o.add_argument("--steps", type=int, default=2000)
    o.add_argument("--width", type=int, default=256)
    o.add_argument("--device", default="cpu")
    f = sub.add_parser(
        "overfit", help="single-scene overfitting with isolated held-out evaluation"
    )
    f.add_argument("--scene", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--device", default="cpu")
    f.add_argument("--encoder-steps", type=int, default=400)
    f.add_argument("--geometry-steps", type=int, default=1200)
    f.add_argument("--vae-steps", type=int, default=1000)
    f.add_argument("--render-steps", type=int, default=1500)
    f.add_argument("--samples", type=int, default=64)
    f.add_argument("--warm-start")
    f.add_argument("--sampling", choices=["uniform", "depth_guided"], default="uniform")
    f.add_argument("--skip-legacy", action="store_true")
    a = vars(p.parse_args())
    command = a.pop("command")
    if command == "generate":
        a["exposure_jitter"] = not a.pop("no_exposure_jitter")
        a["root"] = a.pop("output")
        a["h"] = a.pop("height")
        a["w"] = a.pop("width")
        generate(**a)
    elif command == "generate-native":
        from .native_data import generate_native

        print(json.dumps(generate_native(**a)))
    elif command == "multiview":
        from .multiview import run

        run(**a)
    elif command == "orthofit":
        from .orthofit import run

        run(**a)
    elif command == "render-adapt":
        from .renderadapt import run

        run(**a)
    elif command == "depth-adapt":
        from .depthadapt import run

        run(**a)
    elif command == "global-pose-adapt":
        from .globalpose import run

        run(**a)
    elif command == "pose-adapt":
        from .poseadapt import run

        run(**a)
    elif command == "matching-adapt":
        from .matchingadapt import run

        run(**a)
    elif command == "alignment-export":
        from .alignment_pose import export

        export(**a)
    elif command == "homography-global":
        from .homography_global import run

        run(**a)
    elif command == "homography-benchmark":
        from .homography_benchmark import run

        run(**a)
    elif command == "train":
        print(train(**a))
    elif command == "overfit":
        from .overfit import overfit

        overfit(**a)
    elif command == "evaluate":
        from .evaluate import evaluate

        evaluate(**a)
    else:
        a["path"] = a.pop("input")
        a["global_opt"] = not a.pop("no_global_opt")
        a["use_vae"] = not a.pop("no_vae")
        print(json.dumps(reconstruct(**a), indent=2))


if __name__ == "__main__":
    main()
