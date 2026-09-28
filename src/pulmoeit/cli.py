"""Command line entry point. Each sub-command is one DVC stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pulmoeit")
    parser.add_argument("--params", default="params.yaml", help="pipeline parameters (YAML)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "baseline", "train", "export"):
        sub.add_parser(name)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--enforce-gates", action="store_true", help="exit 1 if a quality gate fails")
    srv = sub.add_parser("serve")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    params = yaml.safe_load(Path(args.params).read_text())
    paths = params.get("paths", {})
    data_dir = Path(paths.get("data", "data"))
    model_dir = Path(paths.get("models", "models"))
    report_dir = Path(paths.get("reports", "reports"))

    if args.cmd == "generate":
        from .dataset import AcquisitionConfig, generate, save

        d = params["data"]
        base = d["acquisition"]
        splits = {
            "train": (d["n_train"], base),
            "val": (d["n_val"], base),
            "test": (d["n_test"], base),
            "test_shift": (d["n_test"], {**base, **d["shift"]}),
            "test_fault": (d["n_test_fault"], {**base, "fault_prob": 1.0}),
        }
        for i, (split, (n, acq)) in enumerate(splits.items()):
            data = generate(n, AcquisitionConfig.from_dict(acq), seed=params["seed"] + 1000 * i, workers=d["workers"])
            save(data_dir / f"{split}.npz", data)
            print(f"{split}: {len(data['target'])} samples")

    elif args.cmd == "baseline":
        from .dataset import load
        from .recon import fit_linear
        from .signals import features

        b = params["baseline"]
        val = load(data_dir / "val.npz")
        recon, report = fit_linear(
            features(val["v_ref"], val["v_insp"]),
            val["target"],
            [float(v) for v in b["lambdas"]],
            b["prior"],
            b["n_boundary"],
        )
        recon.save(model_dir / "baseline.npz")
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "baseline.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))

    elif args.cmd == "train":
        from .train import train

        print(json.dumps(train(params, data_dir, model_dir, report_dir), indent=2))

    elif args.cmd == "evaluate":
        from .evaluate import evaluate

        res = evaluate(params, data_dir, model_dir, report_dir)
        print((report_dir / "report.md").read_text())
        if args.enforce_gates and not res["gates"]["passed"]:
            print("quality gates failed", file=sys.stderr)
            return 1

    elif args.cmd == "export":
        from .export import export

        card = export(params, model_dir, report_dir)
        print(
            f"exported {card['architecture']} ({card['n_parameters']} params), parity err {card['onnx_parity_max_abs_error']:.1e}"
        )

    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("pulmoeit.serve.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
