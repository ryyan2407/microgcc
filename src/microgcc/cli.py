from __future__ import annotations

import argparse

from .training import load_artifacts, train_all
from .visualize import generate_figures


def main() -> None:
    parser = argparse.ArgumentParser(prog="microgcc")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data", default="data.csv")
    train.add_argument("--out", default="artifacts")
    train.add_argument("--fast", action="store_true", help="Run only seasonal-naive baseline for smoke tests.")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--artifacts", default="artifacts")

    visualize = sub.add_parser("visualize")
    visualize.add_argument("--data", default="data.csv")
    visualize.add_argument("--artifacts", default="artifacts")
    visualize.add_argument("--out", default="reports/figures")

    serve = sub.add_parser("serve")
    serve.add_argument("--artifacts", default="artifacts")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "train":
        registry = train_all(args.data, args.out, fast=args.fast)
        print(f"trained artifacts in {args.out}; forecast window {registry['forecast_start']} to {registry['forecast_end']}")
    elif args.command == "evaluate":
        _, _, metrics = load_artifacts(args.artifacts)
        print(metrics.groupby("model")["smape"].mean().sort_values().to_string())
    elif args.command == "visualize":
        paths = generate_figures(args.data, args.artifacts, args.out)
        for path in paths:
            print(path)
    elif args.command == "serve":
        import uvicorn
        from .api import create_app

        uvicorn.run(create_app(args.artifacts), host=args.host, port=args.port)

