from __future__ import annotations

import argparse

from .chronos_runner import run_chronos
from .compare import compare_runs
from .config import load_config
from .train import train_model
from .utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NSW load forecasting research framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--model", required=True, choices=["baseline", "patchtst", "itransformer", "diffusion"])
    p_train.add_argument("--task", required=True, choices=["direct", "residual"])

    p_chronos = sub.add_parser("chronos")
    p_chronos.add_argument("--config", required=True)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--config", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if args.command == "train":
        out = train_model(cfg, model_name=args.model, task=args.task)
        print(out)
    elif args.command == "chronos":
        out = run_chronos(cfg)
        print(out)
    elif args.command == "compare":
        out = compare_runs(cfg)
        print(out)


if __name__ == "__main__":
    main()
