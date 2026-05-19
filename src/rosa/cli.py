"""CLI entry points for training and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rosa.config import ExperimentConfig
from rosa.eval.gsm8k import evaluate_gsm8k
from rosa.eval.math_eval import evaluate_math
from rosa.training.sft import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main_train(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RoSA SFT training")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.from_yaml(args.config)
    train(cfg)


def main_eval(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RoSA evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--benchmark", choices=["gsm8k", "math"], required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    if args.benchmark == "gsm8k":
        result = evaluate_gsm8k(
            args.model,
            args.adapter,
            max_samples=args.max_samples,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        result = evaluate_math(
            args.model,
            args.adapter,
            max_samples=args.max_samples,
            trust_remote_code=args.trust_remote_code,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        sys.argv.pop(1)
        main_eval()
    else:
        main_train()
