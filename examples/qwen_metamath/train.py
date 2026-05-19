#!/usr/bin/env python3
"""Train Qwen3.5-9B with RoSA on MetaMathQA."""

from pathlib import Path

from rosa.config import ExperimentConfig
from rosa.training.sft import train

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    train(ExperimentConfig.from_yaml(config_path))
