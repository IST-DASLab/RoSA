#!/usr/bin/env python3
"""Train Llama-3.2-3B-Instruct with RoSA on MetaMathQA."""

from pathlib import Path

from rosa.config import ExperimentConfig
from rosa.training.sft import train

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    train(ExperimentConfig.from_yaml(config_path))
