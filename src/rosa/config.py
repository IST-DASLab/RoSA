"""Configuration dataclasses for RoSA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RosaAdapterConfig:
    """RoSA / LoRA adapter hyperparameters (maps to peft RosaConfig)."""

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    spa_d: float = 0.01
    spa_num_grads: int = 1
    grad_acc_mode: str = "mean_squared"
    target_modules: str | list[str] = "all-linear"
    schedule: str = "wl64"
    impl: str = "auto"
    lora_lr: float | None = None
    mask_load_path: str | None = None
    mask_save_path: str | None = None


@dataclass
class TrainingConfig:
    """TRL SFT training settings."""

    output_dir: str = "./outputs"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str | list[str] = "none"


@dataclass
class QuantizationConfig:
    """Optional 4-bit QRoSA loading."""

    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration loaded from YAML."""

    model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    dataset_name: str = "meta-math/MetaMathQA"
    dataset_split: str = "train"
    rosa: RosaAdapterConfig = field(default_factory=RosaAdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    trust_remote_code: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        rosa_raw = dict(raw.get("rosa") or {})
        # Backward compatibility: lift mask_phase fields into rosa
        mask_phase = raw.get("mask_phase") or {}
        if mask_phase.get("mask_load_path") and not rosa_raw.get("mask_load_path"):
            rosa_raw["mask_load_path"] = mask_phase["mask_load_path"]
        if mask_phase.get("mask_save_path") and not rosa_raw.get("mask_save_path"):
            rosa_raw["mask_save_path"] = mask_phase["mask_save_path"]
        if mask_phase.get("warmup_schedule") and rosa_raw.get("schedule") in (None, "default"):
            rosa_raw["schedule"] = mask_phase["warmup_schedule"]

        rosa = RosaAdapterConfig(**rosa_raw)
        training = TrainingConfig(**(raw.get("training") or {}))
        quantization = QuantizationConfig(**(raw.get("quantization") or {}))
        known = {"rosa", "training", "quantization", "mask_phase"}
        top = {k: v for k, v in raw.items() if k not in known}
        return cls(
            rosa=rosa,
            training=training,
            quantization=quantization,
            **top,
        )
