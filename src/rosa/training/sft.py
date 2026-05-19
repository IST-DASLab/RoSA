"""TRL SFTTrainer integration with RoSA (peft-rosa)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import get_peft_model
from peft.tuners.rosa import RosaConfig, RosaScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from rosa.config import ExperimentConfig
from rosa.data.metamath import format_metamath_example, load_metamath_dataset

logger = logging.getLogger(__name__)


def _build_rosa_config(cfg: ExperimentConfig) -> RosaConfig:
    r = cfg.rosa
    mask_save = r.mask_save_path
    if mask_save is None and r.spa_d > 0:
        mask_save = str(Path(cfg.training.output_dir) / "masks")

    return RosaConfig(
        r=r.lora_r,
        d=r.spa_d,
        lora_alpha=r.lora_alpha,
        target_modules=r.target_modules,
        lora_dropout=r.lora_dropout,
        impl=r.impl,
        spa_num_grads=r.spa_num_grads,
        grad_acc_mode=r.grad_acc_mode,
        mask_load_path=r.mask_load_path,
        mask_save_path=mask_save,
        schedule=r.schedule,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(cfg: ExperimentConfig):
    quant = cfg.quantization
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": torch.bfloat16 if cfg.training.bf16 else torch.float32,
    }
    if quant.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs.pop("torch_dtype", None)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
    return model, tokenizer


def prepare_dataset(cfg: ExperimentConfig, tokenizer) -> Dataset:
    ds = load_metamath_dataset(cfg.dataset_name, cfg.dataset_split)
    return ds.map(
        lambda ex: format_metamath_example(ex, tokenizer),
        remove_columns=ds.column_names,
        desc="Formatting dataset",
    )


def train(cfg: ExperimentConfig) -> None:
    """RoSA SFT with a single trainer run and RosaScheduler."""
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    logger.info("RoSA training (schedule=%s)", cfg.rosa.schedule)
    model, tokenizer = load_model_and_tokenizer(cfg)
    train_dataset = prepare_dataset(cfg, tokenizer)

    peft_config = _build_rosa_config(cfg)
    model = get_peft_model(model, peft_config)

    t = cfg.training
    if t.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    sft_args = SFTConfig(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        warmup_steps=t.warmup_steps,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        seed=t.seed,
        bf16=t.bf16,
        report_to=t.report_to,
        max_length=t.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[RosaScheduler(model)],
    )
    trainer.train()
    trainer.save_model(t.output_dir)
    tokenizer.save_pretrained(t.output_dir)
