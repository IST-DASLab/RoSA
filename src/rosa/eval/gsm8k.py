"""GSM8K evaluation (exact match on final numeric answer)."""

from __future__ import annotations

import logging
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from rosa.eval.utils import extract_gsm8k_answer, normalize_numeric

logger = logging.getLogger(__name__)

GSM8K_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final numeric answer after ####.\n\n{question}"
)


def evaluate_gsm8k(
    model_name_or_path: str,
    adapter_path: str | None = None,
    *,
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Run GSM8K exact-match evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    for row in tqdm(ds, desc="GSM8K"):
        question = row["question"]
        gold = normalize_numeric(extract_gsm8k_answer(row["answer"]))
        prompt = GSM8K_PROMPT.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = normalize_numeric(extract_gsm8k_answer(pred_text))
        if pred == gold:
            correct += 1
        total += 1

    accuracy = correct / total if total else 0.0
    logger.info("GSM8K accuracy: %.2f%% (%d/%d)", 100 * accuracy, correct, total)
    return {"accuracy": accuracy, "correct": correct, "total": total}
