"""MATH benchmark evaluation (boxed answer extraction)."""

from __future__ import annotations

import logging
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from rosa.eval.utils import extract_boxed_answer, normalize_numeric

logger = logging.getLogger(__name__)

MATH_PROMPT = "Solve the following problem. Put your final answer in \\boxed{{}}.\n\n{problem}"


def evaluate_math(
    model_name_or_path: str,
    adapter_path: str | None = None,
    *,
    dataset_name: str = "lighteval/MATH",
    split: str = "test",
    max_samples: int | None = None,
    max_new_tokens: int = 1024,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Run MATH exact-match on \\boxed{} answers."""
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

    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception:
        ds = load_dataset("hendrycks/competition_math", split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    for row in tqdm(ds, desc="MATH"):
        problem = row.get("problem") or row.get("question", "")
        solution = row.get("solution") or row.get("answer", "")
        gold = normalize_numeric(extract_boxed_answer(solution) or solution)

        prompt = MATH_PROMPT.format(problem=problem)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred_text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred = normalize_numeric(extract_boxed_answer(pred_text))
        if pred == gold:
            correct += 1
        total += 1

    accuracy = correct / total if total else 0.0
    logger.info("MATH accuracy: %.2f%% (%d/%d)", 100 * accuracy, correct, total)
    return {"accuracy": accuracy, "correct": correct, "total": total}
