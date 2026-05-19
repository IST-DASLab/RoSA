"""Shared answer extraction for math benchmarks."""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from model output."""
    matches = list(re.finditer(r"\\boxed\{([^}]*)\}", text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract numeric answer after #### marker (GSM8K format)."""
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def normalize_numeric(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip().replace(",", "").replace("$", "")
    try:
        if "." in s:
            return str(float(s))
        return str(int(float(s)))
    except ValueError:
        return s
