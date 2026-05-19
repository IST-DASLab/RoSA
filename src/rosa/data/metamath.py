"""MetaMathQA dataset loading and chat formatting."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


def format_metamath_example(
    example: dict[str, Any],
    tokenizer,
) -> dict[str, str]:
    """Format a MetaMathQA row into a single text field for SFT."""
    query = example.get("query") or example.get("question") or example.get("input", "")
    response = example.get("response") or example.get("answer") or example.get("output", "")

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = f"### Question:\n{query}\n\n### Answer:\n{response}"

    return {"text": text}


def load_metamath_dataset(
    dataset_name: str = "meta-math/MetaMathQA",
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds
