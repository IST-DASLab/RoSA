#!/usr/bin/env python3
"""Print install and quickstart instructions."""

print(
    """
RoSA quickstart
===============

1. Initialize submodules:
   git submodule update --init --recursive

2. Install dependencies (Python 3.11, CUDA 13.0):
   uv sync

3. Train Llama-3.2 on MetaMathQA:
   bash examples/llama3_metamath/run.sh

4. Evaluate on GSM8K:
   uv run rosa-eval --model meta-llama/Llama-3.2-3B-Instruct \\
       --adapter ./outputs/llama3_metamath --benchmark gsm8k --max-samples 100

See README.md for details.
"""
)
