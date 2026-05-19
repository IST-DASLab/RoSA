"""CPU smoke tests (no GPU / large downloads required for config path)."""

from pathlib import Path

import pytest

from rosa.config import ExperimentConfig
from rosa.eval.utils import extract_boxed_answer, extract_gsm8k_answer, normalize_numeric


def test_config_from_yaml():
    cfg_path = Path(__file__).parent.parent / "examples" / "llama3_metamath" / "config.yaml"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert "Llama" in cfg.model_name_or_path
    assert cfg.rosa.lora_r == 16
    assert cfg.training.bf16 is True


def test_qwen_config():
    cfg_path = Path(__file__).parent.parent / "examples" / "qwen_metamath" / "config.yaml"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert cfg.model_name_or_path == "Qwen/Qwen3.5-9B"


def test_answer_extraction():
    assert extract_gsm8k_answer("Reasoning\n#### 42") == "42"
    assert extract_boxed_answer(r"Thus \boxed{17}") == "17"
    assert normalize_numeric("1,234") == "1234"


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("third_party/peft-rosa/pyproject.toml").exists()
    and not Path(__file__).parent.parent.joinpath("third_party/peft-rosa/setup.py").exists(),
    reason="peft-rosa submodule not initialized",
)
def test_rosa_config_import():
    from peft.tuners.rosa import RosaConfig

    cfg = RosaConfig(r=8, d=0.0, schedule="lora_only", task_type="CAUSAL_LM")
    assert cfg.r == 8
    assert cfg.d == 0.0
