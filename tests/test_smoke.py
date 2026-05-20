"""CPU smoke tests (no GPU / large downloads required for config path)."""

from pathlib import Path

import pytest
import torch

from rosa.config import ExperimentConfig
from rosa.eval.utils import extract_boxed_answer, extract_gsm8k_answer, normalize_numeric


def test_config_from_yaml():
    cfg_path = Path(__file__).parent.parent / "examples" / "llama3_metamath" / "config.yaml"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    assert "Llama" in cfg.model_name_or_path
    assert cfg.rosa.lora_r == 16
    assert cfg.rosa.lora_lr == pytest.approx(7e-4)
    assert cfg.training.learning_rate == pytest.approx(2e-4)
    assert cfg.training.bf16 is True


def test_rosa_optimizer_param_groups():
    import torch.nn as nn

    from rosa.training.sft import build_rosa_optimizer, split_rosa_trainable_params

    class FakeRosa(nn.Module):
        def __init__(self):
            super().__init__()
            self.rosa_A = nn.Parameter(torch.zeros(1))
            self.rosa_spa_values = nn.Parameter(torch.zeros(1))

    model = FakeRosa()
    lora, other = split_rosa_trainable_params(model)
    assert len(lora) == 1
    assert len(other) == 1

    opt = build_rosa_optimizer(model, learning_rate=2e-4, lora_lr=7e-4)
    assert opt is not None
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert opt.param_groups[1]["lr"] == pytest.approx(7e-4)

    assert build_rosa_optimizer(model, learning_rate=2e-4, lora_lr=None) is None


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
