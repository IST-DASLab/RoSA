# Robust Adaptation (RoSA)

This repository includes the code for the paper ["RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation"](https://arxiv.org/abs/2401.04679).



## Installation
1. Create a clean environment and activate it:
```
conda create --name rosa python=3.10 -y
conda activate rosa
```

2. Install the latest version of [pytorch](https://pytorch.org/) compatible with your system (preferably using conda instead of pip to ensure all the dependencies are installed properly). For example, if you have cuda version 11.8, run the following command:
```
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install this repository, which is a fork of [MosaicML's llm-foundry](https://github.com/mosaicml/llm-foundry) including the experiments presented in the paper:
```
git clone https://github.com/IST-DASLab/RoSA.git && cd RoSA
pip install -e .
```

4. Install the [*spops*](https://github.com/IST-DASLab/spops) library, which we use under the hood to perform sparse operations: 
```
pip install git+https://github.com/IST-DASLab/spops.git
```

5. Install RoSA's [integration into the PEFT library](https://github.com/IST-DASLab/peft-rosa) by running:
```
pip install git+https://github.com/IST-DASLab/peft-rosa.git
```

6. For evaluation, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Run the following command to install the compatible version:
```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git reset --hard 2c18e367c6ded428863cd1fd4cf9558ca49d68dc
pip install -e .
cd ..
```

## Quick Runs
First things first, activate the environment and cd into `scripts/train/`
```
conda activate rosa
cd scripts/train/
```

To run quick experiments, simply run any of the following commands, each of which corresponds to one of the single-epoch experiments in the paper:

```
# best QRoSA on gsm8k
CUDA_VISIBLE_DEVICES=0 bash scripts/llama2-7b/restart_7b_gsm_4bit.sh

# best RoSA on gsm8k
CUDA_VISIBLE_DEVICES=0 bash scripts/llama2-7b/restart_7b_gsm_bf16.sh

# best QRoSA on sql
CUDA_VISIBLE_DEVICES=0 bash scripts/llama2-7b/restart_7b_sql_4bit.sh

# best RoSA on sql
CUDA_VISIBLE_DEVICES=0 bash scripts/llama2-7b/restart_7b_sql_bf16.sh

# best QRoSA on viggo
CUDA_VISIBLE_DEVICES=0 bash scripts/llama2-7b/restart_7b_viggo_4bit.sh

# best RoSA on viggo
CUDA_VISIBLE_DEVICES=0 bash scripts/llama2-7b/restart_7b_viggo_bf16.sh
```

These scripts essentially run `scripts/restarter_llama2.sh` with different hyper-parameters. `scripts/restarter_llama2.sh` takes care of low-rank adapter warmup and restarting the training after mask generation. Feel free to tweak the hyper-parameters in any of these scripts.

## TODO
- Include memory and time analysis
- Include accuracies
