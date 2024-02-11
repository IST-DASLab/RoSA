import json
import os.path
import re

import fire
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

WANDB_PROJECT = None

prompt = (
    "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. "
    "This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. "
    "The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']"
    "\n\n### Target sentence:\n{target}\n\n### Meaning representation:\n")
rgx = r'### Meaning representation:\n(.*)'


def batch_eval(model, tokenizer, prompt, rgx, batch):
    model_inputs = tokenizer([prompt.format(target=target) for target in batch], return_tensors="pt", padding=True).to(
        "cuda")
    generated_ids = model.generate(**model_inputs, max_length=512, do_sample=False, num_beams=1)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # print(results)
    results = [re.search(rgx, result).group(1) for result in results]
    results = [result.strip() for result in results]
    return results


@torch.no_grad()
def eval_viggo(model, tokenizer):
    viggo_dataset = load_dataset('GEM/viggo', split='validation')
    dataloader = DataLoader(viggo_dataset, batch_size=16, shuffle=False)

    results = []
    for batch in tqdm(dataloader):
        preds = batch_eval(model, tokenizer, prompt, rgx, batch['target'])
        labels = batch['meaning_representation']
        results.extend([pred.strip() == label.strip() for pred, label in zip(preds, labels)])

    return sum(results) / len(results)


def load_model(model_name_or_path, peft_path, precision):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path if peft_path is None else peft_path)
    
    compute_dtype = torch.bfloat16
    if precision == '4bit':
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    else:
        assert precision == 'bf16', f'precision {precision} not supported'
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=compute_dtype,
        load_in_4bit=precision == '4bit',
        quantization_config=quant_config,
        trust_remote_code=True,
        use_auth_token=True
    )

    if peft_path is not None:
        model = PeftModel.from_pretrained(model, peft_path)

    model.eval()
    return model, tokenizer


def main(
        model_name_or_path,
        peft_path,
        precision='bf16'
):
    out_path = os.path.join(model_name_or_path if peft_path is None else peft_path, 'viggo_eval.txt')
    if os.path.exists(out_path):
        print(f'{out_path} already exists, skipping...')
        return
    model, tokenizer = load_model(model_name_or_path, peft_path, precision)
    acc = eval_viggo(model, tokenizer)
    with open(out_path, 'w') as f:
        metric = {'viggo/validation_acc': acc}
        print(metric)
        json.dump(metric, f)
    
    if WANDB_PROJECT is not None:
        api = wandb.Api()
        run = \
            api.runs(path=f"ist/{WANDB_PROJECT}",
                    filters={"display_name": os.path.basename((model_name_or_path if peft_path is None else peft_path).rstrip('/'))})[0]
        
        run.summary["viggo/validation_acc"] = acc
        run.summary["test_accuracy"] = acc
        run.summary.update()
        run.update()


if __name__ == '__main__':
    fire.Fire(main)
