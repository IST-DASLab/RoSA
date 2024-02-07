import os.path

import fire
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main(model_path):
    model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    print('merging...')
    model = model.merge_and_unload()
    print('saving...')
    model.save_pretrained(os.path.join(model_path, 'merged'))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(os.path.join(model_path, 'merged'))


if __name__ == '__main__':
    fire.Fire(main)
