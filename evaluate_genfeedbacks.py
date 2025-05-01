from __future__ import annotations

import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

import prompt

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main():
    args = parse_args()

    def evaluate(
            dataset_entry,
            temperature=0.6,
            #temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=None,
            **kwargs,
    ):
        prompt = generate_prompt(dataset_entry)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        s = generation_output.sequences[0]
        raw_output: str = tokenizer.decode(s)
        return raw_output


    save_file = args.save_file
    print(f'        -> save_file: {save_file}')
    create_dir('experiment/')
    max_new_tokens = args.max_new_tokens
    print(f'        -> max_new_tokens: {max_new_tokens}')

    dataset = load_data(args)
    tokenizer, model = load_model(args)
    total = len(dataset)
    output_data = []
    pbar = tqdm(total=total)
    for idx, data in enumerate(dataset):

        outputs = evaluate(data, max_new_tokens=max_new_tokens)

        output_data.append({**data, 'raw_output': outputs})
        print('\n')
        print('\033[0;35m---------------\033[0;0m')
        print(f'\033[0;37m{outputs}\033[0;0m\n')
        print('\033[0;35m---------------\033[0;0m')
        print(f'\rtest:{idx + 1}/{total}')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=2)

        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(data_point):
    return (
        f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n'
        f'Below is a math problem. A student answered the problem incorrectly. Their answer and explanation are below. '
        f'The correct answer and explanation are also listed below. Please offer feedback to the student '
        f'explaining why their answer is incorrect. '
        f"Don't just repeat the correct answer; tell the student what they did wrong.\n"
        f'\n'
        f'#### PROBLEM\n'
        f'{data_point["instruction"]}\n'
        f'\n'
        f'#### CORRECT ANSWER\n'
        f'{data_point["answer_correct"]}\n'
        f'\n'
        f'#### CORRECT EXPLANATION\n'
        f'{data_point["output_correct"]}\n'
        f'\n'
        f'#### STUDENT ANSWER\n'
        f'{data_point["answer_given"]}\n'
        f'\n'
        f'#### STUDENT EXPLANATION\n'
        f'{data_point["output"]}\n'
        f'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
        f'<think>\n'
    )


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'{args.dataset}'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True)
    parser.add_argument('--model_tokenizer', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B', 'autotokenizer'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=False)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    parser.add_argument('--max_new_tokens', type=int, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--no_extra_weights', action='store_true', default=False)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.base_model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        print('warning: no lora weights')

    load_8bit = args.load_8bit
    if args.model_tokenizer == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        if not args.no_extra_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"":0}
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if not args.no_extra_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if not args.no_extra_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str, string_to_cut_off_response: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "gsm50", "math50", "math10ktraintest"]:  # gsm50, math50, math10ktraintest
        sentence = sentence.replace(',', '')
        if string_to_cut_off_response:
            sentence = sentence.partition(string_to_cut_off_response)[0]
        pred = [s for s in re.findall(r'[^0-9A-Za-z](-?[0-9,]+\.?[0-9,]*)', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1].replace(',', ''))
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str, string_to_cut_off_response: str) -> str:
    sentence_ = sentence.strip()
    if string_to_cut_off_response:
        sentence_ = sentence_.partition(string_to_cut_off_response)[0]
    # Yes-overlapping matches for [startline/whitespace/punctuation][letter][endline/whitespace/punctuation]
    pred_answers = re.findall(r'(?=(^|\.|,|;| )([ABCDE])($|\.|,|;| ))', sentence_)
    #pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        print(f'  Debug:  grading:  using: {pred_answers[-1]}')
        # pred_answers[-1] e.g. = (' ', 'A', '.')
        return pred_answers[-1][1]
    else:
        return ''


if __name__ == "__main__":
    fire.Fire(main)
