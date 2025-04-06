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
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=None,
            **kwargs,
    ):
        prompt = generate_prompt(instruction)
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
        trimmed_output: str = raw_output.split("### Response:")[1].strip()  # TODO
        return trimmed_output


    save_file = f'experiment/feedback-{args.dataset}.json'  # TODO
    create_dir('experiment/')

    max_new_tokens = args.max_new_tokens
    print(f'        -> max_new_tokens: {max_new_tokens}')

    dataset = load_data(args)
    tokenizer, model = load_model(args)
    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    wrong_answer_data = []
    pbar = tqdm(total=total)
    for idx, data in enumerate(dataset):
        instruction = data.get('instruction')
        input_ = data.get('input')

        outputs = evaluate(instruction, input_, max_new_tokens=max_new_tokens)
        label = data.get('answer')
        flag = False
        if args.dataset.lower() in ['aqua', 'apchem']:  # apchem
            predict = extract_answer_letter(args, outputs, string_to_cut_off_response)
            if label == predict:
                correct += 1
                flag = True
        else:
            if isinstance(label, str):
                label = float(label)
            predict = extract_answer_number(args, outputs, string_to_cut_off_response)
            if abs(label - predict) <= miss:
                correct += 1
                flag = True
        new_data = copy.deepcopy(data)
        new_data['output_pred'] = outputs
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)
        print('\n')
        print('\033[0;35m---------------\033[0;0m')
        print(f'\033[0;37m{instruction}\033[0;0m\n')
        if string_to_cut_off_response is None:
            print(outputs)
            if not flag:
                wrong_answer_data.append({'instruction': instruction, 'output': outputs, 'answer_correct': label, 'answer_given': predict})
        else:
            partitioned = outputs.partition(string_to_cut_off_response)
            print(f'{partitioned[0]}\033[0;90m{partitioned[1]}{partitioned[2]}\033[0;0m')
            if not flag:
                wrong_answer_data.append({'instruction': instruction, 'output': partitioned[0], 'answer_correct': label, 'answer_given': predict})
        print(f'\033[0;36mprediction: {predict}\033[0;0m')
        print(f'\033[0;36mcorrect: {label}\033[0;0m')
        print('\033[0;35m---------------\033[0;0m')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        with open(wrong_answers_filename, 'w+') as f:
            json.dump(wrong_answer_data, f, indent=4)
        # checkpoints/backups
        if idx % 1000 == 0:
            with open(f'{wrong_answers_filename}.checkpoint1000', 'w+') as f:
                json.dump(wrong_answer_data, f, indent=4)
        elif idx % 200 == 0:
            with open(f'{wrong_answers_filename}.checkpoint200', 'w+') as f:
                json.dump(wrong_answer_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction):
    return prompt.get_eval_prompt(instruction)


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP', 'apchem', 'gsm50', 'math50', 'math10ktraintest'],
                        required=True)
    parser.add_argument('--model_tokenizer', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B', 'autotokenizer'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    parser.add_argument('--max_new_tokens', type=int, required=True)

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
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

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
