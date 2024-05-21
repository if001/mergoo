import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM as AutoModelForCausalLMHF
)
from datasets import load_dataset
from mergoo.models.modeling_llama import LlamaForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="team-sanai/llama2_7B_pretrain")
    # parser.add_argument("--peft_id", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="team-sanai/unigram_32000")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt_file", type=str)

    args = parser.parse_args()
    print("args: ", args)
    return args

def gen(model, tokenizer, generationConfig, prompt, ans=None, include_ans=False):
    print("input: ", prompt)
    encoded_text = tokenizer.tokenize(prompt)
    print("encode", len(encoded_text), encoded_text)

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    print("input_ids", input_ids)

    model_outputs = model.generate(
                input_ids=input_ids,
                generation_config=generationConfig
    )
    model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    print("output: ", model_text_output)
    if include_ans:
        if ans in model_text_output:
            print("success")
        else:
            print("fail")
    print("="*50)


def check_next_prob(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(logits, dim=-1)

    ## 上位5つのトークンの確率とトークン自体を取得
    top_5_tokens = torch.topk(probabilities, 10)
    ## それぞれのトークンとその確率を表示
    for i, token_id in enumerate(top_5_tokens.indices):
        token = tokenizer.decode([token_id])
        print(f"Rank {i+1}: Token = {token}, Probability = {top_5_tokens.values[i]:.4f}")


def main():
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        args.repo_id,
        torch_dtype=torch.float16,
    )

    tanuki_model = AutoModelForCausalLMHF.from_pretrained(
        "hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160",
        torch_dtype=torch.float16,
    )
    generationConfig = GenerationConfig(do_sample=True, repetition_penalty=1.1, temperature=0.2, max_new_tokens=30)

    if args.prompt:
        gen(model, tokenizer, generationConfig, args.prompt)
        gen(tanuki_model, tokenizer, generationConfig, args.prompt)

    
    dataset = load_dataset("csv", data_files=args.prompt_file, split="train")
    for v in dataset:
        include_ans = int(v["include_ans"]) == 1
        gen(model, tokenizer, generationConfig, v["text"], v["ans"], include_ans)
        gen(tanuki_model, tokenizer, generationConfig, v["text"], v["ans"], include_ans)

if __name__ == "__main__":
    main()
