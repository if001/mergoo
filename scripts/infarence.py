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

def load_prompt(prompt_file = None, prompt = None):
    if prompt:
        return [prompt]
    li = []
    with open(prompt_file) as f:
        li = f.readlines()
    li2 = []
    for l in li:
        li2.append(l.strip())
    return li2

def main():
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        args.repo_id,
        torch_dtype=torch.float16,
    )

    #for name, weight in model.named_parameters():
    #    print(name, weight[:10])

    ht_model = AutoModelForCausalLMHF.from_pretrained(
        "hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160",
        torch_dtype=torch.float16,
    )

    prompt = load_prompt(prompt_file=args.prompt_file, prompt=args.prompt)
    for v in prompt:
        print("input: ", v)
        input_ids = tokenizer(v, return_tensors="pt", add_special_tokens=False).input_ids
        
        #outputs = model(input_ids=input_ids)
        #logits = outputs.logits[0, -1, :]
        #probabilities = torch.softmax(logits, dim=-1)

        # 上位5つのトークンの確率とトークン自体を取得
        #top_5_tokens = torch.topk(probabilities, 10)
        # それぞれのトークンとその確率を表示
        #for i, token_id in enumerate(top_5_tokens.indices):
        #    token = tokenizer.decode([token_id])
        #    print(f"Rank {i+1}: Token = {token}, Probability = {top_5_tokens.values[i]:.4f}")
        #exit()

        generationConfig = GenerationConfig(do_sample=True, repetition_penalty=1.1, temperature=0.2, max_new_tokens=30)

        peft_model_outputs = model.generate(
                input_ids=input_ids,
                generation_config=generationConfig
        )
        peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
        _t = tokenizer.tokenize(v)
        print("encode", len(_t), _t)
        print("input_ids", input_ids)
        print("output: ", peft_model_text_output)
        print("="*100)

        ht_model_outputs = ht_model.generate(
            input_ids=input_ids, 
            generation_config=generationConfig
        )
        ht_model_text_output = tokenizer.decode(ht_model_outputs[0], skip_special_tokens=True)
        print("ht output: ", ht_model_text_output)
        print("="*100)
        print("*"*200)
if __name__ == "__main__":
    main()
