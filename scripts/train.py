import wandb
import torch
import numpy as np
import argparse
import os
from typing import Optional
os.environ["TOKENIZERS_PARALLELISM"]="false"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mergoo.models.modeling_llama import LlamaForCausalLM
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    # BitsAndBytesConfig
)
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

print("torch version: ", torch.version.cuda)

from callbacks import ComputeThroughputCallback, TokenCountCallback
from prepare_dataset import prepare_dataset


MAX_TOKENS = 2 * 1000 * 1000

MAX_LENGTH = 4096
# MAX_LENGTH=10
BATCH_SIZE=4
# BATCH_SIZE=1024
BATCH_SIZE=1
# BATCH_SIZE=2
LOGGING_STEPS=2
SAVE_STEPS=100
NUM_GPUS=int(os.environ['WORLD_SIZE'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])

def rank_0_print(*args, **kwargs):
    if LOCAL_RANK == 0:
        print(*args, **kwargs)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, required=True)
    parser.add_argument("--upload_repo_id", type=str)
    parser.add_argument("--tokenizer", type=str, default="team-sanai/unigram_32000")
    parser.add_argument("--include_lm_head", action='store_true')
    parser.add_argument("--ds_config_path", type=str)
    parser.add_argument("--load_8bit", action='store_true')
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()
    print("args: ", args)
    return args

def format_number(num):
    if abs(num) >= 10**12:  # Trillion
        return "{:.2f}T".format(num / 10**12)
    elif abs(num) >= 10**9:  # Billion
        return "{:.2f}B".format(num / 10**9)
    elif abs(num) >= 10**6:  # Million
        return "{:.2f}M".format(num / 10**6)
    else:
        return str(num)

def show_total_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format_number(params)

ratios = {
    "wiki_ja": 1.0,
}
def make_dataset(tokenizer):
    target_list = {
        "wiki_ja": "izumi-lab/wikinews-ja-20230728",
    }
    datasets = {name: load_dataset(path, split="train", num_proc=8) for name, path in target_list.items()}
    ds = []
    # print(datasets)
    for name, dataset in datasets.items():
        rank_0_print(name, ratios[name])
        # ds_part = dataset.select(range(10))
        # ds_part = dataset.shuffle(seed=42).select(range(1000))
        ds_part = dataset.shuffle(seed=42)
        filtered_list = []
        for name in ds_part.column_names:
            if "text" != name:
                filtered_list.append(name)
        ds_part = ds_part.remove_columns(filtered_list)
        ds.append(ds_part)
    combined_dataset = concatenate_datasets(ds)
    return combined_dataset.shuffle(seed=42).train_test_split(test_size=0.1)


class DDPTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        sampler = self._get_train_sampler()
        data_loader = DataLoader(self.train_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=sampler is None,
                                 sampler=sampler,
                                 collate_fn=self.data_collator,
                                 drop_last=self.args.dataloader_drop_last,
                                 )
        return self.accelerator.prepare(data_loader)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=NUM_GPUS, rank=dist.get_rank(), shuffle=True) if NUM_GPUS > 1 else None
        return train_sampler

def main():
    args = parse_arguments()
    # LOCAL_RANK = args.local_rank
    if LOCAL_RANK == 0:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        args.repo_id,
        torch_dtype=torch.float16,
    )

    ## freeze other than gate
    rank_0_print("before freeze: ", show_total_params(model))
    n_weights, n_router_weights  = 0,0
    for name, weight in model.named_parameters():
        print(name, weight.size)
        if ".gate." in name:
            weight.requires_grad_(True)
            n_router_weights += 1
        else:
            weight.requires_grad_(False)
        n_weights += 1
    rank_0_print("n_router_weights", n_router_weights)
    rank_0_print("n_weights", n_weights)
    rank_0_print("after freeze: ", show_total_params(model))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    rank_0_print("--- making dataset ... ---")
    dataset = make_dataset(tokenizer)
    train_dataset = prepare_dataset(dataset["train"], tokenizer)
    test_dataset = prepare_dataset(dataset["test"], tokenizer)

    micro_batch_size = 1
    batch_size = BATCH_SIZE / micro_batch_size
    rank_0_print("--- training start ... ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        seed=42,
        data_seed=42,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size,
        warmup_steps=10,
        evaluation_strategy="steps",
        eval_steps=200,
        weight_decay=0.01,
        optim="adamw_apex_fused",
        # optim="adafactor",
        logging_dir=args.output_dir,
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        learning_rate=6.0e-5,
        # min_lr
        save_strategy="steps" if LOCAL_RANK == 0 else "no",
        save_total_limit=3,
        save_steps=SAVE_STEPS,
        report_to="wandb",
        # bf16=True,
        fp16=True,
        ddp_backend="nccl",
        half_precision_backend="apex",
        deepspeed=args.ds_config_path,
        dataloader_pin_memory=True,
        dataloader_num_workers=16,
        # torch_compile=True,
        # num_workers=16,
        fsdp="full_shard"
    )
    rank_0_print("parallel_mode: ", training_args.parallel_mode)
    rank_0_print("world_size", training_args.world_size)

    show_total_params(model)

    computeThroughput = ComputeThroughputCallback(
        vocab_size=model.config.vocab_size,
        #seq_length=model.config.max_sequence_length,
        seq_length=model.config.max_position_embeddings,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        world_size=NUM_GPUS,
        log_steps=LOGGING_STEPS,
    )
    tokenCounter = TokenCountCallback(max_token_count=MAX_TOKENS)
    trainer = DDPTrainer(
        model=model,
        args=training_args,
        #train_dataset=dataset["train"],
        #eval_dataset=dataset["test"],
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[tokenCounter, computeThroughput]
    )
    trainer.train()

    model.save_pretrained(args.output_dir,
                        save_embedding_layers=args.include_lm_head,
                        is_main_process=LOCAL_RANK==0)

    if args.upload_repo_id:
        print("--- push to hf ---")
        # output_model_id = "team-sanai/llama2_0.1B_lora_sample"
        # model.push_to_hub(args.upload_repo_id, save_embedding_layers=True)
        model.push_to_hub(args.upload_repo_id)

if __name__ == "__main__":
    main()
