import torch
import datasets
import warnings

from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from trl.trainer.utils import ConstantLengthDataset

def prepare_dataset(
        dataset,
        tokenizer,
        packing = True,
        dataset_text_field = "text",
        max_seq_length = 4096,
        formatting_func = None,
        num_of_sequences = 1024,
        chars_per_token = 3.6,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if skip_prepare_dataset:
            return dataset

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored."
                )

            return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(
            dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, )
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        if not packing:
            assert ValueError("not impl")
            # return self._prepare_non_packed_dataloader(
            #     tokenizer,
            #     dataset,
            #     dataset_text_field,
            #     max_seq_length,
            #     formatting_func,
            #     add_special_tokens,
            #     remove_unused_columns,
            # )

        else:
            return _prepare_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                num_of_sequences,
                chars_per_token,
                formatting_func,
                append_concat_token,
                add_special_tokens,
            )
        

def _prepare_packed_dataloader(
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

            constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            if isinstance(dataset, datasets.IterableDataset):
                return constant_length_iterator

            def data_generator(constant_length_iterator):
                yield from constant_length_iterator

            try:
                import time
                start = time.time()
                packed_dataset = Dataset.from_generator(
                        data_generator, num_proc=20, gen_kwargs={"constant_length_iterator": constant_length_iterator}
                )
                end = time.time()
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence."
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )


def prepare_dataset2(dataset, tokenizer):
    max_tokens = 4096

    tokenized_dataset = Dataset.from_dict({"input_ids": [], "labels": []})
    trains = []
    input_ids = []
    labels = []
    from tqdm import tqdm
    for v in tqdm(dataset):
        tokenized_text = tokenizer(v["text"], add_special_tokens=False)['input_ids']

        remaining_space = max_tokens - len(input_ids)
        # print(len(input_ids)) 
        if len(input_ids) + len(tokenized_text) < max_tokens:
            input_ids.extend([tokenizer.eos_token_id] + tokenized_text)
        else:
            ## len(input_ids) + len(tokenized_text) > max_tokens
            remaining_space = max_tokens - len(input_ids)
            input_ids.extend(tokenized_text[:remaining_space])
            labels = input_ids[:]
            tokenized_dataset = tokenized_dataset.add_item({'input_ids': input_ids, 'labels': labels})
            input_ids = []
            labels = []
            remaining_tokens = tokenized_text[remaining_space:]
            while True:
                input_ids.extend(remaining_tokens[:max_tokens])
                remaining_tokens = remaining_tokens[max_tokens:]
                if len(remaining_tokens) == 0:
                    break
    return tokenized_dataset

def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160")
    # ds = load_dataset("json", data_files="/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl", split="train")
    ds = load_dataset("json", data_files="/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl", split="train")
    n = int(len(ds)/1000)
    ds = ds.select(range(n))
    print(ds)
    
    ds2 = prepare_dataset(ds, tokenizer)
    print(ds2)
    exit(0)
    for v in ds2:
        print(v["input_ids"])
        print(tokenizer.decode(v["input_ids"]))
        print("--")

if __name__ == "__main__":
    main()
