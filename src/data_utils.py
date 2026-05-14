from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def load_parallel_data(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]
    assert len(src_lines) == len(tgt_lines), f"行数不一致: {src_file} vs {tgt_file}"
    return Dataset.from_dict({"source": src_lines, "target": tgt_lines})

def preprocess_function(examples, tokenizer, max_length, src_lang=None, tgt_lang=None):
    if src_lang is not None and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if tgt_lang is not None and hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang
    
    model_inputs = tokenizer(
        examples["source"], 
        text_target=examples["target"],
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    return model_inputs

def get_tokenized_datasets(config, include_test=False):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    train_dataset = load_parallel_data(config.data.train_src, config.data.train_tgt)
    valid_dataset = load_parallel_data(config.data.valid_src, config.data.valid_tgt)
    datasets = {"train": train_dataset, "validation": valid_dataset}
    
    if include_test:
        test_dataset = load_parallel_data(config.data.test_src, config.data.test_tgt)
        datasets["test"] = test_dataset
    
    datasets = DatasetDict(datasets)
    
    # 从 config 中读取 src_lang 和 tgt_lang（可能为 None）
    src_lang = getattr(config.model, "src_lang", None)
    tgt_lang = getattr(config.model, "tgt_lang", None)
    
    tokenized_datasets = datasets.map(
        lambda x: preprocess_function(x, tokenizer, config.data.max_length, src_lang, tgt_lang),
        batched=True
    )
    
    if include_test:
        return tokenized_datasets, tokenizer
    else:
        return DatasetDict({k: tokenized_datasets[k] for k in ["train", "validation"]}), tokenizer