#!/usr/bin/env python3
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.config_loader import load_config

def generate_pseudo(reverse_model_path, mono_file, output_prefix, batch_size=32, max_length=128):
    model = AutoModelForSeq2SeqLM.from_pretrained(reverse_model_path)
    tokenizer = AutoTokenizer.from_pretrained(reverse_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    with open(mono_file, 'r', encoding='utf-8') as f:
        mono = [line.strip() for line in f if line.strip()]
    print(f"单语句子数: {len(mono)}")
    
    pseudo_zh = []
    for i in tqdm(range(0, len(mono), batch_size)):
        batch = mono[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=max_length, num_beams=4)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        pseudo_zh.extend(decoded)
    
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    out_zh = f"{output_prefix}.zh"
    out_hani = f"{output_prefix}.hani"
    with open(out_zh, 'w', encoding='utf-8') as fz, open(out_hani, 'w', encoding='utf-8') as fh:
        for zh, ha in zip(pseudo_zh, mono):
            fz.write(zh + '\n')
            fh.write(ha + '\n')
    print(f"伪语料保存至 {out_zh} 和 {out_hani}")
    return out_zh, out_hani

def merge_corpora(orig_zh, orig_hani, pseudo_zh, pseudo_hani, out_zh, out_hani):
    with open(orig_zh, 'r', encoding='utf-8') as f: orig_zh_lines = [l.strip() for l in f if l.strip()]
    with open(orig_hani, 'r', encoding='utf-8') as f: orig_hani_lines = [l.strip() for l in f if l.strip()]
    with open(pseudo_zh, 'r', encoding='utf-8') as f: pseudo_zh_lines = [l.strip() for l in f if l.strip()]
    with open(pseudo_hani, 'r', encoding='utf-8') as f: pseudo_hani_lines = [l.strip() for l in f if l.strip()]
    assert len(pseudo_zh_lines) == len(pseudo_hani_lines)
    merged_zh = orig_zh_lines + pseudo_zh_lines
    merged_hani = orig_hani_lines + pseudo_hani_lines
    os.makedirs(os.path.dirname(out_zh), exist_ok=True)
    with open(out_zh, 'w', encoding='utf-8') as fz, open(out_hani, 'w', encoding='utf-8') as fh:
        for zh, ha in zip(merged_zh, merged_hani):
            fz.write(zh + '\n')
            fh.write(ha + '\n')
    print(f"合并后训练集大小: {len(merged_zh)} (原始 {len(orig_zh_lines)} + 伪 {len(pseudo_zh_lines)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="反向模型的配置文件（exp2_reverse.yaml）")
    parser.add_argument("--model", required=True, help="训练好的反向模型目录，如 outputs/exp2_reverse/final")
    parser.add_argument("--mono", required=True, help="哈尼语单语文件路径")
    parser.add_argument("--output_prefix", default="augmented_data/pseudo/pseudo")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    pseudo_zh, pseudo_hani = generate_pseudo(
        args.model, args.mono, args.output_prefix,
        batch_size=args.batch_size, max_length=cfg.data.max_length
    )
    # 合并原始正向训练集（data/train.zh 和 data/train.hani）
    merge_corpora(
        "data/train.zh", "data/train.hani",
        pseudo_zh, pseudo_hani,
        "augmented_data/merged/train.zh", "augmented_data/merged/train.hani"
    )
    # 复制验证集到增强数据目录
    import shutil
    os.makedirs("augmented_data/merged", exist_ok=True)
    shutil.copy("data/valid.zh", "augmented_data/merged/valid.zh")
    shutil.copy("data/valid.hani", "augmented_data/merged/valid.hani")
    print("验证集已复制到 augmented_data/merged/")