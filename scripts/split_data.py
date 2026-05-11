#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import argparse

def load_parallel_corpus(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"行数不一致: {src_file} ({len(src_lines)}) vs {tgt_file} ({len(tgt_lines)})")
    return src_lines, tgt_lines

def split_indices(n, ratios, seed):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train+n_valid]
    test_idx = indices[n_train+n_valid:]
    return train_idx, valid_idx, test_idx

def write_split(src_lines, tgt_lines, indices, prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    src_out = os.path.join(output_dir, f"{prefix}.zh")
    tgt_out = os.path.join(output_dir, f"{prefix}.hani")
    with open(src_out, 'w', encoding='utf-8') as f_src, open(tgt_out, 'w', encoding='utf-8') as f_tgt:
        for idx in indices:
            f_src.write(src_lines[idx] + '\n')
            f_tgt.write(tgt_lines[idx] + '\n')
    print(f"已写入 {len(indices)} 对: {src_out} / {tgt_out}")

def main():
    parser = argparse.ArgumentParser(description="划分平行语料为训练/验证/测试集")
    parser.add_argument("--src", required=True, help="源语言文件（中文全量）")
    parser.add_argument("--tgt", required=True, help="目标语言文件（哈尼语全量）")
    parser.add_argument("--output", default="data", help="输出目录")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="train/valid/test 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    src_lines, tgt_lines = load_parallel_corpus(args.src, args.tgt)
    n_total = len(src_lines)
    print(f"总句对数: {n_total}")

    train_idx, valid_idx, test_idx = split_indices(n_total, args.ratios, args.seed)
    print(f"训练集: {len(train_idx)}")
    print(f"验证集: {len(valid_idx)}")
    print(f"测试集: {len(test_idx)}")

    write_split(src_lines, tgt_lines, train_idx, "train", args.output)
    write_split(src_lines, tgt_lines, valid_idx, "valid", args.output)
    write_split(src_lines, tgt_lines, test_idx, "test", args.output)

    print("\n完成！生成文件：")
    print(f"  {args.output}/train.zh / train.hani")
    print(f"  {args.output}/valid.zh / valid.hani")
    print(f"  {args.output}/test.zh / test.hani")

if __name__ == "__main__":
    main()