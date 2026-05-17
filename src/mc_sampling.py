import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

def load_model_for_mc(base_name, lora_path):
    print(f"\n[MC模式] 正在加载基座模型: {base_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # 保持词表对齐逻辑一致
    hani_token = "hani_Latn"
    if hani_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [hani_token]})
    base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"[MC模式] 正在挂载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 先整体切到 eval 状态，再强制将所有 Dropout 模块拨回 train 状态
    model.eval() 
    dropout_count = 0
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()  # 激活随机失活路径
            dropout_count += 1
            
    print(f"成功激活隐式集成：共找到并开启 {dropout_count} 个 Dropout 层。")
    return model, tokenizer, device

def generate_pseudo_mc(model, tokenizer, device, input_file, output_file, tgt_lang, max_length=32, batch_size=128):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    results = []
    
    # 每次前向传播由于 Dropout 的随机二进制掩码（Mask）不同，生成的 Token 序列会产生微妙差异
    for i in tqdm(range(0, len(lines), batch_size), desc=f"MC 随机采样 -> {os.path.basename(output_file)}"):
        batch = lines[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=max_length, 
                num_beams=4,
                forced_bos_token_id=forced_bos_token_id
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(res + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--fwd_lora", default="outputs/exp1_nllb/final")      # 借用实验一训练好的正向模型
    parser.add_argument("--rev_lora", default="outputs/exp2_reverse/final")   # 借用实验二训练好的反向模型
    parser.add_argument("--orig_zh", default="data/train.zh")
    parser.add_argument("--orig_hani", default="data/train.hani")
    parser.add_argument("--out_dir", default="augmented_data/exp4_mc_merged")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 启动单反向模型的 3 次独立 MC 随机路径预测（真哈尼语 -> 3 份不同的伪中文）
    if os.path.exists(args.rev_lora):
        model, tok, dev = load_model_for_mc(args.base_model, args.rev_lora)
        for i in range(1, 4):
            out_path = os.path.join(args.out_dir, f"pseudo_rev_mc_{i}.zh")
            generate_pseudo_mc(model, tok, dev, args.orig_hani, out_path, tgt_lang="zho_Hans")
        del model
        torch.cuda.empty_cache()

    # 2. 启动单正向模型的 3 次独立 MC 随机路径预测（真中文 -> 3 份不同的伪哈尼语）
    if os.path.exists(args.fwd_lora):
        model, tok, dev = load_model_for_mc(args.base_model, args.fwd_lora)
        for i in range(1, 4):
            out_path = os.path.join(args.out_dir, f"pseudo_fwd_mc_{i}.hani")
            generate_pseudo_mc(model, tok, dev, args.orig_zh, out_path, tgt_lang="hani_Latn")
        del model
        torch.cuda.empty_cache()

    # 3. 级联合并数据 (1 份原始平行数据 + 3 份反向 MC 伪平行 + 3 份正向 MC 伪平行 = 7 倍规模)
    print("\n正在执行 7 份数据集的内生级联合并...")
    final_zh = os.path.join(args.out_dir, "train.zh")
    final_hani = os.path.join(args.out_dir, "train.hani")
    
    with open(final_zh, 'w', encoding='utf-8') as f_zh, open(final_hani, 'w', encoding='utf-8') as f_hani:
        # 写入原始双语
        f_zh.write(open(args.orig_zh).read())
        f_hani.write(open(args.orig_hani).read())
        
        # 写入反向 MC 生成的 (伪中文_t - 真实哈尼语)
        for i in range(1, 4):
            p_zh = os.path.join(args.out_dir, f"pseudo_rev_mc_{i}.zh")
            f_zh.write(open(p_zh).read())
            f_hani.write(open(args.orig_hani).read())
            
            # 写入正向 MC 生成的 (真实中文 - 伪哈尼语_t)
            p_hani = os.path.join(args.out_dir, f"pseudo_fwd_mc_{i}.hani")
            f_zh.write(open(args.orig_zh).read())
            f_hani.write(open(p_hani).read())

    print(f"🎉 实验四隐式集成语料库构建完毕！文件存放于: {args.out_dir}")

if __name__ == "__main__":
    main()