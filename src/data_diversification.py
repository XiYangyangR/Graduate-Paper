import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

def load_model_and_tokenizer(base_name, lora_path):
    print(f"\n加载基座: {base_name} | LoRA权重: {lora_path}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # 兼容你的哈尼语 Token
    hani_token = "hani_Latn"
    if hani_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [hani_token]})
    base_model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device

def generate_pseudo(model, tokenizer, device, input_file, output_file, tgt_lang, max_length=32, batch_size=128):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    results = []
    
    for i in tqdm(range(0, len(lines), batch_size), desc=f"正在生成 -> {os.path.basename(output_file)}"):
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
            # 你提到的 "q " 乱码如果需要这里去除，可以在这加上 res = res.replace("q ", "") 等后处理逻辑
            f.write(res + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--orig_zh", default="data/train.zh")
    parser.add_argument("--orig_hani", default="data/train.hani")
    parser.add_argument("--out_dir", default="augmented_data/exp3_merged")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [42, 43, 44]
    
    pseudo_zh_files = []
    pseudo_hani_files = []

    # 1. 运行 3 个反向模型：真哈尼 -> 伪中文
    for s in seeds:
        lora_path = f"outputs/exp3_bwd_seed{s}/final"
        out_zh = os.path.join(args.out_dir, f"pseudo_bwd_{s}.zh")
        if os.path.exists(lora_path) and not os.path.exists(out_zh):
            model, tok, dev = load_model_and_tokenizer(args.base_model, lora_path)
            generate_pseudo(model, tok, dev, args.orig_hani, out_zh, tgt_lang="zho_Hans")
        pseudo_zh_files.append(out_zh)

    # 2. 运行 3 个正向模型：真中文 -> 伪哈尼
    for s in seeds:
        lora_path = f"outputs/exp3_fwd_seed{s}/final"
        out_hani = os.path.join(args.out_dir, f"pseudo_fwd_{s}.hani")
        if os.path.exists(lora_path) and not os.path.exists(out_hani):
            model, tok, dev = load_model_and_tokenizer(args.base_model, lora_path)
            generate_pseudo(model, tok, dev, args.orig_zh, out_hani, tgt_lang="hani_Latn")
        pseudo_hani_files.append(out_hani)

    # 3. 终极数据合并 (Data Diversification)
    print("\n开始合并 7 份数据 (1原 + 3反向生成 + 3正向生成)...")
    final_zh = os.path.join(args.out_dir, "train.zh")
    final_hani = os.path.join(args.out_dir, "train.hani")
    
    with open(final_zh, 'w', encoding='utf-8') as f_zh, open(final_hani, 'w', encoding='utf-8') as f_hani:
        # 写入 1 份原始数据
        f_zh.write(open(args.orig_zh).read())
        f_hani.write(open(args.orig_hani).read())
        
        # 写入 3 份反向生成的对齐数据 (伪中文 - 真哈尼)
        for p_zh in pseudo_zh_files:
            if os.path.exists(p_zh):
                f_zh.write(open(p_zh).read())
                f_hani.write(open(args.orig_hani).read()) 
                
        # 写入 3 份正向生成的对齐数据 (真中文 - 伪哈尼)
        for p_hani in pseudo_hani_files:
            if os.path.exists(p_hani):
                f_zh.write(open(args.orig_zh).read()) 
                f_hani.write(open(p_hani).read())

    print(f"数据合并完成！增强后的多样化训练集已保存至 {args.out_dir}")

if __name__ == "__main__":
    main()