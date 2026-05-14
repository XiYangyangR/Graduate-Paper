import os
import sys
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.config_loader import load_config
from src.data_utils import load_parallel_data, preprocess_function
from src.metrics import compute_metrics
import numpy as np

def evaluate_on_test(model_path, config, test_src_file, test_tgt_file, batch_size=16):
    print(f"加载模型: {model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.resize_token_embeddings(len(tokenizer))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = load_parallel_data(test_src_file, test_tgt_file)

    all_preds = []
    all_labels = []

    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating on test set"):
        batch = test_dataset.select(range(i, min(i + batch_size, len(test_dataset))))
        processed = batch.map(
            lambda x: preprocess_function(x, tokenizer, config.model.src_lang, config.data.max_length),
            batched=True
        )
        input_ids = torch.tensor(processed["input_ids"]).to(device)
        attention_mask = torch.tensor(processed["attention_mask"]).to(device)

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(config.model.tgt_lang) 
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.data.max_length,
                num_beams=4,
                forced_bos_token_id=forced_bos_token_id

            )
        all_preds.append(generated_ids.cpu().numpy())
        all_labels.append(np.array(processed["labels"]))

    predictions = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    eval_pred = (predictions, labels)

    metrics = compute_metrics(eval_pred, tokenizer)
    return metrics

if __name__ == "__main__":
    if len(sys.argv) not in [3, 5]:
        print("用法: python -m src.evaluate <config_path> <model_dir> [<test_src> <test_tgt>]")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    model_dir = sys.argv[2]

    if len(sys.argv) == 5:
        test_src = sys.argv[3]
        test_tgt = sys.argv[4]
    else:
        test_src = cfg.data.test_src
        test_tgt = cfg.data.test_tgt

    scores = evaluate_on_test(model_dir, cfg, test_src, test_tgt, batch_size=cfg.evaluation.test_batch_size)

    print("\n=== Test Set Results ===")
    print(f"BLEU: {scores['bleu']:.2f}")
    print(f"chrF++: {scores['chrf']:.2f}")

    result_file = os.path.join(model_dir, "test_results.json")
    with open(result_file, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"结果已保存至 {result_file}")