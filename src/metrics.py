import evaluate
import numpy as np

def compute_metrics(eval_pred, tokenizer):
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 将 -100 替换为 pad_token_id 以解码参考译文
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]

    bleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_result = chrf.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"]
    }