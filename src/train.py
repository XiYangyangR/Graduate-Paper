import os
import sys
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from src.config_loader import load_config
from src.data_utils import get_tokenized_datasets
from src.metrics import compute_metrics
from src.utils import set_seed

def train(config):
    set_seed(42)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model.name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    hani_token="hani_Latn"
    if hani_token not in tokenizer.get_vocab():
        print(f"Adding new token: {hani_token}")
        tokenizer.add_special_tokens({'additional_special_tokens': [hani_token]})
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets, _ = get_tokenized_datasets(config, include_test=False)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.output_dir,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        weight_decay=config.training.weight_decay,
        num_train_epochs=config.training.num_train_epochs,
        predict_with_generate=config.training.predict_with_generate,
        logging_dir=os.path.join(config.training.output_dir, "logs"),
        logging_steps=config.training.logging_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16 and torch.cuda.is_available(),
        report_to=config.training.report_to,
        warmup_ratio=config.training.get("warmup_ratio", 0.1),

        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    def wrapped_compute_metrics(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=wrapped_compute_metrics,
    )

    trainer.train()

    final_dir = os.path.join(config.training.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    return trainer

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python -m src.train <config_path>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    train(cfg)