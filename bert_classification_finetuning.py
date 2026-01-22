import argparse
import os
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
import torch


def load_or_create_imdb_subset(subset_dir: str, seed: int = 42, n: int = 500):
    """
    If subset_dir exists -> load it.
    Else -> download imdb, shuffle, select n examples, and save to subset_dir.
    Returns a HuggingFace Dataset (train split subset).
    """
    try:
        if os.path.exists(subset_dir):
            return load_from_disk(subset_dir)

        dataset = load_dataset("imdb")
        subset = dataset["train"].shuffle(seed=seed).select(range(n))
        subset.save_to_disk(subset_dir)
        return subset
    except Exception:
        raise


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("imdb_subset_path", help="Path to the saved IMDb subset (load_from_disk folder)")
        args = parser.parse_args()

        subset = load_or_create_imdb_subset(args.imdb_subset_path, seed=42, n=500)

        dataset = subset.train_test_split(test_size=0.2, seed=42)
        train_ds = dataset["train"]
        test_ds = dataset["test"]

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True)

        train_ds = train_ds.map(tokenize, batched=True)
        test_ds = test_ds.map(tokenize, batched=True)

        train_ds = train_ds.rename_column("label", "labels")
        test_ds = test_ds.rename_column("label", "labels")

        train_ds.set_format("torch")
        test_ds.set_format("torch")

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )

        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return accuracy.compute(predictions=preds, references=labels)

        training_args = TrainingArguments(
            output_dir="./bert_output",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            report_to="none"
        )

        data_collator = DataCollatorWithPadding(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        results = trainer.evaluate()
        print("Test accuracy:", results["eval_accuracy"])
    except Exception:
        return


if __name__ == "__main__":
    main()
