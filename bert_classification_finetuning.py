import numpy as np
from datasets import load_from_disk
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
import torch

# --------- Load subset ----------
subset = load_from_disk("imdb_subset")

# split ל-train/test
dataset = subset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

# --------- Tokenizer ----------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Trainer מצפה לעמודה בשם "labels"
train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch")
test_ds.set_format("torch")

# --------- Model ----------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# --------- Metric ----------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# --------- Training args ----------
training_args = TrainingArguments(
    output_dir="./bert_output",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    fp16=True,                     # GPU
    logging_steps=10,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer)

# --------- Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --------- Train ----------
trainer.train()

# --------- Evaluate ----------
results = trainer.evaluate()
print("Test accuracy:", results["eval_accuracy"])
