import os
import argparse
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

PROMPT = "The movie was"


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


def build_lm_dataset(ds, tokenizer, max_length=150):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = ds.map(tok, batched=True, remove_columns=ds.column_names)
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})
    tokenized.set_format("torch")
    return tokenized


def finetune_one(ds_text, save_dir, seed=42):
    try:
        os.makedirs(save_dir, exist_ok=True)
        set_seed(seed)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

        train_ds = build_lm_dataset(ds_text, tokenizer, max_length=150)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            warmup_ratio=0.05,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        return model, tokenizer
    except Exception:
        raise


@torch.no_grad()
def generate_samples(model, tokenizer, n=10, max_new_tokens=100):
    try:
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

        outputs = []
        for _ in range(n):
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            outputs.append(text)
        return outputs
    except Exception:
        raise


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--subset_path", type=str, required=True, help="path to imdb_subset")
        parser.add_argument("--output_file", type=str, required=True, help="path to generated_reviews.txt")
        parser.add_argument("--models_dir", type=str, required=True, help="directory to save finetuned models")
        args = parser.parse_args()

        subset = load_or_create_imdb_subset(args.subset_path, seed=42, n=500)

        # 1 = positive, 0 = negative in IMDb
        pos = subset.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(100))
        neg = subset.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(100))

        pos_dir = os.path.join(args.models_dir, "gpt2_positive")
        neg_dir = os.path.join(args.models_dir, "gpt2_negative")

        finetune_one(pos, pos_dir)
        finetune_one(neg, neg_dir)

        # Reload from disk (requirement)
        pos_model = GPT2LMHeadModel.from_pretrained(pos_dir)
        pos_tok = GPT2Tokenizer.from_pretrained(pos_dir)

        neg_model = GPT2LMHeadModel.from_pretrained(neg_dir)
        neg_tok = GPT2Tokenizer.from_pretrained(neg_dir)

        pos_tok.pad_token = pos_tok.eos_token
        neg_tok.pad_token = neg_tok.eos_token

        pos_gen = generate_samples(pos_model, pos_tok, n=10)
        neg_gen = generate_samples(neg_model, neg_tok, n=10)

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("Reviews generated by positive model:\n")
            for i, txt in enumerate(pos_gen, 1):
                f.write(f"{i}. {txt.strip()}\n")

            f.write("\n")

            f.write("Reviews generated by negative model:\n")
            for i, txt in enumerate(neg_gen, 1):
                f.write(f"{i}. {txt.strip()}\n")

    except Exception:
        return


if __name__ == "__main__":
    main()
