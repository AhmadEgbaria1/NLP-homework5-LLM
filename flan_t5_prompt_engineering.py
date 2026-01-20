import re
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets
OUT_FILE = "flan_t5_imdb_results.txt"

def normalize_label(text: str) -> str:
    """
    Must return exactly 'positive' or 'negative' or '' (invalid).
    We ignore case and spaces. If the model outputs extra text, we try to extract.
    """
    if text is None:
        return ""
    t = text.strip().lower()

    # exact match
    if t in ("positive", "negative"):
        return t

    # try to extract the first occurrence
    m = re.search(r"\b(positive|negative)\b", t)
    if m:
        return m.group(1)

    return ""  # invalid output per instructions

def make_prompts(review_text: str) -> dict:
    # ZERO-SHOT
    zero_shot = (
        "Classify the sentiment of the following movie review as positive or negative.\n"
        "Output ONLY one word: positive or negative.\n\n"
        f"Review:\n{review_text}"
    )

    # FEW-SHOT (2 examples: 1 pos, 1 neg) - examples are short and clear
    few_shot = (
        "Classify the sentiment of each movie review as positive or negative.\n"
        "Output ONLY one word: positive or negative.\n\n"
        "Example 1:\n"
        "Review: I loved this movie. The acting was great and the story was amazing.\n"
        "Sentiment: positive\n\n"
        "Example 2:\n"
        "Review: This movie was terrible. The plot was boring and the acting was bad.\n"
        "Sentiment: negative\n"
         "Example 3:\n"
        "Review: At first I thought it would be boring and I almost turned it off, but the story gets better and the ending was great. I really enjoyed it.\n"
        "Sentiment: positive\n"
        "Now classify this review:\n\n"
        f"Review: {review_text}\n"
        "Sentiment:"
    )

    # INSTRUCTION-BASED (more explicit constraints)
    instruction_based = (
        "You are a sentiment classifier for movies reviews.\n"
        "Task: Determine if the movie review is positive or negative.\n"
        "Rules:\n"
        "1) Output exactly one word: positive OR negative.\n"
        "2) Do not output anything else.\n\n"
        f"Movie review:\n{review_text}\n\n"
        "Answer:"
    )

    return {
        "zero": zero_shot,
        "few": few_shot,
        "inst": instruction_based,
    }

@torch.no_grad()
def predict_label(model, tokenizer, prompt: str, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=3,     # enough for 'positive'/'negative'
        do_sample=False       # deterministic
    )
    txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return normalize_label(txt)

def main():
    # 1) Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2) Load your saved subset and sample 50 stratified (25 pos + 25 neg)
    subset = load_from_disk("imdb_subset")
    pos = subset.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(25))
    neg = subset.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(25))


    sampled = concatenate_datasets([pos, neg]).shuffle(seed=42)
    # mix them

    # Tracking accuracy
    total = len(sampled)
    correct_zero = 0
    correct_few = 0
    correct_inst = 0

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for i, ex in enumerate(sampled, start=1):
            review_text = ex["text"]
            true_label = "positive" if ex["label"] == 1 else "negative"

            prompts = make_prompts(review_text)

            pred_zero = predict_label(model, tokenizer, prompts["zero"], device)
            pred_few = predict_label(model, tokenizer, prompts["few"], device)
            pred_inst = predict_label(model, tokenizer, prompts["inst"], device)

            # invalid outputs count as wrong (per instructions)
            if pred_zero == true_label:
                correct_zero += 1
            if pred_few == true_label:
                correct_few += 1
            if pred_inst == true_label:
                correct_inst += 1

            # 4) Write results in the required format
            f.write(f">Review {i}: {review_text}\n")
            f.write(f">Review {i} true label: {true_label}\n")
            f.write(f">Review {i} zero-shot: {pred_zero if pred_zero else 'INVALID_OUTPUT'}\n")
            f.write(f">Review {i} few-shot: {pred_few if pred_few else 'INVALID_OUTPUT'}\n")
            f.write(f">Review {i} instruction-based: {pred_inst if pred_inst else 'INVALID_OUTPUT'}\n\n")

        # 5) Accuracy summary at the end
        acc_zero = correct_zero / total
        acc_few = correct_few / total
        acc_inst = correct_inst / total

        f.write("============================================================\n")
        f.write(f"Zero-shot accuracy: {acc_zero:.4f}\n")
        f.write(f"Few-shot accuracy: {acc_few:.4f}\n")
        f.write(f"Instruction-based accuracy: {acc_inst:.4f}\n")

    print(f"[DONE] Wrote results to: {OUT_FILE}")
    print(f"Zero-shot accuracy: {acc_zero:.4f}")
    print(f"Few-shot accuracy: {acc_few:.4f}")
    print(f"Instruction-based accuracy: {acc_inst:.4f}")

if __name__ == "__main__":
    main()
