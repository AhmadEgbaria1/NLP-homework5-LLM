import re
import argparse
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Normalize model output to exactly "positive" or "negative"
# If the model outputs extra text, we try to extract the label
def normalize_label(text: str) -> str:
    if text is None:
        return ""
    t = text.strip().lower()

    # Perfect match
    if t in ("positive", "negative"):
        return t

    # Try to find the label inside longer text
    m = re.search(r"\b(positive|negative)\b", t)
    return m.group(1) if m else ""


# Build all prompt variants for a single review
def make_prompts(review_text: str) -> dict:
    # Zero-shot prompt: only instructions, no examples
    zero_shot = (
        "Classify the sentiment of the following movie review as positive or negative.\n"
        "Output ONLY one word: positive or negative.\n\n"
        f"Review:\n{review_text}"
    )

    # Few-shot prompt with exactly 2 examples:
    few_shot = (
        "Classify the sentiment of each movie review as positive or negative.\n"
        "Output ONLY one word: positive or negative.\n\n"
        "Example 1:\n"
        "Review: At first I thought it would be boring and I almost turned it off, but the story gets better and the ending was great. I really enjoyed it.\n"
        "Sentiment: positive\n"
        "Example 2:\n"
        "Review: I hated this movie. It was boring, poorly acted, and a complete waste of time.\n"
        "Sentiment: negative\n\n"
        "Now classify this review:\n\n"
        f"Review: {review_text}\n"
        "Sentiment:"
    )

    # Instruction-based prompt: explicit task description, no examples
    instruction_based = (
        "Determine the OVERALL sentiment of the following movie review.\n"
        "Focus on the reviewer's final opinion.\n"
        "Output only one word: positive or negative.\n\n"
        f"Review:\n{review_text}\n\n"
        "Answer:"
    )

    return {"zero": zero_shot, "few": few_shot, "inst": instruction_based}


# Run the model on a single prompt and return the normalized label
@torch.no_grad()
def predict_label(model, tokenizer, prompt: str, device: str) -> str:
    # Tokenize and truncate long reviews
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Deterministic generation (no sampling)
    out_ids = model.generate(**inputs, max_new_tokens=3, do_sample=False)

    # Decode output tokens to text
    txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return normalize_label(txt)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("imdb_subset_path", help="Path to the saved IMDb subset (load_from_disk folder)")
    parser.add_argument("output_path", help="Path to output txt file (e.g., flan_t5_imdb_results.txt)")
    args = parser.parse_args()

    # Load FLAN-T5 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load IMDb subset from disk
    subset = load_from_disk(args.imdb_subset_path)

    # Create a balanced sample: 25 positive + 25 negative
    pos = subset.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(25))
    neg = subset.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(25))
    sampled = concatenate_datasets([pos, neg]).shuffle(seed=42)

    total = len(sampled)
    correct_zero = correct_few = correct_inst = 0

    # Write predictions and results to output file
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(sampled, start=1):
            review_text = ex["text"]
            true_label = "positive" if ex["label"] == 1 else "negative"

            prompts = make_prompts(review_text)

            pred_zero = predict_label(model, tokenizer, prompts["zero"], device)
            pred_few = predict_label(model, tokenizer, prompts["few"], device)
            pred_inst = predict_label(model, tokenizer, prompts["inst"], device)

            # Count correct predictions
            if pred_zero == true_label:
                correct_zero += 1
            if pred_few == true_label:
                correct_few += 1
            if pred_inst == true_label:
                correct_inst += 1

            # Write per-review results
            f.write(f"Review {i}: {review_text}\n")
            f.write(f"Review {i} true label: {true_label}\n")
            f.write(f"Review {i} zero-shot: {pred_zero if pred_zero else 'INVALID_OUTPUT'}\n")
            f.write(f"Review {i} few-shot: {pred_few if pred_few else 'INVALID_OUTPUT'}\n")
            f.write(f"Review {i} instruction-based: {pred_inst if pred_inst else 'INVALID_OUTPUT'}\n\n")

        # Final accuracy summary
        acc_zero = correct_zero / total
        acc_few = correct_few / total
        acc_inst = correct_inst / total

   #     f.write("============================================================\n")
   #     f.write(f"Zero-shot accuracy: {acc_zero:.4f}\n")
   #     f.write(f"Few-shot accuracy: {acc_few:.4f}\n")
   #     f.write(f"Instruction-based accuracy: {acc_inst:.4f}\n")

    # Print summary to console
   # print(f"[DONE] Wrote results to: {args.output_path}")
   # print(f"Zero-shot accuracy: {acc_zero:.4f}")
   # print(f"Few-shot accuracy: {acc_few:.4f}")
   # print(f"Instruction-based accuracy: {acc_inst:.4f}")


if __name__ == "__main__":
    main()
