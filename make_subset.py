import os
from datasets import load_dataset, load_from_disk

def load_or_create_imdb_subset(subset_dir: str = "imdb_subset", seed: int = 42, n: int = 500):
    """
    If subset_dir exists -> load it.
    Else -> download imdb, shuffle, select n examples, and save to subset_dir.
    Returns a HuggingFace Dataset (train split subset).
    """
    if os.path.exists(subset_dir):
        print(f"[INFO] Loading subset from disk: {subset_dir}")
        subset = load_from_disk(subset_dir)
        return subset

    print("[INFO] Downloading 'imdb' dataset from HuggingFace...")
    dataset = load_dataset("imdb")

    print(f"[INFO] Creating subset: shuffle(seed={seed}) + select({n}) from train split")
    subset = dataset["train"].shuffle(seed=seed).select(range(n))

    print(f"[INFO] Saving subset to disk: {subset_dir}")
    subset.save_to_disk(subset_dir)
    return subset


if __name__ == "__main__":
    subset = load_or_create_imdb_subset("imdb_subset", seed=42, n=500)
    print(subset)
    print("Columns:", subset.column_names)
    print("First example:", subset[0])
