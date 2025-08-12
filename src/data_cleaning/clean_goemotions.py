import pandas as pd
import re
import ast
from pathlib import Path

# --------------------
# 1. Paths
# --------------------
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# 2. Text cleaning function
# --------------------
def clean_text(text: str) -> str:
    """
    Cleans the input text for NLP training.
    Steps:
    1. Lowercase text (case-insensitive training)
    2. Remove placeholders like [NAME], [URL]
    3. Remove markdown/HTML artifacts (>, *, _)
    4. Remove URLs (http, https, www)
    5. Remove extra spaces
    """
    # Lowercase
    text = text.lower()

    # Remove placeholders like [NAME], [URL]
    text = re.sub(r"\[.*?\]", "", text)

    # Remove markdown blockquotes and bold/italic markers
    text = re.sub(r">", " ", text)        # blockquote
    text = re.sub(r"\*+", " ", text)      # bold/italic asterisks
    text = re.sub(r"_+", " ", text)       # underscores

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --------------------
# 3. Processing function
# --------------------
def process_split(split_name: str):
    """
    Loads a raw CSV, cleans text, parses labels, and saves processed version.
    """
    print(f"Processing {split_name} split...")
    raw_path = RAW_DATA_DIR / f"goemotions_{split_name}.csv"
    df = pd.read_csv(raw_path)

    # Parse label_names column safely
    df["label_names"] = df["label_names"].apply(lambda x: ast.literal_eval(x))

    # Clean text
    df["text_clean"] = df["text"].apply(clean_text)

    # Save processed CSV
    save_path = PROCESSED_DATA_DIR / f"goemotions_{split_name}_clean.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved cleaned {split_name} data to {save_path}")

# --------------------
# 4. Run for all splits
# --------------------
if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        process_split(split)

    print("All splits processed and saved in data/processed/")
