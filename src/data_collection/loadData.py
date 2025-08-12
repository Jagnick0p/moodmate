from datasets import load_dataset
import pandas as pd
from pathlib import Path

# 1. Define where to save the file (relative to project root)
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Make sure folder exists

# 2. Load the GoEmotions dataset from Hugging Face
# "simplified" means emotions are merged into 27 labels
print("Downloading GoEmotions dataset...")
dataset = load_dataset("go_emotions", "simplified")

# 3. Convert each split (train, validation, test) into pandas DataFrame
for split in ["train", "validation", "test"]:
    print(f"Processing {split} split...")
    df = pd.DataFrame(dataset[split])

    # Get the label names (index → string mapping)
    label_names = dataset[split].features["labels"].feature.names

    # Convert label IDs into label names (multi-label possible)
    def decode_labels(label_ids):
        return [label_names[i] for i in label_ids]

    df["label_names"] = df["labels"].apply(decode_labels)

    # Save CSV file into data/raw/
    save_path = RAW_DATA_DIR / f"goemotions_{split}.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved {split} data to {save_path}")

print("All splits saved in data/raw/")
