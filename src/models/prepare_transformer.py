import ast
from pathlib import Path
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "processed"   # cleaned input
OUT = PROJECT_ROOT / "data" / "processed" / "transformer"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128

def load_split(name):
    df = pd.read_csv(RAW / f"goemotions_{name}_clean.csv")
    df["label_names"] = df["label_names"].apply(lambda s: ast.literal_eval(s))
    return df

def tokenize_texts(tokenizer, texts):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    # enc contains input_ids and attention_mask (token_type_ids not used by DistilBERT)
    return enc["input_ids"], enc["attention_mask"]

def main():
    print(f"Loading cleaned splits and tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df_train = load_split("train")
    df_val   = load_split("validation")
    df_test  = load_split("test")

    # Multi-label binarizer (fit on TRAIN ONLY)
    print("Fitting label binarizer on train labels...")
    mlb = MultiLabelBinarizer()
    y_train = torch.tensor(mlb.fit_transform(df_train["label_names"]), dtype=torch.float32)
    y_val   = torch.tensor(mlb.transform(df_val["label_names"]), dtype=torch.float32)
    y_test  = torch.tensor(mlb.transform(df_test["label_names"]), dtype=torch.float32)

    # Tokenize texts
    print("Tokenizing text_clean with transformer tokenizer...")
    train_ids, train_mask = tokenize_texts(tokenizer, df_train["text_clean"].astype(str))
    val_ids, val_mask     = tokenize_texts(tokenizer, df_val["text_clean"].astype(str))
    test_ids, test_mask   = tokenize_texts(tokenizer, df_test["text_clean"].astype(str))

    # Save as .pt files (PyTorch tensors)
    print("Saving tokenized tensors...")
    torch.save({
        "input_ids": train_ids,
        "attention_mask": train_mask,
        "labels": y_train
    }, OUT / "train.pt")

    torch.save({
        "input_ids": val_ids,
        "attention_mask": val_mask,
        "labels": y_val
    }, OUT / "val.pt")

    torch.save({
        "input_ids": test_ids,
        "attention_mask": test_mask,
        "labels": y_test
    }, OUT / "test.pt")

    # Save label binarizer classes for decoding predictions later
    classes_path = OUT / "label_classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        for label in mlb.classes_:
            f.write(label + "\n")

    print("Done. Files saved under data/processed/transformer/")
    print("   - train.pt / val.pt / test.pt (PyTorch tensors)")
    print("   - label_classes.txt (27 labels in order)")

if __name__ == "__main__":
    main()
