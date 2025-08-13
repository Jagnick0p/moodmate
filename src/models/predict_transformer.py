import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Adjust these paths if needed
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "transformer" / "DistilBERT" / "model"
CLASSES_FILE = PROJECT_ROOT / "data" / "processed" / "transformer" / "label_classes.txt"

THRESHOLD = 0.5

def load_classes():
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def predict(text, tokenizer, model, classes, k=5):
    enc = tokenizer([text], truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0].numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    idx = np.argsort(-probs)[:k]
    top_preds = [(classes[i], float(probs[i])) for i in idx]

    above = [(classes[i], float(probs[i])) for i in range(len(classes)) if probs[i] >= THRESHOLD]
    above.sort(key=lambda x: -x[1])

    return top_preds, above

def main():
    print("MoodMate Interactive Chat (type 'quit' to exit)")
    classes = load_classes()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        top_preds, above = predict(text, tokenizer, model, classes, k=5)

        print("\nTop predictions:")
        for lab, pr in top_preds:
            print(f"  {lab:15s} : {pr:.3f}")

        if above:
            print("\nLabels above threshold:")
            for lab, p in above:
                print(f"  {lab:15s} : {p:.3f}")
        else:
            print("\n(no labels above threshold)")

if __name__ == "__main__":
    main()