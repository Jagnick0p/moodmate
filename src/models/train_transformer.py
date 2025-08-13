import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Use Path objects for paths
IN  = Path("/processed/transformer")
OUT = Path("/processed/transformer/DistilBERT")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
THRESHOLD = 0.5
EPOCHS = 3
BATCH_TRAIN = 16
BATCH_EVAL = 32
MAX_LEN = 128

class TensorDatasetMLC(Dataset):
    def __init__(self, pt_path: Path):
        if not Path(pt_path).exists():
            raise FileNotFoundError(f"Missing tensor file: {pt_path}")
        pack = torch.load(pt_path, map_location="cpu")
        self.input_ids = pack["input_ids"]
        self.attention_mask = pack["attention_mask"]
        self.labels = pack["labels"].float()
    def __len__(self):
        return self.input_ids.size(0)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_samples": f1_score(labels, preds, average="samples", zero_division=0)
    }

def main():
    import transformers, sys, subprocess
    print("python exe:", sys.executable)
    print("transformers ver:", transformers.__version__)
    print("cuda available:", torch.cuda.is_available())

    # Quick sanity check of files
    print("Listing /content/data:")
    subprocess.run(["bash", "-lc", "ls -lah /content/data || true"])

    # Load datasets (use Path objects)
    train_ds = TensorDatasetMLC(IN / "train.pt")
    val_ds   = TensorDatasetMLC(IN / "val.pt")
    test_ds  = TensorDatasetMLC(IN / "test.pt")
    num_labels = train_ds.labels.size(1)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification"
    )

    # Version-agnostic TrainingArguments
    args = TrainingArguments(
        output_dir=str(OUT / "trainer_runs"),
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        learning_rate=5e-5,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),  # mixed precision if GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()

    print("Evaluating...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    # Save metrics
    (OUT / "model").mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    # Save model + tokenizer
    print("Saving model...")
    trainer.save_model(str(OUT / "model"))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.save_pretrained(str(OUT / "model"))

    print("Done. Artifacts in:", OUT)

if __name__ == "__main__":
    main()