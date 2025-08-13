from pathlib import Path
import json
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.utils import class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN = PROJECT_ROOT / "data" / "processed" / "tfidf"
OUT = PROJECT_ROOT / "data" / "processed" / "tfidf" / "baseline_model"
OUT.mkdir(parents=True, exist_ok=True)

def load_split(split):
    X = load_npz(IN / f"X_{split}.npz")
    y = np.load(IN / f"y_{split}.npy")
    return X, y

def main():
    print("Loading features/labels...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("y_val".replace("y_", ""))  # just to be explicit
    X_val, y_val     = load_split("val")
    X_test,  y_test  = load_split("test")

    # (Optional) class_weight per class to handle imbalance
    # scikit-learn's LogisticRegression supports class_weight for binary problems;
    # OvR trains one binary model per class, so we pass 'balanced'.
    base_lr = LogisticRegression(
        solver="liblinear",   # good for smaller problems / sparse data
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None
    )
    clf = OneVsRestClassifier(base_lr, n_jobs=-1)

    print("Training One-vs-Rest Logistic Regression...")
    clf.fit(X_train, y_train)

    print("Evaluating (val/test)...")
    # Decision function → probabilities via sigmoid-like mapping not directly available for liblinear;
    # predict_proba is available; we’ll threshold at 0.5.
    y_val_prob = clf.predict_proba(X_val)
    y_test_prob = clf.predict_proba(X_test)

    y_val_pred = (y_val_prob >= 0.5).astype(int)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    metrics = {
        "val": {
            "f1_micro": float(f1_score(y_val, y_val_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_val, y_val_pred, average="macro", zero_division=0)),
            "f1_samples": float(f1_score(y_val, y_val_pred, average="samples", zero_division=0)),
            "roc_auc_macro": float(roc_auc_score(y_val, y_val_prob, average="macro"))
        },
        "test": {
            "f1_micro": float(f1_score(y_test, y_test_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
            "f1_samples": float(f1_score(y_test, y_test_pred, average="samples", zero_division=0)),
            "roc_auc_macro": float(roc_auc_score(y_test, y_test_prob, average="macro"))
        }
    }

    print(json.dumps(metrics, indent=2))

    # Save artifacts
    print("Saving model & metrics...")
    joblib.dump(clf, OUT / "ovr_logreg.joblib")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Model in: {OUT}")

if __name__ == "__main__":
    main()
