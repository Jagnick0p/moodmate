from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "tfidf"
MODEL_PATH = BASE_DIR / "baseline_model" / "ovr_logreg.joblib"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
MLB_PATH = BASE_DIR / "mlb.pkl"

THRESHOLD = 0.5  # probability cutoff

def load_model():
    print("Loading TF-IDF model...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    return vectorizer, model, mlb

def predict_tfidf(text, vectorizer, model, mlb, k=5):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)  # list of arrays for OvR
    probs = np.array([p[0] if p.ndim > 1 else p for p in probs]).flatten()

    idx = np.argsort(-probs)[:k]
    top_preds = [(mlb.classes_[i], float(probs[i])) for i in idx]

    above = [(mlb.classes_[i], float(probs[i])) for i in range(len(mlb.classes_)) if probs[i] >= THRESHOLD]
    above.sort(key=lambda x: -x[1])

    return top_preds, above

def main():
    print("MoodMate TF-IDF Interactive Chat (type 'quit' to exit)")
    vectorizer, model, mlb = load_model()

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        top_preds, above = predict_tfidf(text, vectorizer, model, mlb)

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