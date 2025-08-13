import ast
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "processed"  # we use the CLEANED files
OUT = PROJECT_ROOT / "data" / "processed" / "tfidf"
OUT.mkdir(parents=True, exist_ok=True)

def load_split(name):
    df = pd.read_csv(RAW / f"goemotions_{name}_clean.csv")
    # parse list-like strings safely to Python lists
    df["label_names"] = df["label_names"].apply(lambda s: ast.literal_eval(s))
    return df

def main():
    print("Loading cleaned splits...")
    df_train = load_split("train")
    df_val   = load_split("validation")
    df_test  = load_split("test")

    # -----------------------------
    # Multi-label binarizer (fit on TRAIN ONLY)
    # -----------------------------
    print("Fitting label binarizer on train...")
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(df_train["label_names"])
    y_val   = mlb.transform(df_val["label_names"])
    y_test  = mlb.transform(df_test["label_names"])

    # -----------------------------
    # TF-IDF vectorizer (fit on TRAIN ONLY)
    # -----------------------------
    print("Fitting TF-IDF on train text_clean...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),       # unigrams + bigrams
        min_df=3,                # ignore very rare tokens
        max_df=0.9,              # ignore tokens in >90% of docs
        strip_accents="unicode",
        lowercase=False,         # we already lowercased in cleaning
        sublinear_tf=True        # log(1+tf) style
    )
    X_train = vectorizer.fit_transform(df_train["text_clean"].astype(str))
    X_val   = vectorizer.transform(df_val["text_clean"].astype(str))
    X_test  = vectorizer.transform(df_test["text_clean"].astype(str))

    # -----------------------------
    # Save artifacts and matrices
    # -----------------------------
    print("Saving TF-IDF matrices and label arrays...")
    save_npz(OUT / "X_train.npz", X_train)
    save_npz(OUT / "X_val.npz", X_val)
    save_npz(OUT / "X_test.npz", X_test)

    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "y_val.npy", y_val)
    np.save(OUT / "y_test.npy", y_test)

    joblib.dump(vectorizer, OUT / "vectorizer.pkl")
    joblib.dump(mlb, OUT / "mlb.pkl")

    print("Done. Files saved under data/processed/tfidf/")
    print("   - X_*.npz (sparse features)")
    print("   - y_*.npy (multi-hot labels)")
    print("   - vectorizer.pkl, mlb.pkl")

if __name__ == "__main__":
    main()