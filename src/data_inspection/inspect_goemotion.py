import pandas as pd
from pathlib import Path
from collections import Counter

# 1. Path to raw data
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# 2. Load train split CSV
train_path = RAW_DATA_DIR / "goemotions_train.csv"
df_train = pd.read_csv(train_path)

print("=== First 5 rows of the train data ===")
print(df_train.head(), "\n")

print("=== Columns in the dataset ===")
print(list(df_train.columns), "\n")

# 3. Show total samples
print(f"Total samples in train split: {len(df_train)}\n")

# 4. Basic info about columns
print("=== Dataset info ===")
print(df_train.info(), "\n")

# 5. Count label frequency
# label_names column contains strings like "['joy', 'admiration']"
# We need to turn them into lists first
label_counter = Counter()
for labels_str in df_train["label_names"]:
    # Convert the string representation of list back to Python list
    labels = eval(labels_str)  # safe here because we control the CSV
    label_counter.update(labels)

# Convert counter to DataFrame for easy viewing
label_counts = pd.DataFrame(label_counter.items(), columns=["label", "count"])
label_counts = label_counts.sort_values(by="count", ascending=False)

print("=== Top 10 most common emotions ===")
print(label_counts.head(10), "\n")

# 6. Save full label distribution to CSV
label_counts.to_csv(RAW_DATA_DIR / "label_distribution_train.csv", index=False)
print(f"✅ Saved label distribution to {RAW_DATA_DIR / 'label_distribution_train.csv'}\n")

# 7. Show 5 random samples
print("=== Random 5 samples ===")
print(df_train.sample(5), "\n")
