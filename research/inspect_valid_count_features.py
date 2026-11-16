
import pandas as pd

df = pd.read_csv(
    "data/processed/possession_features_clean.csv",
    nrows=5
)

print("FEATURES CONTAINING VALID_COUNT:")
for col in df.columns:
    if "valid_count" in col.lower():
        print(col)
