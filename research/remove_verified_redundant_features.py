from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/possession_with_observation_age.csv")
OUTPUT = Path("data/processed/possession_features_clean.csv")

REDUNDANT_COLUMNS = [
    "team0_nearest_track_changed_valid_count_1s",
    "team0_nearest_track_changed_valid_count_3s",
    "team1_nearest_track_changed_valid_count_1s",
    "team1_nearest_track_changed_valid_count_3s",
]

df = pd.read_csv(INPUT)

missing = [col for col in REDUNDANT_COLUMNS if col not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

clean_df = df.drop(columns=REDUNDANT_COLUMNS)

clean_df.to_csv(OUTPUT, index=False)

print("=" * 65)
print("CLEAN FEATURE DATASET")
print("=" * 65)
print(f"Original rows   : {len(df)}")
print(f"Original columns: {len(df.columns)}")
print(f"Clean rows      : {len(clean_df)}")
print(f"Clean columns   : {len(clean_df.columns)}")
print(f"Removed columns : {len(REDUNDANT_COLUMNS)}")
print(f"Output          : {OUTPUT}")
print("\nDataset berhasil disimpan.")