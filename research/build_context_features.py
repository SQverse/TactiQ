
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_temporal_features.csv")
OUTPUT = Path("data/processed/possession_context_features.csv")

df = pd.read_csv(INPUT)

# Pastikan urutan frame benar
df = df.sort_values("frame").reset_index(drop=True)

# Fitur yang diringkas berdasarkan jendela waktu
features = [
    "ball_displacement",
    "team0_nearest_distance",
    "team1_nearest_distance",
    "nearest_distance_gap",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
]

# Jendela 1 detik dan 3 detik pada 25 FPS
windows = {
    "1s": 25,
    "3s": 75,
}

for feature in features:
    if feature not in df.columns:
        continue

    for label, window in windows.items():
        # Rata-rata hanya dari frame saat ini dan sebelumnya
        df[f"{feature}_mean_{label}"] = (
            df[feature]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        # Simpangan baku untuk melihat kestabilan perubahan
        df[f"{feature}_std_{label}"] = (
            df[feature]
            .rolling(window=window, min_periods=2)
            .std()
        )

# Jumlah pergantian nearest track dalam jendela waktu
for team in ["team0", "team1"]:
    col = f"{team}_nearest_track_changed"

    for label, window in windows.items():
        df[f"{team}_track_switches_{label}"] = (
            df[col]
            .rolling(window=window, min_periods=1)
            .sum()
        )

# Simpan hasil
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("CONTEXT FEATURE DATASET BERHASIL DIBUAT")
print("=" * 60)
print("Total frame:", len(df))
print("Jumlah kolom:", len(df.columns))
print("Fitur tambahan:", len(df.columns) - 21)
print("Window: 1 detik dan 3 detik")
print("Arah window: masa lalu + frame saat ini")
print("Output:", OUTPUT)
print("=" * 60)
