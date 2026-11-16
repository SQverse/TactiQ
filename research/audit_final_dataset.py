
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")

df = pd.read_csv(INPUT)

features = [
    "ball_displacement",
    "team0_nearest_distance",
    "team1_nearest_distance",
    "nearest_distance_gap",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
    "team0_nearest_track_changed",
    "team1_nearest_track_changed",
]

windows = {
    "1s": 25,
    "3s": 75,
}

errors = []

print("=" * 60)
print("FINAL DATASET AUDIT")
print("=" * 60)

print("\n[1] STRUKTUR")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Frame ordered:", df["frame"].is_monotonic_increasing)
print("Duplicate frames:", df["frame"].duplicated().sum())

if len(df) != 199319:
    errors.append("Jumlah frame tidak sesuai.")

if len(df.columns) != 71:
    errors.append("Jumlah kolom tidak sesuai.")

# Periksa semua coverage columns
print("\n[2] COVERAGE VALIDATION")

for feature in features:
    for label, window in windows.items():
        source = df[feature]
        col = f"{feature}_valid_count_{label}"

        expected = source.rolling(
            window=window,
            min_periods=1
        ).count()

        if col not in df.columns:
            errors.append(f"Kolom hilang: {col}")
            continue

        actual = df[col]

        mismatch = (
            (actual - expected).abs() > 1e-9
        ) & ~(actual.isna() & expected.isna())

        count = int(mismatch.sum())

        print(f"{col}: {'OK' if count == 0 else 'ERROR'}")

        if count:
            errors.append(
                f"{col}: {count} nilai tidak cocok."
            )

        if (actual.dropna() > window).any():
            errors.append(f"{col}: melebihi ukuran window.")

        if (actual.dropna() < 0).any():
            errors.append(f"{col}: memiliki nilai negatif.")

# Periksa kelengkapan kolom
print("\n[3] MISSING VALUES")
print("Total missing seluruh dataset:", 
      int(df.isna().sum().sum()))

print("\n[4] HASIL AKHIR")

if errors:
    print("DITEMUKAN MASALAH:")
    for error in errors:
        print("-", error)
else:
    print("SEMUA PEMERIKSAAN BERHASIL.")
    print("Dataset siap digunakan untuk tahap inference.")

print("=" * 60)
