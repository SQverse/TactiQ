
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_temporal_features.csv")

df = pd.read_csv(INPUT)

temporal_cols = [
    "team0_player_count_change",
    "team1_player_count_change",
    "ball_displacement",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
    "team0_nearest_track_changed",
    "team1_nearest_track_changed",
]

print("=" * 60)
print("AUDIT TEMPORAL FEATURE DATASET")
print("=" * 60)

# 1. Struktur dataset
print("\n[1] STRUKTUR")
print("Jumlah baris:", len(df))
print("Jumlah kolom:", len(df.columns))
print("Frame berurutan:", df["frame"].is_monotonic_increasing)
print("Duplikat frame:", df["frame"].duplicated().sum())
print("Kolom temporal tersedia:",
      all(col in df.columns for col in temporal_cols))

# 2. Missing values
print("\n[2] MISSING VALUES")
for col in temporal_cols:
    print(f"{col}: {df[col].isna().sum():,}")

# 3. Perubahan jumlah pemain
print("\n[3] PERUBAHAN JUMLAH PEMAIN")
for col in [
    "team0_player_count_change",
    "team1_player_count_change"
]:
    values = df[col].dropna()
    print(f"\n{col}")
    if len(values):
        print("Minimum:", values.min())
        print("Maksimum:", values.max())
        print("Rata-rata:", round(values.mean(), 6))
        print("Nilai di luar batas perubahan:", 
              ((values.abs() > 46)).sum())

# 4. Pergerakan bola
print("\n[4] BALL DISPLACEMENT")
values = df["ball_displacement"].dropna()

print("Jumlah valid:", len(values))
if len(values):
    print("Minimum:", values.min())
    print("Maksimum:", values.max())
    print("Rata-rata:", values.mean())
    print("Median:", values.median())
    print("Nilai negatif:", (values < 0).sum())
    print("Nilai non-finite:", 
          (~np.isfinite(values)).sum())

# 5. Perubahan jarak dan gap
print("\n[5] DISTANCE CHANGES")

for col in [
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change"
]:
    values = df[col].dropna()
    print(f"\n{col}")
    print("Jumlah valid:", len(values))
    if len(values):
        print("Minimum:", values.min())
        print("Maksimum:", values.max())
        print("Rata-rata:", values.mean())
        print("Median:", values.median())
        print("Non-finite:", 
              (~np.isfinite(values)).sum())

# 6. Pergantian nearest track
print("\n[6] NEAREST TRACK CHANGES")

for col in [
    "team0_nearest_track_changed",
    "team1_nearest_track_changed"
]:
    values = df[col].dropna()
    print(f"\n{col}")
    print("Jumlah valid:", len(values))
    print("Nilai unik:", sorted(values.unique().tolist()))
    print("Track berubah:", (values == 1).sum())
    print("Track tetap:", (values == 0).sum())

# 7. Konsistensi ball displacement
print("\n[7] KONSISTENSI BALL")

ball_valid = df["ball_valid"].fillna(0).astype(bool)
expected_ball_pairs = (
    ball_valid & ball_valid.shift(1, fill_value=False)
)

actual_ball_pairs = df["ball_displacement"].notna()

print("Pasangan ball valid berdasarkan data:",
      expected_ball_pairs.sum())
print("Ball displacement terisi:",
      actual_ball_pairs.sum())
print("Ketidaksesuaian:",
      (expected_ball_pairs != actual_ball_pairs).sum())

# 8. Kesimpulan otomatis
print("\n[8] PEMERIKSAAN AKHIR")

issues = []

if len(df) != 199319:
    issues.append("Jumlah baris tidak sesuai.")

if not df["frame"].is_monotonic_increasing:
    issues.append("Urutan frame tidak benar.")

if df["frame"].duplicated().sum() > 0:
    issues.append("Ada frame duplikat.")

if (expected_ball_pairs != actual_ball_pairs).any():
    issues.append("Ball displacement tidak konsisten.")

for col in temporal_cols:
    values = df[col].dropna()
    if len(values) and not np.isfinite(values).all():
        issues.append(f"Ada nilai non-finite di {col}.")

if issues:
    print("DITEMUKAN HAL YANG PERLU DIPERIKSA:")
    for issue in issues:
        print("-", issue)
else:
    print("Struktur dan pemeriksaan dasar temporal berhasil.")

print("\nAudit selesai.")
print("=" * 60)
