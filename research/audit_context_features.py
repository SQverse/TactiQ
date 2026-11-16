
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_context_features.csv")

df = pd.read_csv(INPUT)

windows = {
    "1s": 25,
    "3s": 75,
}

features = [
    "ball_displacement",
    "team0_nearest_distance",
    "team1_nearest_distance",
    "nearest_distance_gap",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
]

print("=" * 60)
print("AUDIT CONTEXT FEATURE DATASET")
print("=" * 60)

# 1. Struktur
print("\n[1] STRUKTUR")
print("Jumlah baris:", len(df))
print("Jumlah kolom:", len(df.columns))
print("Frame berurutan:", df["frame"].is_monotonic_increasing)
print("Duplikat frame:", df["frame"].duplicated().sum())

# 2. Periksa seluruh fitur rolling
print("\n[2] VALIDASI ROLLING FEATURES")

errors = []
checked = 0

for feature in features:
    for label, window in windows.items():
        for stat in ["mean", "std"]:
            col = f"{feature}_{stat}_{label}"

            if col not in df.columns:
                errors.append(f"Kolom hilang: {col}")
                continue

            if stat == "mean":
                expected = df[feature].rolling(
                    window=window,
                    min_periods=1
                ).mean()
            else:
                expected = df[feature].rolling(
                    window=window,
                    min_periods=2
                ).std()

            actual = df[col]

            matches = np.isclose(
                actual.to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-9,
                atol=1e-12
            )

            mismatch = int((~matches).sum())
            checked += 1

            if mismatch:
                errors.append(
                    f"{col}: {mismatch} nilai tidak cocok"
                )

            print(
                f"{col}: {'OK' if mismatch == 0 else 'ERROR'}"
            )

# 3. Validasi rolling track switches
print("\n[3] VALIDASI TRACK SWITCHES")

for team in ["team0", "team1"]:
    source = f"{team}_nearest_track_changed"

    for label, window in windows.items():
        col = f"{team}_track_switches_{label}"

        if col not in df.columns:
            errors.append(f"Kolom hilang: {col}")
            continue

        expected = df[source].rolling(
            window=window,
            min_periods=1
        ).sum()

        actual = df[col]

        matches = np.isclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-9,
            atol=1e-12
        )

        mismatch = int((~matches).sum())

        if mismatch:
            errors.append(
                f"{col}: {mismatch} nilai tidak cocok"
            )

        print(
            f"{col}: {'OK' if mismatch == 0 else 'ERROR'}"
        )

# 4. Missing values
print("\n[4] MISSING VALUES")

context_cols = [
    col for col in df.columns
    if "_mean_" in col
    or "_std_" in col
    or "_track_switches_" in col
]

for col in context_cols:
    print(f"{col}: {df[col].isna().sum():,}")

# 5. Rentang track switches
print("\n[5] RENTANG TRACK SWITCHES")

for col in [
    "team0_track_switches_1s",
    "team0_track_switches_3s",
    "team1_track_switches_1s",
    "team1_track_switches_3s",
]:
    values = df[col].dropna()
    print(
        f"{col}: min={values.min()}, max={values.max()}"
    )

# 6. Periksa nilai standar deviasi
print("\n[6] STANDAR DEVIASI")

std_cols = [col for col in context_cols if "_std_" in col]

for col in std_cols:
    values = df[col].dropna()
    negative = int((values < 0).sum())
    nonfinite = int((~np.isfinite(values)).sum())

    print(
        f"{col}: negatif={negative}, nonfinite={nonfinite}"
    )

    if negative or nonfinite:
        errors.append(f"Nilai tidak valid pada {col}")

# 7. Ringkasan
print("\n[7] HASIL AKHIR")
print("Fitur rolling diperiksa:", checked)
print("Total fitur konteks:", len(context_cols))

if errors:
    print("DITEMUKAN MASALAH:")
    for error in errors:
        print("-", error)
else:
    print("Semua pemeriksaan konsistensi berhasil.")

print("\nAudit selesai.")
print("=" * 60)
