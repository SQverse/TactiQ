
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")

df = pd.read_csv(INPUT)

print("=" * 65)
print("POSSESSION EVIDENCE ANALYZER")
print("=" * 65)

# 1. Ketersediaan bukti spasial
ball_valid = df["ball_valid"].fillna(0).astype(bool)

t0_valid = df["team0_nearest_distance"].notna()
t1_valid = df["team1_nearest_distance"].notna()

both_valid = ball_valid & t0_valid & t1_valid

print("\n[1] SPATIAL EVIDENCE")
print("Ball detected:", int(ball_valid.sum()))
print("Team 0 nearest available:", int(t0_valid.sum()))
print("Team 1 nearest available:", int(t1_valid.sum()))
print("Both teams comparable:", int(both_valid.sum()))

# 2. Perbandingan jarak
gap = df.loc[both_valid, "nearest_distance_gap"]

print("\n[2] NEAREST DISTANCE GAP")
print("Team 0 closer:", int((gap < 0).sum()))
print("Team 1 closer:", int((gap > 0).sum()))
print("Equal:", int((gap == 0).sum()))

print("Mean:", gap.mean())
print("Median:", gap.median())
print("Standard deviation:", gap.std())

# 3. Ketersediaan temporal
print("\n[3] TEMPORAL EVIDENCE")

for col in [
    "ball_displacement",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
]:
    valid = df[col].notna()
    print(f"{col}: {int(valid.sum())}")

# 4. Coverage dalam jendela 1 detik
print("\n[4] ONE-SECOND COVERAGE")

for col in [
    "ball_displacement_valid_count_1s",
    "team0_nearest_distance_valid_count_1s",
    "team1_nearest_distance_valid_count_1s",
    "nearest_distance_gap_valid_count_1s",
]:
    values = df[col]

    print(f"\n{col}")
    print("Mean:", round(values.mean(), 2))
    print("Median:", values.median())
    print("Zero coverage:", int((values == 0).sum()))
    print("Full coverage:", int((values == 25).sum()))

# 5. Hubungan tanda gap dengan perubahan gap
print("\n[5] GAP DIRECTION AND TEMPORAL CHANGE")

valid = (
    both_valid
    & df["nearest_distance_gap_change"].notna()
)

subset = df.loc[
    valid,
    ["nearest_distance_gap", "nearest_distance_gap_change"]
]

print("Comparable consecutive observations:", len(subset))

print(
    "Gap negative, change negative:",
    int(
        (
            (subset["nearest_distance_gap"] < 0)
            & (subset["nearest_distance_gap_change"] < 0)
        ).sum()
    )
)

print(
    "Gap negative, change positive:",
    int(
        (
            (subset["nearest_distance_gap"] < 0)
            & (subset["nearest_distance_gap_change"] > 0)
        ).sum()
    )
)

print(
    "Gap positive, change negative:",
    int(
        (
            (subset["nearest_distance_gap"] > 0)
            & (subset["nearest_distance_gap_change"] < 0)
        ).sum()
    )
)

print(
    "Gap positive, change positive:",
    int(
        (
            (subset["nearest_distance_gap"] > 0)
            & (subset["nearest_distance_gap_change"] > 0)
        ).sum()
    )
)

print("\nAnalysis complete.")
print("=" * 65)
