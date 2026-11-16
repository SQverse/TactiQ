
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
df = pd.read_csv(INPUT)

print("=" * 65)
print("SIGNAL CONSISTENCY ANALYSIS")
print("=" * 65)

# Gunakan hanya observasi yang tersedia
valid = df[
    [
        "nearest_distance_gap",
        "nearest_distance_gap_change",
        "team0_nearest_distance_change",
        "team1_nearest_distance_change",
        "ball_displacement",
    ]
].dropna()

print("\nJumlah observasi lengkap:", len(valid))

# 1. Korelasi antarfitur
print("\n[1] CORRELATION MATRIX")

cols = [
    "nearest_distance_gap",
    "nearest_distance_gap_change",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "ball_displacement",
]

print(valid[cols].corr().round(4).to_string())

# 2. Arah perubahan jarak pemain
print("\n[2] PLAYER DISTANCE DIRECTION")

for team in ["team0", "team1"]:
    col = f"{team}_nearest_distance_change"
    values = valid[col]

    print(f"\n{team}")
    print("Mendekat:", int((values < 0).sum()))
    print("Menjauh:", int((values > 0).sum()))
    print("Tetap:", int((values == 0).sum()))

# 3. Distribusi ball displacement
print("\n[3] BALL MOVEMENT DISTRIBUTION")

print(
    valid["ball_displacement"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
    ).round(6)
)

# 4. Konsistensi arah gap dan perubahan jarak
print("\n[4] GAP AND DISTANCE CHANGE RELATION")

for condition, subset in [
    ("Gap negative", valid[valid["nearest_distance_gap"] < 0]),
    ("Gap positive", valid[valid["nearest_distance_gap"] > 0]),
]:
    print(f"\n{condition}: {len(subset)} observations")

    print(
        "Mean Team 0 distance change:",
        round(subset["team0_nearest_distance_change"].mean(), 6)
    )

    print(
        "Mean Team 1 distance change:",
        round(subset["team1_nearest_distance_change"].mean(), 6)
    )

    print(
        "Mean gap change:",
        round(subset["nearest_distance_gap_change"].mean(), 6)
    )

print("\nAnalysis complete.")
print("=" * 65)
