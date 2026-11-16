import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")

df = pd.read_csv(INPUT)

# Kelompokkan jumlah pemain yang terdeteksi.
bins = [0, 5, 10, 15, 20, 25, 50]
labels = [
    "1-5",
    "6-10",
    "11-15",
    "16-20",
    "21-25",
    "26+"
]

# Analisis menggunakan jumlah pemain dari kedua tim.
df["total_players"] = (
    df["team0_player_count"]
    + df["team1_player_count"]
)

df["player_group"] = pd.cut(
    df["total_players"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print("=" * 70)
print("MOVEMENT BY PLAYER COUNT")
print("=" * 70)

for team in [0, 1]:

    feature = (
        f"team{team}_player_count"
    )

    print(f"\nTEAM {team}")

    for group in labels:

        subset = df[df["player_group"] == group]

        print(f"\nPlayer group: {group}")
        print("Frames:", len(subset))
        print(
            "Ball available:",
            int(subset["ball_valid"].sum())
        )
        print(
            "Ball unavailable:",
            int((~subset["ball_valid"].astype(bool)).sum())
        )
        print(
            "Average player count:",
            round(subset[feature].mean(), 3)
        )

print("\nAnalysis complete.")