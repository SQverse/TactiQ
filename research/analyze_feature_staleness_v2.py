import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_with_observation_age.csv")

features = [
    "ball_displacement",
    "team0_nearest_distance",
    "team1_nearest_distance",
    "nearest_distance_gap",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
]

df = pd.read_csv(INPUT)

print("=" * 70)
print("CORRECTED FEATURE STALENESS ANALYSIS")
print("=" * 70)

for feature in features:
    age = df[f"{feature}_age_seconds"]

    no_history = age.isna()
    known_age = age.dropna()

    print(f"\nFEATURE: {feature}")
    print("No previous observation:", int(no_history.sum()))
    print("Has previous observation:", len(known_age))

    if len(known_age) == 0:
        continue

    print("\nAge statistics (seconds):")
    print(
        known_age.describe(
            percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
        ).round(2)
    )

    buckets = [
        ("<= 1 second", 0, 1),
        ("1-3 seconds", 1, 3),
        ("3-5 seconds", 3, 5),
        ("5-10 seconds", 5, 10),
        ("> 10 seconds", 10, float("inf")),
    ]

    print("\nAge distribution:")

    for label, lower, upper in buckets:
        if upper == float("inf"):
            count = int((known_age > lower).sum())
        elif lower == 0:
            count = int((known_age <= upper).sum())
        else:
            count = int(
                ((known_age > lower) & (known_age <= upper)).sum()
            )

        print(f"{label:18s}: {count:7d}")

print("\nAnalysis complete.")