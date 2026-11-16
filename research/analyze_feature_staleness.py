import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
FPS = 25

df = pd.read_csv(INPUT)

features = [
    "ball_displacement",
    "team0_nearest_distance",
    "team1_nearest_distance",
    "nearest_distance_gap",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
]

print("=" * 70)
print("FEATURE STALENESS ANALYSIS")
print("=" * 70)

for feature in features:

    values = df[feature].notna().to_numpy()
    frames = df["frame"].to_numpy()

    # Frame valid terakhir, termasuk frame saat ini jika valid.
    last_valid = np.maximum.accumulate(
        np.where(values, frames, -1)
    )

    # Usia observasi terakhir.
    age = frames - last_valid

    # Hanya periksa frame yang fitur saat ini hilang.
    missing = ~values
    missing_age = age[missing]

    # -1 berarti belum pernah ada observasi valid sebelumnya.
    no_history = missing_age < 0
    known_age = missing_age[~no_history]

    print(f"\n{'=' * 55}")
    print("FEATURE:", feature)
    print("Missing frames:", int(missing.sum()))
    print("Missing without previous history:", int(no_history.sum()))
    print("Missing with previous history:", len(known_age))

    if len(known_age) == 0:
        continue

    print("\nAge statistics (seconds):")
    print(
        pd.Series(known_age / FPS)
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
        .round(2)
    )

    print("\nAge distribution:")

    thresholds = [
        ("<= 1 second", 25),
        ("1-3 seconds", 75),
        ("3-5 seconds", 125),
        ("5-10 seconds", 250),
        ("> 10 seconds", float("inf")),
    ]

    previous = 0

    for label, upper in thresholds:
        if upper == float("inf"):
            count = int((known_age > previous).sum())
        else:
            count = int(
                ((known_age > previous) & (known_age <= upper)).sum()
            )

        print(f"{label:18s}: {count:7d}")
        previous = upper

print("\nAnalysis complete.")
print("=" * 70)