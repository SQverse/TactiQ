import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path(
    "data/processed/possession_with_observation_age.csv"
)

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

age_columns = [
    f"{feature}_age_frames"
    for feature in features
]

ages = df[age_columns].to_numpy()

fresh_count = (ages == 0).sum(axis=1)
stale_count = (ages > 0).sum(axis=1)
no_history_count = np.isnan(ages).sum(axis=1)

print("=" * 70)
print("OBSERVATION FRESHNESS ANALYSIS")
print("=" * 70)

print("\nFeature freshness per frame:")

for count in range(8):
    total = int((fresh_count == count).sum())

    print(
        f"{count} fresh features: "
        f"{total:8d} frames "
        f"({total / len(df) * 100:.2f}%)"
    )

print("\nHistory availability:")

for count in range(8):
    total = int((no_history_count == count).sum())

    print(
        f"{count} features without history: "
        f"{total:8d} frames"
    )

print("\nCombined categories:")

categories = {
    "All 7 fresh": (fresh_count == 7),
    "At least 5 fresh": (fresh_count >= 5),
    "At least 3 fresh": (fresh_count >= 3),
    "At least 1 fresh": (fresh_count >= 1),
    "No fresh features": (fresh_count == 0),
}

for label, mask in categories.items():
    total = int(mask.sum())

    print(
        f"{label:22s}: {total:8d} "
        f"({total / len(df) * 100:.2f}%)"
    )

print("\nAnalysis complete.")