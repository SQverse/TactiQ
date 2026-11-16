import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
OUTPUT = Path("data/processed/possession_with_observation_age.csv")

FPS = 25

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
frames = df["frame"].to_numpy()

for feature in features:
    valid = df[feature].notna().to_numpy()

    last_valid = np.maximum.accumulate(
        np.where(valid, frames, -1)
    )

    age_frames = np.full(len(df), np.nan)

    has_history = last_valid >= 0
    age_frames[has_history] = (
        frames[has_history] - last_valid[has_history]
    )

    df[f"{feature}_age_frames"] = age_frames
    df[f"{feature}_age_seconds"] = age_frames / FPS

df.to_csv(OUTPUT, index=False)

print("=" * 65)
print("OBSERVATION AGE FEATURES - CORRECTED")
print("=" * 65)

for feature in features:
    age = df[f"{feature}_age_seconds"]

    print(f"\n{feature}")
    print("No previous observation:", int(age.isna().sum()))
    print("Has previous observation:", int(age.notna().sum()))

print("\nSaved:", OUTPUT)