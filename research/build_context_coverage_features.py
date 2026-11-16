
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_features.csv")
OUTPUT = Path("data/processed/possession_context_final.csv")

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

added = 0

for feature in features:
    for label, window in windows.items():
        col = f"{feature}_valid_count_{label}"

        df[col] = (
            df[feature]
            .rolling(window=window, min_periods=1)
            .count()
        )

        added += 1

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("FINAL CONTEXT DATASET BERHASIL DIBUAT")
print("=" * 60)
print("Total frame:", len(df))
print("Kolom awal:", 53)
print("Fitur coverage tambahan:", added)
print("Total kolom:", len(df.columns))
print("Output:", OUTPUT)
print("=" * 60)
