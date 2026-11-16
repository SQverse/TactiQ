import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
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
print("TEMPORAL FEATURE RECOVERY ANALYSIS")
print("=" * 70)

for feature in features:

    print(f"\n{'=' * 60}")
    print("FEATURE:", feature)

    current = df[feature].notna()

    for window in ["1s", "3s"]:

        coverage_col = f"{feature}_valid_count_{window}"

        coverage = df[coverage_col]

        print(f"\nWindow: {window}")

        for threshold in [1, 5, 12, 25] if window == "1s" else [1, 5, 25, 50, 75]:

            enough = coverage >= threshold

            recovered = (~current) & enough

            print(
                f"Coverage >= {threshold:2d}: "
                f"{int(enough.sum()):7d} frames | "
                f"Current missing but history available: "
                f"{int(recovered.sum()):7d}"
            )

print("\nAnalysis complete.")
print("=" * 70)