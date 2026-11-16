import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
df = pd.read_csv(INPUT)

signals = {
    "gap": "nearest_distance_gap",
    "gap_change": "nearest_distance_gap_change",
    "team0_change": "team0_nearest_distance_change",
    "team1_change": "team1_nearest_distance_change",
    "ball_movement": "ball_displacement",
}

print("=" * 65)
print("MULTI-SIGNAL AVAILABILITY ANALYSIS")
print("=" * 65)

print("\n[1] INDIVIDUAL AVAILABILITY")

for name, col in signals.items():
    valid = int(df[col].notna().sum())
    missing = len(df) - valid

    print(f"\n{name}")
    print("Valid:", valid)
    print("Missing:", missing)
    print("Availability:", round(valid / len(df) * 100, 2), "%")

print("\n[2] COMBINED AVAILABILITY")

for group_name, columns in {
    "Gap + Gap Change": [
        signals["gap"],
        signals["gap_change"],
    ],
    "Gap + Both Distance Changes": [
        signals["gap"],
        signals["team0_change"],
        signals["team1_change"],
    ],
    "Gap + Ball Movement": [
        signals["gap"],
        signals["ball_movement"],
    ],
    "All Five Signals": list(signals.values()),
}.items():

    valid = int(df[columns].notna().all(axis=1).sum())

    print(f"\n{group_name}")
    print("Complete frames:", valid)
    print("Availability:", round(valid / len(df) * 100, 2), "%")

print("\n[3] SIGNAL COVERAGE BY FRAME")

coverage = df[list(signals.values())].notna().sum(axis=1)

print(coverage.value_counts().sort_index().to_string())

print("\n[4] SIGNAL AVAILABILITY BY 10-MINUTE INTERVAL")

interval = (df["frame"] // (FPS := 25 * 60 * 10)).astype(int)

for name, col in signals.items():
    print(f"\n{name}")

    summary = (
        df.assign(interval=interval)
        .groupby("interval")[col]
        .apply(lambda x: round(x.notna().mean() * 100, 2))
    )

    print(summary.to_string())

print("\nAnalysis complete.")
print("=" * 65)