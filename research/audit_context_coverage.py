
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_features.csv")

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

print("=" * 70)
print("CONTEXT WINDOW COVERAGE AUDIT")
print("=" * 70)

for feature in features:
    print(f"\nFEATURE: {feature}")

    for label, window in windows.items():
        counts = df[feature].rolling(
            window=window,
            min_periods=1
        ).count()

        total = len(counts)

        print(f"\n  Window {label} ({window} frame)")
        print("  Rata-rata observasi valid:", round(counts.mean(), 2))
        print("  Minimum:", int(counts.min()))
        print("  Maksimum:", int(counts.max()))

        for minimum in [1, 5, 12, 25]:
            if minimum > window:
                continue

            sufficient = int((counts >= minimum).sum())
            percentage = sufficient / total * 100

            print(
                f"  Jendela dengan >= {minimum} observasi: "
                f"{sufficient:,} ({percentage:.2f}%)"
            )

print("\n" + "=" * 70)
print("AUDIT COVERAGE SELESAI")
print("=" * 70)
