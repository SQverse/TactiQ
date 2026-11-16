import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
FPS = 25
INTERVAL_FRAMES = FPS * 60 * 10

df = pd.read_csv(INPUT)

columns = [
    "nearest_distance_gap",
    "nearest_distance_gap_change",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "ball_displacement",
]

df["available"] = df[columns].notna().all(axis=1)
df["interval"] = df["frame"] // INTERVAL_FRAMES

print("=" * 70)
print("MISSING TRACKING BY MATCH INTERVAL")
print("=" * 70)

for interval, subset in df.groupby("interval"):

    available = subset["available"]
    missing = ~available

    # Rangkaian missing dalam interval ini.
    group = missing.ne(missing.shift()).cumsum()

    missing_runs = (
        pd.DataFrame({
            "missing": missing,
            "group": group,
        })
        .groupby("group")
        .agg(
            status=("missing", "first"),
            length=("missing", "size"),
        )
    )

    missing_runs = missing_runs[missing_runs["status"]]

    longest_missing = (
        missing_runs["length"].max()
        if not missing_runs.empty else 0
    )

    print(f"\nINTERVAL {interval}")
    print("Frames:", len(subset))
    print("Available:", int(available.sum()))
    print("Missing:", int(missing.sum()))
    print(
        "Availability:",
        round(available.mean() * 100, 2), "%"
    )
    print(
        "Missing runs:",
        len(missing_runs)
    )
    print(
        "Longest missing:",
        round(longest_missing / FPS, 2),
        "seconds"
    )

print("\nAnalysis complete.")
print("=" * 70)