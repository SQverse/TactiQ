import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
FPS = 25

df = pd.read_csv(INPUT)

# Status ketersediaan seluruh lima sinyal.
columns = [
    "nearest_distance_gap",
    "nearest_distance_gap_change",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "ball_displacement",
]

available = df[columns].notna().all(axis=1)

# Kelompokkan rangkaian status yang berurutan.
group = available.ne(available.shift()).cumsum()

runs = (
    pd.DataFrame({
        "frame": df["frame"],
        "available": available,
        "group": group,
    })
    .groupby("group")
    .agg(
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        available=("available", "first"),
        length=("frame", "size"),
    )
)

runs["duration_seconds"] = runs["length"] / FPS

print("=" * 65)
print("TRACKING AVAILABILITY RUN ANALYSIS")
print("=" * 65)

for status, name in [(True, "AVAILABLE"), (False, "MISSING")]:
    subset = runs[runs["available"] == status]

    print(f"\n[{name}]")
    print("Number of runs:", len(subset))
    print("Total frames:", int(subset["length"].sum()))

    print("\nDuration statistics:")
    print(
        subset["duration_seconds"]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
        .round(2)
    )

    print("\nDuration thresholds:")
    for seconds in [1, 5, 10, 30, 60, 120, 300]:
        count = int((subset["duration_seconds"] >= seconds).sum())
        print(f">= {seconds:3d} seconds: {count}")

print("\n[LONGEST MISSING PERIODS]")

missing = runs[runs["available"] == False].nlargest(
    10, "length"
)

for row in missing.itertuples():
    print(
        f"Frame {row.start_frame}-{row.end_frame} | "
        f"{row.duration_seconds:.2f} seconds"
    )

print("\n[LONGEST AVAILABLE PERIODS]")

present = runs[runs["available"] == True].nlargest(
    10, "length"
)

for row in present.itertuples():
    print(
        f"Frame {row.start_frame}-{row.end_frame} | "
        f"{row.duration_seconds:.2f} seconds"
    )

print("\nAnalysis complete.")
print("=" * 65)