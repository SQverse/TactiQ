import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
FPS = 25

df = pd.read_csv(INPUT)

gap = df["nearest_distance_gap"]

# Abaikan frame ketika gap tidak tersedia.
valid = gap.dropna()

# -1 = Team 0 lebih dekat
# +1 = Team 1 lebih dekat
#  0 = jarak sama
signal = pd.Series(np.sign(valid.to_numpy()), index=valid.index)

# Buat ID rangkaian berdasarkan perubahan tanda.
group_id = signal.ne(signal.shift()).cumsum()

segments = (
    pd.DataFrame({
        "frame": signal.index,
        "signal": signal.to_numpy(),
        "group": group_id.to_numpy()
    })
    .groupby("group")
    .agg(
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        signal=("signal", "first"),
        length=("frame", "size")
    )
)

segments["duration_seconds"] = segments["length"] / FPS

print("=" * 65)
print("GAP SIGNAL STABILITY ANALYSIS")
print("=" * 65)

print("\n[1] BASIC STATISTICS")
print("Valid observations:", len(valid))
print("Total segments:", len(segments))
print("Team 0 closer segments:", int((segments["signal"] < 0).sum()))
print("Team 1 closer segments:", int((segments["signal"] > 0).sum()))

print("\n[2] SEGMENT DURATION")
print(
    segments["duration_seconds"]
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
    .round(3)
)

print("\n[3] DURATION DISTRIBUTION")

for seconds in [1, 2, 3, 5, 10, 20]:
    count = int((segments["duration_seconds"] >= seconds).sum())
    print(f"< {seconds:4.1f} seconds: {count}")

print("\n[4] SHORT SEGMENTS")

for seconds in [0.5, 1, 2, 3]:
    count = int((segments["duration_seconds"] < seconds).sum())
    print(f"< {seconds:4.1f} seconds: {count}")

print("\n[5] TEAM DURATION")

for label, name in [(-1, "Team 0"), (1, "Team 1")]:
    subset = segments[segments["signal"] == label]

    print(f"\n{name}")
    print("Segments:", len(subset))
    print(
        "Mean duration:",
        round(subset["duration_seconds"].mean(), 3)
    )
    print(
        "Median duration:",
        round(subset["duration_seconds"].median(), 3)
    )

print("\nAnalysis complete.")
print("=" * 65)