import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
FPS = 25

df = pd.read_csv(INPUT)

segments = []
current = None
previous_frame = None

for row in df[["frame", "nearest_distance_gap"]].itertuples(index=False):
    frame, gap = row

    # Data gap hilang: putuskan segmen.
    if pd.isna(gap):
        if current is not None:
            segments.append(current)
            current = None

        previous_frame = None
        continue

    signal = int(np.sign(gap))

    # Gap tepat nol dianggap netral dan memutus segmen.
    if signal == 0:
        if current is not None:
            segments.append(current)
            current = None

        previous_frame = frame
        continue

    # Putuskan segmen jika frame tidak berurutan.
    consecutive = (
        previous_frame is not None
        and frame == previous_frame + 1
    )

    # Mulai segmen baru jika tanda berubah atau ada jeda.
    if (
        current is None
        or not consecutive
        or current["signal"] != signal
    ):
        if current is not None:
            segments.append(current)

        current = {
            "start_frame": frame,
            "end_frame": frame,
            "signal": signal,
            "length": 1
        }
    else:
        current["end_frame"] = frame
        current["length"] += 1

    previous_frame = frame

if current is not None:
    segments.append(current)

result = pd.DataFrame(segments)

if result.empty:
    print("Tidak ditemukan segmen valid.")
    raise SystemExit

result["duration_seconds"] = result["length"] / FPS

print("=" * 65)
print("GAP SIGNAL STABILITY ANALYSIS V2")
print("=" * 65)

print("\n[1] BASIC STATISTICS")
print("Total frames:", len(df))
print("Valid gap:", int(df["nearest_distance_gap"].notna().sum()))
print("Missing gap:", int(df["nearest_distance_gap"].isna().sum()))
print("Total segments:", len(result))

print("Team 0 closer:", int((result["signal"] == -1).sum()))
print("Team 1 closer:", int((result["signal"] == 1).sum()))

print("\n[2] SEGMENT DURATION")
print(
    result["duration_seconds"]
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
    .round(3)
)

print("\n[3] DURATION DISTRIBUTION")

for seconds in [1, 2, 3, 5, 10, 20]:
    count = int((result["duration_seconds"] >= seconds).sum())
    print(f">= {seconds:2d} seconds: {count}")

print("\n[4] SHORT SEGMENTS")

for seconds in [0.5, 1, 2, 3]:
    count = int((result["duration_seconds"] < seconds).sum())
    print(f"< {seconds:4.1f} seconds: {count}")

print("\n[5] TEAM DURATION")

for label, name in [(-1, "Team 0"), (1, "Team 1")]:
    subset = result[result["signal"] == label]

    print(f"\n{name}")
    print("Segments:", len(subset))
    print("Mean duration:",
          round(subset["duration_seconds"].mean(), 3))
    print("Median duration:",
          round(subset["duration_seconds"].median(), 3))

print("\n[6] SEGMENT LENGTH COUNTS")
print("1 frame:", int((result["length"] == 1).sum()))
print("2-5 frames:", int(result["length"].between(2, 5).sum()))
print("6-25 frames:", int(result["length"].between(6, 25).sum()))
print("More than 25 frames:", int((result["length"] > 25).sum()))

print("\nAnalysis complete.")
print("=" * 65)