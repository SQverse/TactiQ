
from pathlib import Path

import numpy as np
import pandas as pd

from starter.baseline import team_of_each_track


ROOT = Path(__file__).resolve().parent

ATD_PATH = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
META_PATH = ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"
COVERAGE_PATH = ROOT / "data/processed/player_coverage_during_outage.csv"

OUTPUT_PATH = ROOT / "data/processed/spatial_structure_quality.csv"

BUCKETS = [
    "not_outage",
    "<=1s",
    "1-3s",
    "3-5s",
    "5-10s",
    ">10s",
]


def extract_positions(player_data, team_mapping):

    positions = {0: [], 1: []}

    for track_id, entry in enumerate(player_data.split(";")):

        if track_id >= len(team_mapping):
            break

        values = entry.split(",")

        if len(values) < 2:
            continue

        try:
            x = float(values[0])
            y = float(values[1])
        except ValueError:
            continue

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        team = int(team_mapping[track_id])

        if team in positions:
            positions[team].append((x, y))

    return positions


def spatial_metrics(points, prefix):

    arr = np.asarray(points, dtype=float)

    if len(arr) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_out_of_bounds_rate": np.nan,
            f"{prefix}_median_x": np.nan,
            f"{prefix}_median_y": np.nan,
            f"{prefix}_median_radius": np.nan,
            f"{prefix}_iqr_x": np.nan,
            f"{prefix}_iqr_y": np.nan,
        }

    x = arr[:, 0]
    y = arr[:, 1]

    # Report out-of-range coordinates, but do not discard or clip them.
    in_bounds = (
        (x >= 0) & (x <= 1) &
        (y >= 0) & (y <= 1)
    )

    center_x = np.median(x)
    center_y = np.median(y)

    distances = np.sqrt(
        (x - center_x) ** 2 +
        (y - center_y) ** 2
    )

    qx = np.percentile(x, [25, 75])
    qy = np.percentile(y, [25, 75])

    return {
        f"{prefix}_count": len(arr),
        f"{prefix}_out_of_bounds_rate": 1 - in_bounds.mean(),
        f"{prefix}_median_x": center_x,
        f"{prefix}_median_y": center_y,
        f"{prefix}_median_radius": np.median(distances),
        f"{prefix}_iqr_x": qx[1] - qx[0],
        f"{prefix}_iqr_y": qy[1] - qy[0],
    }


print("=" * 75)
print("SPATIAL STRUCTURE QUALITY AUDIT")
print("=" * 75)

team_mapping = team_of_each_track(str(META_PATH))

coverage = pd.read_csv(COVERAGE_PATH)
coverage = coverage.sort_values("frame").reset_index(drop=True)

lookup = coverage.set_index("frame")[
    "outage_age_bucket"
].to_dict()

records = []

print("\nReading tracking data...")

with open(ATD_PATH, "r", encoding="utf-8") as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        parts = line.split(":")

        if len(parts) < 3:
            continue

        try:
            frame = int(parts[0])
        except ValueError:
            continue

        if frame not in lookup:
            continue

        positions = extract_positions(
            parts[1],
            team_mapping
        )

        bucket = lookup[frame]

        for team in (0, 1):

            prefix = f"team{team}"

            metrics = spatial_metrics(
                positions[team],
                prefix
            )

            records.append({
                "frame": frame,
                "bucket": bucket,
                "team": team,
                **metrics,
            })


df = pd.DataFrame(records)

# ------------------------------------------------------------
# PLAYER COUNT STRATIFICATION
# ------------------------------------------------------------

df["count_bin"] = pd.cut(
    df["team0_count"].where(df["team"] == 0, df["team1_count"]),
    bins=[-1, 5, 10, 15, 20, 100],
    labels=["1-5", "6-10", "11-15", "16-20", "21+"]
)

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
print("Rows:", f"{len(df):,}")


# ------------------------------------------------------------
# OUT-OF-BOUNDS AUDIT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("OUT-OF-BOUNDS COORDINATE RATE")
print("=" * 75)

for team in (0, 1):

    group = df[df["team"] == team]

    print(f"\nTEAM {team}")

    for bucket in BUCKETS:

        subset = group[group["bucket"] == bucket]

        rate = subset["team%d_out_of_bounds_rate" % team]

        print(
            f"{bucket:12s} "
            f"mean={rate.mean():.6f} "
            f"valid={rate.notna().sum():,}"
        )


# ------------------------------------------------------------
# COUNT-CONTROLLED SPATIAL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("SPATIAL STRUCTURE BY PLAYER COUNT")
print("=" * 75)

for team in (0, 1):

    group = df[df["team"] == team]

    prefix = f"team{team}"

    print(f"\n{'=' * 60}")
    print(f"TEAM {team}")
    print(f"{'=' * 60}")

    for count_bin in ["1-5", "6-10", "11-15", "16-20", "21+"]:

        print(f"\nPLAYER COUNT: {count_bin}")

        count_group = group[
            group["count_bin"] == count_bin
        ]

        for bucket in BUCKETS:

            subset = count_group[
                count_group["bucket"] == bucket
            ]

            if subset.empty:
                continue

            radius = subset[f"{prefix}_median_radius"]
            iqr_x = subset[f"{prefix}_iqr_x"]
            iqr_y = subset[f"{prefix}_iqr_y"]

            print(
                f"{bucket:12s} "
                f"frames={len(subset):6,} "
                f"radius={radius.mean():.6f} "
                f"IQR_X={iqr_x.mean():.6f} "
                f"IQR_Y={iqr_y.mean():.6f}"
            )

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)