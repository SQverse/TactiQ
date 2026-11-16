
from pathlib import Path

import numpy as np
import pandas as pd

from starter.baseline import team_of_each_track


ROOT = Path(__file__).resolve().parent

ATD_PATH = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
META_PATH = ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"
COVERAGE_PATH = ROOT / "data/processed/player_coverage_during_outage.csv"

OUTPUT_PATH = ROOT / "data/processed/player_spatial_structure.csv"

BUCKETS = [
    "not_outage",
    "<=1s",
    "1-3s",
    "3-5s",
    "5-10s",
    ">10s",
]


def extract_team_positions(player_data, team_mapping):

    positions = {
        0: [],
        1: [],
    }

    entries = player_data.split(";")

    for track_id, entry in enumerate(entries):

        if track_id >= len(team_mapping):
            break

        if not entry:
            continue

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


def calculate_team_structure(points, prefix):

    if len(points) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_centroid_x": np.nan,
            f"{prefix}_centroid_y": np.nan,
            f"{prefix}_spread_x": np.nan,
            f"{prefix}_spread_y": np.nan,
            f"{prefix}_mean_radius": np.nan,
            f"{prefix}_range_x": np.nan,
            f"{prefix}_range_y": np.nan,
        }

    arr = np.asarray(points, dtype=float)

    centroid = arr.mean(axis=0)

    distances = np.linalg.norm(
        arr - centroid,
        axis=1
    )

    return {
        f"{prefix}_count": len(arr),
        f"{prefix}_centroid_x": centroid[0],
        f"{prefix}_centroid_y": centroid[1],
        f"{prefix}_spread_x": arr[:, 0].std(),
        f"{prefix}_spread_y": arr[:, 1].std(),
        f"{prefix}_mean_radius": distances.mean(),
        f"{prefix}_range_x": np.ptp(arr[:, 0]),
        f"{prefix}_range_y": np.ptp(arr[:, 1]),
    }


print("=" * 75)
print("PLAYER SPATIAL STRUCTURE ANALYSIS")
print("=" * 75)

print("\nLoading team mapping...")

team_mapping = team_of_each_track(str(META_PATH))

print("Team 0 slots:", int(np.sum(team_mapping == 0)))
print("Team 1 slots:", int(np.sum(team_mapping == 1)))


print("\nLoading coverage data...")

coverage = pd.read_csv(COVERAGE_PATH)
coverage = coverage.sort_values("frame").reset_index(drop=True)

coverage_lookup = coverage.set_index("frame")[
    ["ball_valid", "outage_age_bucket"]
].to_dict("index")


print("\nReading tracking data...")

records = []

total_frames = 0

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

        if frame not in coverage_lookup:
            continue

        total_frames += 1

        positions = extract_team_positions(
            parts[1],
            team_mapping
        )

        t0 = calculate_team_structure(
            positions[0],
            "team0"
        )

        t1 = calculate_team_structure(
            positions[1],
            "team1"
        )

        centroid_distance = np.nan

        if (
            np.isfinite(t0["team0_centroid_x"])
            and np.isfinite(t1["team1_centroid_x"])
        ):
            centroid_distance = np.sqrt(
                (
                    t0["team0_centroid_x"]
                    - t1["team1_centroid_x"]
                ) ** 2
                +
                (
                    t0["team0_centroid_y"]
                    - t1["team1_centroid_y"]
                ) ** 2
            )

        info = coverage_lookup[frame]

        records.append({
            "frame": frame,
            "bucket": info["outage_age_bucket"],
            **t0,
            **t1,
            "team_centroid_distance": centroid_distance,
        })

        if total_frames % 50000 == 0:
            print(f"Processed {total_frames:,} frames...")


df = pd.DataFrame(records)

# ------------------------------------------------------------
# TEMPORAL FEATURES
# ------------------------------------------------------------

df = df.sort_values("frame").reset_index(drop=True)

episode_id = (
    (~df["bucket"].eq("not_outage"))
    .ne((~df["bucket"].eq("not_outage")).shift(fill_value=False))
    .cumsum()
)

df["episode_id"] = episode_id

same_episode = df["episode_id"].eq(
    df["episode_id"].shift()
)

consecutive = df["frame"].diff().eq(1)

valid_pair = same_episode & consecutive

temporal_columns = [
    "team0_centroid_x",
    "team0_centroid_y",
    "team1_centroid_x",
    "team1_centroid_y",
]

for column in temporal_columns:

    change_column = column + "_change"

    df[change_column] = (
        df.groupby("episode_id")[column]
        .diff()
    )

    df.loc[~valid_pair, change_column] = np.nan

for team in (0, 1):

    x_change = df[f"team{team}_centroid_x_change"]
    y_change = df[f"team{team}_centroid_y_change"]

    df[f"team{team}_centroid_displacement"] = np.sqrt(
        x_change ** 2 + y_change ** 2
    )


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PROCESSING SUMMARY")
print("=" * 75)

print("Total frames:", f"{len(df):,}")
print("Saved:", OUTPUT_PATH)


summary_columns = [
    "team0_count",
    "team1_count",
    "team0_spread_x",
    "team0_spread_y",
    "team1_spread_x",
    "team1_spread_y",
    "team0_mean_radius",
    "team1_mean_radius",
    "team_centroid_distance",
    "team0_centroid_displacement",
    "team1_centroid_displacement",
]

for bucket in BUCKETS:

    group = df[df["bucket"] == bucket]

    if group.empty:
        continue

    print("\n" + "=" * 65)
    print(f"[{bucket}]")
    print("Frames:", f"{len(group):,}")

    for column in summary_columns:

        values = group[column].dropna()

        if values.empty:
            continue

        print(
            f"{column}: "
            f"mean={values.mean():.6f}, "
            f"median={values.median():.6f}, "
            f"valid={len(values):,}"
        )


print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)
