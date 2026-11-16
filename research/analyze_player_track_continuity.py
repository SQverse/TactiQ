
from pathlib import Path

import numpy as np
import pandas as pd

from starter.baseline import team_of_each_track


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

ATD_PATH = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
META_PATH = ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"
COVERAGE_PATH = ROOT / "data/processed/player_coverage_during_outage.csv"

OUTPUT_PATH = ROOT / "data/processed/player_track_continuity.csv"

BUCKETS = [
    "not_outage",
    "<=1s",
    "1-3s",
    "3-5s",
    "5-10s",
    ">10s",
]


# ============================================================
# HELPERS
# ============================================================

def parse_active_tracks(player_data, team_mapping):
    """
    Return active track-slot indices for Team 0 and Team 1.

    A track is considered active when both x and y are finite.
    """

    active = {
        0: set(),
        1: set(),
    }

    player_entries = player_data.split(";")

    for track_id, entry in enumerate(player_entries):

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

        if team in active:
            active[team].add(track_id)

    return active


def calculate_pair_metrics(previous, current):
    """
    Calculate track-slot continuity between two adjacent frames.
    """

    retained = previous & current
    appeared = current - previous
    disappeared = previous - current

    previous_count = len(previous)
    current_count = len(current)

    retention_rate = (
        len(retained) / previous_count
        if previous_count > 0
        else np.nan
    )

    appearance_rate = (
        len(appeared) / current_count
        if current_count > 0
        else np.nan
    )

    union = previous | current

    jaccard = (
        len(retained) / len(union)
        if len(union) > 0
        else np.nan
    )

    return {
        "previous_count": previous_count,
        "current_count": current_count,
        "retained_count": len(retained),
        "appeared_count": len(appeared),
        "disappeared_count": len(disappeared),
        "retention_rate": retention_rate,
        "appearance_rate": appearance_rate,
        "jaccard": jaccard,
    }


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 75)
print("PLAYER TRACK CONTINUITY ANALYSIS")
print("=" * 75)

print("\nLoading team mapping...")

team_mapping = team_of_each_track(str(META_PATH))

print("Track slots:", len(team_mapping))
print("Team 0 slots:", int(np.sum(team_mapping == 0)))
print("Team 1 slots:", int(np.sum(team_mapping == 1)))


# ============================================================
# LOAD OUTAGE INFORMATION
# ============================================================

print("\nLoading outage information...")

coverage = pd.read_csv(COVERAGE_PATH)

coverage = coverage.sort_values("frame").reset_index(drop=True)

coverage["frame"] = coverage["frame"].astype(int)

coverage_lookup = coverage.set_index("frame")[
    ["ball_valid", "outage_age_bucket"]
].to_dict("index")

print("Coverage rows:", len(coverage_lookup))


# ============================================================
# STREAM RAW TRACKING
# ============================================================

print("\nReading raw tracking data...")

results = []

previous_frame = None
previous_active = None
previous_episode = None

episode_id = 0
previous_ball_missing = None

total_frames = 0
valid_pairs = 0
skipped_pairs = 0

with open(ATD_PATH, "r", encoding="utf-8") as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        parts = line.split(":")

        if len(parts) < 3:
            continue

        try:
            frame_number = int(parts[0])
        except ValueError:
            continue

        if frame_number not in coverage_lookup:
            continue

        total_frames += 1

        # Player coordinates are stored in the second section.
        current_active = parse_active_tracks(
            parts[1],
            team_mapping
        )

        info = coverage_lookup[frame_number]

        ball_missing = not bool(info["ball_valid"])
        bucket = info["outage_age_bucket"]

        # Create a new episode whenever ball availability changes.
        if previous_ball_missing is None:
            episode_id = 0

        elif ball_missing != previous_ball_missing:
            episode_id += 1

        current_episode = episode_id

        # Only compare consecutive frames in the same episode.
        valid_pair = (
            previous_frame is not None
            and frame_number == previous_frame + 1
            and current_episode == previous_episode
        )

        if valid_pair:

            valid_pairs += 1

            for team in (0, 1):

                metrics = calculate_pair_metrics(
                    previous_active[team],
                    current_active[team]
                )

                results.append({
                    "frame": frame_number,
                    "team": team,
                    "bucket": bucket,
                    **metrics,
                })

        else:
            if previous_frame is not None:
                skipped_pairs += 1

        previous_frame = frame_number
        previous_active = current_active
        previous_episode = current_episode
        previous_ball_missing = ball_missing

        if total_frames % 50000 == 0:
            print(f"Processed {total_frames:,} frames...")


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No valid adjacent frame pairs were found.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PROCESSING SUMMARY")
print("=" * 75)

print("Processed frames:", f"{total_frames:,}")
print("Valid adjacent pairs:", f"{valid_pairs:,}")
print("Skipped pairs:", f"{skipped_pairs:,}")
print("Team-level observations:", f"{len(df):,}")

print("\nSaved detailed results:")
print(OUTPUT_PATH)


# ============================================================
# AGGREGATED ANALYSIS
# ============================================================

for team in (0, 1):

    team_df = df[df["team"] == team]

    print("\n" + "=" * 60)
    print(f"TEAM {team}")
    print("=" * 60)

    for bucket in BUCKETS:

        group = team_df[team_df["bucket"] == bucket]

        if group.empty:
            continue

        total_previous = group["previous_count"].sum()
        total_current = group["current_count"].sum()

        total_retained = group["retained_count"].sum()
        total_appeared = group["appeared_count"].sum()
        total_disappeared = group["disappeared_count"].sum()

        pooled_retention = (
            total_retained / total_previous
            if total_previous > 0
            else np.nan
        )

        pooled_appearance = (
            total_appeared / total_current
            if total_current > 0
            else np.nan
        )

        print(f"\n[{bucket}]")
        print("Adjacent pairs:", f"{len(group):,}")

        print(
            "Mean previous tracks:",
            round(group["previous_count"].mean(), 4)
        )

        print(
            "Mean current tracks:",
            round(group["current_count"].mean(), 4)
        )

        print(
            "Mean retained tracks:",
            round(group["retained_count"].mean(), 4)
        )

        print(
            "Mean appeared tracks:",
            round(group["appeared_count"].mean(), 4)
        )

        print(
            "Mean disappeared tracks:",
            round(group["disappeared_count"].mean(), 4)
        )

        print(
            "Pooled retention rate (%):",
            round(pooled_retention * 100, 4)
        )

        print(
            "Pooled appearance rate (%):",
            round(pooled_appearance * 100, 4)
        )

        print(
            "Mean Jaccard similarity:",
            round(group["jaccard"].mean(), 4)
        )

        print(
            "Median Jaccard similarity:",
            round(group["jaccard"].median(), 4)
        )


print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)
