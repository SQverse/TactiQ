
from pathlib import Path
import numpy as np
import pandas as pd

from starter.baseline import team_of_each_track

TRACKING_FILE = Path("data/tactical/BAR-ATH_tactical_atd.txt")
META_FILE = Path("data/tactical/BAR-ATH_tactical_atd.xml")
OUTPUT_FILE = Path("data/processed/player_coverage_during_outage.csv")

FPS = 25

# ------------------------------------------------------------
# 1. TEAM MAPPING
# ------------------------------------------------------------

team_mapping = team_of_each_track(str(META_FILE))

team0_slots = np.where(team_mapping == 0)[0]
team1_slots = np.where(team_mapping == 1)[0]

print("Team 0 slots:", len(team0_slots))
print("Team 1 slots:", len(team1_slots))

# ------------------------------------------------------------
# 2. READ TRACKING AND COUNT VALID PLAYER POSITIONS
# ------------------------------------------------------------

team0_counts = []
team1_counts = []
frame_numbers = []
ball_valid_flags = []

with TRACKING_FILE.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file):
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

        player_data = parts[1].split(";")

        player_valid = np.zeros(len(team_mapping), dtype=bool)

        for slot, item in enumerate(player_data[:len(team_mapping)]):
            values = item.split(",")

            if len(values) < 2:
                continue

            try:
                x = float(values[0])
                y = float(values[1])
            except ValueError:
                continue

            if np.isfinite(x) and np.isfinite(y):
                player_valid[slot] = True

        ball_data = parts[2].split(",")

        try:
            ball_x = float(ball_data[0])
            ball_y = float(ball_data[1])
            ball_valid = np.isfinite(ball_x) and np.isfinite(ball_y)
        except (ValueError, IndexError):
            ball_valid = False

        team0_counts.append(player_valid[team0_slots].sum())
        team1_counts.append(player_valid[team1_slots].sum())

        frame_numbers.append(frame_number)
        ball_valid_flags.append(ball_valid)

        if (line_number + 1) % 50000 == 0:
            print(f"Processed {line_number + 1:,} lines")

# ------------------------------------------------------------
# 3. BUILD DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame({
    "frame": frame_numbers,
    "ball_valid": ball_valid_flags,
    "team0_player_count": team0_counts,
    "team1_player_count": team1_counts,
})

df = df.sort_values("frame").reset_index(drop=True)

# ------------------------------------------------------------
# 4. CALCULATE OUTAGE AGE
# ------------------------------------------------------------

ball_missing = ~df["ball_valid"]

run_id = ball_missing.ne(
    ball_missing.shift(fill_value=False)
).cumsum()

elapsed_frames = df.groupby(run_id).cumcount()
elapsed_seconds = (elapsed_frames / FPS).where(ball_missing)

conditions = [
    ball_missing & elapsed_seconds.le(1),
    ball_missing & elapsed_seconds.gt(1) & elapsed_seconds.le(3),
    ball_missing & elapsed_seconds.gt(3) & elapsed_seconds.le(5),
    ball_missing & elapsed_seconds.gt(5) & elapsed_seconds.le(10),
    ball_missing & elapsed_seconds.gt(10),
]

buckets = ["<=1s", "1-3s", "3-5s", "5-10s", ">10s"]

df["outage_age_bucket"] = np.select(
    conditions,
    buckets,
    default="not_outage"
)

# ------------------------------------------------------------
# 5. CALCULATE COVERAGE AND TOTAL PLAYER COUNT
# ------------------------------------------------------------

df["total_player_count"] = (
    df["team0_player_count"] + df["team1_player_count"]
)

df["team0_coverage_percent"] = (
    df["team0_player_count"] / len(team0_slots) * 100
)

df["team1_coverage_percent"] = (
    df["team1_player_count"] / len(team1_slots) * 100
)

# ------------------------------------------------------------
# 6. VALIDATE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PLAYER COVERAGE VALIDATION")
print("=" * 70)

print("Total frames:", len(df))
print("Ball missing:", ball_missing.sum())
print("Outage frames classified:", (df["outage_age_bucket"] != "not_outage").sum())

assert len(df) == 199319
assert ball_missing.sum() == 85831
assert (df["outage_age_bucket"] != "not_outage").sum() == 85831

# ------------------------------------------------------------
# 7. SUMMARY BY OUTAGE AGE
# ------------------------------------------------------------

metrics = [
    "team0_player_count",
    "team1_player_count",
    "total_player_count",
    "team0_coverage_percent",
    "team1_coverage_percent",
]

print("\n" + "=" * 70)
print("PLAYER COVERAGE BY OUTAGE AGE")
print("=" * 70)

summary = (
    df.groupby("outage_age_bucket")[metrics]
    .agg(["mean", "median", "std", "min", "max"])
    .round(4)
)

print(summary)

# ------------------------------------------------------------
# 8. SAVE
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
