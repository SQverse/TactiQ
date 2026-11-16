
import numpy as np
from pathlib import Path

from starter.baseline import team_of_each_track

INPUT = Path("data/tactical/BAR-ATH_tactical_atd.txt")
META = Path("data/tactical/BAR-ATH_tactical_atd.xml")

teams = team_of_each_track(META)

track_groups = {
    team: np.where(teams == team)[0]
    for team in [0, 1]
}

previous_positions = None
previous_ball_valid = None
previous_movement = {0: None, 1: None}

pairs = {
    (team, available): {
        "previous": [],
        "current": []
    }
    for team in [0, 1]
    for available in [True, False]
}

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

with open(INPUT, "r", encoding="utf-8") as file:

    for frame, line in enumerate(file):

        parts = line.strip().split(":")

        players = parts[1].split(";")
        ball = parts[2].split(",")

        current_positions = np.full(
            (len(teams), 2),
            np.nan
        )

        for i, player in enumerate(players):

            if i >= len(teams):
                break

            xy = player.split(",")

            x = safe_float(xy[0])
            y = safe_float(xy[1])

            if np.isfinite(x) and np.isfinite(y):
                current_positions[i] = [x, y]

        ball_valid = (
            np.isfinite(safe_float(ball[0]))
            and np.isfinite(safe_float(ball[1]))
        )

        if previous_positions is not None:

            valid = (
                np.isfinite(current_positions).all(axis=1)
                & np.isfinite(previous_positions).all(axis=1)
            )

            displacement = np.linalg.norm(
                current_positions - previous_positions,
                axis=1
            )

            current_movement = {}

            for team, tracks in track_groups.items():

                values = displacement[tracks][valid[tracks]]

                current_movement[team] = (
                    np.mean(values) if len(values) else np.nan
                )

                # Hanya pasangan frame berurutan
                # dengan status bola yang sama.
                if (
                    ball_valid == previous_ball_valid
                    and np.isfinite(current_movement[team])
                    and previous_movement[team] is not None
                    and np.isfinite(previous_movement[team])
                ):

                    bucket = pairs[(team, ball_valid)]

                    bucket["previous"].append(
                        previous_movement[team]
                    )

                    bucket["current"].append(
                        current_movement[team]
                    )

            previous_movement = current_movement

        previous_positions = current_positions
        previous_ball_valid = ball_valid

print("=" * 70)
print("MOVEMENT TEMPORAL CONSISTENCY V2")
print("=" * 70)

for team in [0, 1]:

    print(f"\nTEAM {team}")

    for available, name in [
        (True, "Ball available"),
        (False, "Ball unavailable")
    ]:

        bucket = pairs[(team, available)]

        previous = np.array(bucket["previous"])
        current = np.array(bucket["current"])

        print(f"\n{name}")
        print("Consecutive pairs:", len(current))

        if len(current) < 2:
            print("Insufficient data")
            continue

        autocorr = np.corrcoef(previous, current)[0, 1]

        mean_change = np.mean(
            np.abs(current - previous)
        )

        print(
            "Lag-1 autocorrelation:",
            round(autocorr, 6)
        )

        print(
            "Mean absolute change:",
            round(mean_change, 8)
        )

print("\nAnalysis complete.")
