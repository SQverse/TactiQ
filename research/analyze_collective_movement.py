import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("data/tactical/BAR-ATH_tactical_atd.txt")
META = Path("data/tactical/BAR-ATH_tactical_atd.xml")

FPS = 25

# Import helper dari baseline tanpa mengubah baseline.py.
from starter.baseline import team_of_each_track

team_of_track = team_of_each_track(META)

team0_tracks = np.where(team_of_track == 0)[0]
team1_tracks = np.where(team_of_track == 1)[0]

# Baca tracking mentah.
previous_positions = None

stats = {
    0: [],
    1: []
}

ball_available = []

with open(INPUT, "r", encoding="utf-8") as file:

    for frame_idx, line in enumerate(file):

        parts = line.strip().split(":")

        player_data = parts[1].split(";")
        ball_data = parts[2].split(",")

        current_positions = np.full(
            (len(team_of_track), 2),
            np.nan
        )

        for track_idx, player in enumerate(player_data):

            values = player.split(",")

            if len(values) < 2:
                continue

            x = float(values[0])
            y = float(values[1])

            if np.isfinite(x) and np.isfinite(y):
                current_positions[track_idx] = [x, y]

        ball_x = float(ball_data[0])
        ball_y = float(ball_data[1])

        ball_valid = (
            np.isfinite(ball_x)
            and np.isfinite(ball_y)
        )

        ball_available.append(ball_valid)

        if previous_positions is not None:

            valid_pairs = (
                np.isfinite(current_positions).all(axis=1)
                & np.isfinite(previous_positions).all(axis=1)
            )

            displacement = np.linalg.norm(
                current_positions - previous_positions,
                axis=1
            )

            for team, tracks in [
                (0, team0_tracks),
                (1, team1_tracks)
            ]:

                team_valid = valid_pairs[tracks]

                values = displacement[tracks][team_valid]

                stats[team].append({
                    "frame": frame_idx,
                    "ball_valid": ball_valid,
                    "player_pairs": len(values),
                    "mean_displacement": (
                        float(np.mean(values))
                        if len(values) else np.nan
                    ),
                    "median_displacement": (
                        float(np.median(values))
                        if len(values) else np.nan
                    ),
                    "std_displacement": (
                        float(np.std(values))
                        if len(values) else np.nan
                    )
                })

        previous_positions = current_positions

print("=" * 70)
print("COLLECTIVE PLAYER MOVEMENT ANALYSIS")
print("=" * 70)

for team in [0, 1]:

    result = pd.DataFrame(stats[team])

    print(f"\nTEAM {team}")

    for name, mask in [
        ("Ball available", result["ball_valid"]),
        ("Ball unavailable", ~result["ball_valid"])
    ]:

        subset = result[mask]

        print(f"\n{name}")
        print("Frames:", len(subset))

        for column in [
            "player_pairs",
            "mean_displacement",
            "median_displacement",
            "std_displacement"
        ]:

            print(
                column,
                ":",
                round(subset[column].mean(), 6)
            )

print("\nAnalysis complete.")