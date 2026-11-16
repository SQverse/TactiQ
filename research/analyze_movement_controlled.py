import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("data/tactical/BAR-ATH_tactical_atd.txt")
META = Path("data/tactical/BAR-ATH_tactical_atd.xml")

from starter.baseline import team_of_each_track

team_of_track = team_of_each_track(META)

team_tracks = {
    0: np.where(team_of_track == 0)[0],
    1: np.where(team_of_track == 1)[0],
}

previous = None
records = []

with open(INPUT, "r", encoding="utf-8") as file:
    for frame_idx, line in enumerate(file):

        parts = line.strip().split(":")
        players = parts[1].split(";")
        ball = parts[2].split(",")

        current = np.full(
            (len(team_of_track), 2),
            np.nan
        )

        for i, player in enumerate(players):
            xy = player.split(",")

            x, y = float(xy[0]), float(xy[1])

            if np.isfinite(x) and np.isfinite(y):
                current[i] = [x, y]

        bx, by = float(ball[0]), float(ball[1])

        ball_valid = np.isfinite(bx) and np.isfinite(by)

        if previous is not None:

            valid = (
                np.isfinite(current).all(axis=1)
                & np.isfinite(previous).all(axis=1)
            )

            displacement = np.linalg.norm(
                current - previous,
                axis=1
            )

            counts = {
                team: int(
                    np.isfinite(current[tracks]).all(axis=1).sum()
                )
                for team, tracks in team_tracks.items()
            }

            for team, tracks in team_tracks.items():

                team_valid = valid[tracks]
                values = displacement[tracks][team_valid]

                records.append({
                    "frame": frame_idx,
                    "team": team,
                    "ball_valid": ball_valid,
                    "total_players": counts[0] + counts[1],
                    "player_pairs": len(values),
                    "mean_displacement": (
                        np.mean(values) if len(values) else np.nan
                    ),
                })

        previous = current

df = pd.DataFrame(records)

bins = [0, 5, 10, 15, 20, 25, 50]
labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26+"]

df["player_group"] = pd.cut(
    df["total_players"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print("=" * 70)
print("MOVEMENT CONTROLLED BY PLAYER COUNT")
print("=" * 70)

for team in [0, 1]:
    print(f"\nTEAM {team}")

    subset = df[df["team"] == team]

    for group in labels:
        group_df = subset[subset["player_group"] == group]

        if group_df.empty:
            continue

        print(f"\nPlayer group: {group}")

        for available, name in [
            (True, "Ball available"),
            (False, "Ball unavailable")
        ]:
            part = group_df[group_df["ball_valid"] == available]

            print(
                name,
                "| Frames:", len(part),
                "| Mean displacement:",
                round(part["mean_displacement"].mean(), 7)
            )

print("\nAnalysis complete.")