import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/tactical/BAR-ATH_tactical_atd.txt")
META = Path("data/tactical/BAR-ATH_tactical_atd.xml")

from starter.baseline import team_of_each_track

teams = team_of_each_track(META)

track_groups = {
    team: np.where(teams == team)[0]
    for team in [0, 1]
}

previous = None
records = []

with open(INPUT, "r", encoding="utf-8") as file:
    for frame, line in enumerate(file):

        parts = line.strip().split(":")
        players = parts[1].split(";")
        ball = parts[2].split(",")

        current = np.full((len(teams), 2), np.nan)

        for i, player in enumerate(players):
            xy = player.split(",")

            x, y = float(xy[0]), float(xy[1])

            if np.isfinite(x) and np.isfinite(y):
                current[i] = [x, y]

        ball_valid = (
            np.isfinite(float(ball[0]))
            and np.isfinite(float(ball[1]))
        )

        if previous is not None:

            valid = (
                np.isfinite(current).all(axis=1)
                & np.isfinite(previous).all(axis=1)
            )

            displacement = np.linalg.norm(
                current - previous,
                axis=1
            )

            row = {
                "frame": frame,
                "ball_valid": ball_valid
            }

            for team, tracks in track_groups.items():

                values = displacement[tracks][valid[tracks]]

                row[f"team{team}_movement"] = (
                    np.mean(values) if len(values) else np.nan
                )

            records.append(row)

        previous = current

df = pd.DataFrame(records)

print("=" * 70)
print("MOVEMENT TEMPORAL CONSISTENCY")
print("=" * 70)

for team in [0, 1]:

    col = f"team{team}_movement"

    print(f"\nTEAM {team}")

    for available, name in [
        (True, "Ball available"),
        (False, "Ball unavailable")
    ]:

        subset = df[df["ball_valid"] == available]
        values = subset[col]

        print(f"\n{name}")
        print("Valid frames:", values.notna().sum())
        print("Mean:", round(values.mean(), 7))
        print("Std:", round(values.std(), 7))

        # Autocorrelation satu frame (40 ms).
        autocorr = values.autocorr(lag=1)

        print(
            "Lag-1 autocorrelation:",
            round(autocorr, 5)
            if pd.notna(autocorr) else "N/A"
        )

        changes = values.diff().abs()

        print(
            "Mean absolute change:",
            round(changes.mean(), 7)
        )

print("\nAnalysis complete.")