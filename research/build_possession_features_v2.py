
from pathlib import Path
import csv
import sys
import numpy as np

sys.path.insert(0, "starter")
from baseline import team_of_each_track

tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")
metadata_path = Path("data/tactical/BAR-ATH_tactical_atd.xml")
output_path = Path("data/processed/possession_features_v2.csv")

team_of_track = team_of_each_track(metadata_path)

output_path.parent.mkdir(parents=True, exist_ok=True)

columns = [
    "frame",
    "ball_valid",
    "ball_x",
    "ball_y",
    "team0_player_count",
    "team1_player_count",
    "team0_nearest_track",
    "team0_nearest_distance",
    "team0_second_nearest_distance",
    "team1_nearest_track",
    "team1_nearest_distance",
    "team1_second_nearest_distance",
    "nearest_distance_gap",
]

total_frames = 0
valid_ball_frames = 0
frames_both_teams = 0

with open(tracking_path, "r") as source, \
     open(output_path, "w", newline="") as destination:

    writer = csv.DictWriter(destination, fieldnames=columns)
    writer.writeheader()

    for line in source:
        line = line.strip()

        if not line:
            continue

        frame = total_frames
        total_frames += 1

        _, tracking_data = line.split(":", 1)
        player_data, ball_data = tracking_data.rsplit(":", 1)

        players = player_data.split(";")
        ball_values = ball_data.split(",")

        ball_x = None
        ball_y = None

        if len(ball_values) == 3:
            bx = float(ball_values[0])
            by = float(ball_values[1])

            if np.isfinite(bx) and np.isfinite(by):
                ball_x = bx
                ball_y = by
                valid_ball_frames += 1

        # Hitung pemain secara independen dari bola
        team_players = {
            0: [],
            1: [],
        }

        for index, player in enumerate(players):
            if index >= len(team_of_track):
                continue

            values = player.split(",")

            if len(values) != 3:
                continue

            x = float(values[0])
            y = float(values[1])

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            team = int(team_of_track[index])

            if team not in team_players:
                continue

            # Simpan posisi pemain dahulu
            team_players[team].append((index, x, y))

        # Jarak hanya dihitung jika bola tersedia
        team_distances = {
            0: [],
            1: [],
        }

        if ball_x is not None:
            for team in [0, 1]:
                for track_id, x, y in team_players[team]:
                    distance = float(np.hypot(
                        x - ball_x,
                        y - ball_y
                    ))

                    team_distances[team].append(
                        (distance, track_id)
                    )

                team_distances[team].sort()

        def get_item(team, position):
            data = team_distances[team]

            if len(data) > position:
                return data[position]

            return None

        def get_track(item):
            return item[1] if item is not None else ""

        def get_distance(item):
            return round(item[0], 8) if item is not None else ""

        t0_nearest = get_item(0, 0)
        t0_second = get_item(0, 1)
        t1_nearest = get_item(1, 0)
        t1_second = get_item(1, 1)

        if t0_nearest is not None and t1_nearest is not None:
            frames_both_teams += 1
            gap = round(t0_nearest[0] - t1_nearest[0], 8)
        else:
            gap = ""

        row = {
            "frame": frame,
            "ball_valid": int(ball_x is not None),
            "ball_x": ball_x if ball_x is not None else "",
            "ball_y": ball_y if ball_y is not None else "",
            "team0_player_count": len(team_players[0]),
            "team1_player_count": len(team_players[1]),
            "team0_nearest_track": get_track(t0_nearest),
            "team0_nearest_distance": get_distance(t0_nearest),
            "team0_second_nearest_distance": get_distance(t0_second),
            "team1_nearest_track": get_track(t1_nearest),
            "team1_nearest_distance": get_distance(t1_nearest),
            "team1_second_nearest_distance": get_distance(t1_second),
            "nearest_distance_gap": gap,
        }

        writer.writerow(row)

print("=" * 55)
print("FEATURE DATASET V2 BERHASIL DIBUAT")
print("=" * 55)
print("Total frame:", total_frames)
print("Frame dengan bola valid:", valid_ball_frames)
print("Frame kedua tim memiliki pemain dan bola:", frames_both_teams)
print("Output:", output_path)
print("Jumlah kolom:", len(columns))
