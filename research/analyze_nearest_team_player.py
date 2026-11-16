
from pathlib import Path
from collections import Counter
from array import array
import sys
import numpy as np

sys.path.insert(0, "starter")
from baseline import team_of_each_track

tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")
metadata_path = Path("data/tactical/BAR-ATH_tactical_atd.xml")

team_of_track = team_of_each_track(metadata_path)

stats = {
    0: {
        "distances": array("d"),
        "track_changes": 0,
        "track_observations": 0,
        "track_ids": Counter(),
    },
    1: {
        "distances": array("d"),
        "track_changes": 0,
        "track_observations": 0,
        "track_ids": Counter(),
    },
}

distance_gaps = array("d")
nearest_team0_ids = array("i")
nearest_team1_ids = array("i")

previous_nearest = {0: None, 1: None}

total_frames = 0
valid_ball_frames = 0
frames_both_teams = 0
frames_only_team0 = 0
frames_only_team1 = 0
frames_neither = 0
consecutive_comparisons = 0

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        total_frames += 1

        _, tracking_data = line.split(":", 1)
        player_data, ball_data = tracking_data.rsplit(":", 1)

        players = player_data.split(";")
        ball_values = ball_data.split(",")

        if len(ball_values) != 3:
            previous_nearest = {0: None, 1: None}
            continue

        bx = float(ball_values[0])
        by = float(ball_values[1])

        if not (np.isfinite(bx) and np.isfinite(by)):
            previous_nearest = {0: None, 1: None}
            continue

        valid_ball_frames += 1

        nearest = {
            0: (None, float("inf")),
            1: (None, float("inf")),
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

            if team not in nearest:
                continue

            distance = float(np.hypot(x - bx, y - by))

            if distance < nearest[team][1]:
                nearest[team] = (index, distance)

        id0, dist0 = nearest[0]
        id1, dist1 = nearest[1]

        if id0 is not None and id1 is not None:
            frames_both_teams += 1

            stats[0]["distances"].append(dist0)
            stats[1]["distances"].append(dist1)

            stats[0]["track_ids"][id0] += 1
            stats[1]["track_ids"][id1] += 1

            stats[0]["track_observations"] += 1
            stats[1]["track_observations"] += 1

            distance_gaps.append(dist0 - dist1)

            nearest_team0_ids.append(id0)
            nearest_team1_ids.append(id1)

            for team, current_id in [(0, id0), (1, id1)]:
                previous_id = previous_nearest[team]

                if previous_id is not None:
                    if current_id != previous_id:
                        stats[team]["track_changes"] += 1

            consecutive_comparisons += 1

        elif id0 is not None:
            frames_only_team0 += 1
            stats[0]["distances"].append(dist0)

        elif id1 is not None:
            frames_only_team1 += 1
            stats[1]["distances"].append(dist1)

        else:
            frames_neither += 1

        if id0 is not None and id1 is not None:
            previous_nearest = {0: id0, 1: id1}
        else:
            previous_nearest = {0: None, 1: None}


def print_distance_stats(team):
    values = np.frombuffer(
        stats[team]["distances"],
        dtype=np.float64
    )

    print(f"\nTEAM {team}")

    if len(values) == 0:
        print("Tidak ada data.")
        return

    print("Jumlah pengamatan:", len(values))
    print("Rata-rata jarak:", round(float(np.mean(values)), 6))
    print("Median jarak:", round(float(np.median(values)), 6))
    print("P95:", round(float(np.percentile(values, 95)), 6))
    print("P99:", round(float(np.percentile(values, 99)), 6))

    print("Pergantian track terdekat:", stats[team]["track_changes"])

    print("\n10 TRACK PALING SERING MENJADI TERDEKAT")

    for track_id, count in stats[team]["track_ids"].most_common(10):
        print(f"Track {track_id}: {count} kali")


print("=" * 60)
print("ANALISIS PEMAIN TERDEKAT DARI SETIAP TIM")
print("=" * 60)

print("Total frame:", total_frames)
print("Frame dengan bola valid:", valid_ball_frames)
print("Frame kedua tim memiliki pemain valid:", frames_both_teams)
print("Frame hanya Team 0 memiliki pemain:", frames_only_team0)
print("Frame hanya Team 1 memiliki pemain:", frames_only_team1)
print("Frame kedua tim tidak memiliki pemain:", frames_neither)

for team in [0, 1]:
    print_distance_stats(team)

print("\n" + "=" * 60)
print("PERBANDINGAN JARAK KEDUA TIM")
print("=" * 60)

gaps = np.frombuffer(distance_gaps, dtype=np.float64)

if len(gaps) > 0:
    print("Jumlah perbandingan:", len(gaps))

    # Positif: Team 0 lebih jauh.
    # Negatif: Team 1 lebih jauh.
    print("Rata-rata selisih (Team 0 - Team 1):",
          round(float(np.mean(gaps)), 6))

    print("Median selisih:",
          round(float(np.median(gaps)), 6))

    print("Team 0 lebih dekat:",
          int(np.sum(gaps < 0)))

    print("Team 1 lebih dekat:",
          int(np.sum(gaps > 0)))

    print("Jarak sama:", int(np.sum(gaps == 0)))

print("\n" + "=" * 60)
print("ANALISIS SELESAI")
print("=" * 60)
