
from pathlib import Path
from array import array
from collections import Counter
import sys
import numpy as np

sys.path.insert(0, "starter")
from baseline import team_of_each_track

tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")
metadata_path = Path("data/tactical/BAR-ATH_tactical_atd.xml")

team_of_track = team_of_each_track(metadata_path)

# Statistik
total_frames = 0
ball_valid_frames = 0
ball_with_players_frames = 0
ball_without_players_frames = 0

player_count_when_ball_valid = array("d")

nearest_overall = array("d")
nearest_team_0 = array("d")
nearest_team_1 = array("d")

all_player_distances = array("d")

team_presence = Counter()

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        total_frames += 1

        _, tracking_data = line.split(":", 1)
        player_data, ball_data = tracking_data.rsplit(":", 1)

        ball = ball_data.split(",")

        if len(ball) != 3:
            continue

        ball_x = float(ball[0])
        ball_y = float(ball[1])

        if not (np.isfinite(ball_x) and np.isfinite(ball_y)):
            continue

        ball_valid_frames += 1

        players = player_data.split(";")

        distances_0 = []
        distances_1 = []
        frame_distances = []

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

            if team not in (0, 1):
                continue

            distance = float(
                np.hypot(x - ball_x, y - ball_y)
            )

            frame_distances.append(distance)
            all_player_distances.append(distance)

            if team == 0:
                distances_0.append(distance)
            else:
                distances_1.append(distance)

        player_count_when_ball_valid.append(len(frame_distances))

        if not frame_distances:
            ball_without_players_frames += 1
            continue

        ball_with_players_frames += 1

        if distances_0:
            nearest_team_0.append(min(distances_0))
            team_presence["team_0"] += 1

        if distances_1:
            nearest_team_1.append(min(distances_1))
            team_presence["team_1"] += 1

        nearest_overall.append(min(frame_distances))


def show_distribution(values, title):
    print(f"\n=== {title} ===")

    if not values:
        print("Tidak ada data.")
        return

    data = np.frombuffer(values, dtype=np.float64)

    print("Jumlah:", len(data))
    print("Minimum:", round(float(np.min(data)), 6))
    print("Maksimum:", round(float(np.max(data)), 6))
    print("Rata-rata:", round(float(np.mean(data)), 6))
    print("Median:", round(float(np.median(data)), 6))
    print("P95:", round(float(np.percentile(data, 95)), 6))
    print("P99:", round(float(np.percentile(data, 99)), 6))


print("=" * 55)
print("ANALISIS JARAK BOLA DAN PEMAIN")
print("=" * 55)

print("Total frame:", total_frames)
print("Frame bola terdeteksi:", ball_valid_frames)
print("Frame bola dan pemain terdeteksi:", ball_with_players_frames)
print("Frame bola tanpa pemain terdeteksi:", ball_without_players_frames)

if ball_valid_frames:
    print(
        "Rata-rata pemain terdeteksi saat bola tersedia:",
        round(float(np.mean(player_count_when_ball_valid)), 2)
    )

print("\n=== KEHADIRAN TIM ===")
print("Frame dengan pemain Team 0:", team_presence["team_0"])
print("Frame dengan pemain Team 1:", team_presence["team_1"])

show_distribution(
    all_player_distances,
    "SEMUA JARAK PEMAIN KE BOLA"
)

show_distribution(
    nearest_overall,
    "JARAK PEMAIN TERDEKAT KE BOLA"
)

show_distribution(
    nearest_team_0,
    "JARAK PEMAIN TERDEKAT TEAM 0"
)

show_distribution(
    nearest_team_1,
    "JARAK PEMAIN TERDEKAT TEAM 1"
)

print("\n" + "=" * 55)
print("ANALISIS SELESAI")
print("=" * 55)
