
from pathlib import Path
from collections import defaultdict
from array import array
import sys
import numpy as np

sys.path.insert(0, "starter")
from baseline import team_of_each_track

tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")
metadata_path = Path("data/tactical/BAR-ATH_tactical_atd.xml")

team_of_track = team_of_each_track(metadata_path)

# Menyimpan posisi pemain dari frame sebelumnya
previous_players = [None] * len(team_of_track)
previous_ball = None

# Statistik perubahan jarak pemain terhadap bola
stats = {
    0: {
        "closer": 0,
        "farther": 0,
        "unchanged": 0,
        "changes": array("d"),
        "previous_distances": array("d"),
        "current_distances": array("d"),
    },
    1: {
        "closer": 0,
        "farther": 0,
        "unchanged": 0,
        "changes": array("d"),
        "previous_distances": array("d"),
        "current_distances": array("d"),
    },
}

total_frames = 0
valid_ball_pairs = 0
valid_player_ball_pairs = 0
frames_with_valid_comparison = 0

# Perubahan jarak hanya dihitung jika:
# 1. Bola terdeteksi di dua frame berurutan.
# 2. Track pemain yang sama memiliki koordinat valid
#    di dua frame berurutan.
# 3. Tidak ada interpolasi pada data yang hilang.

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        total_frames += 1

        _, tracking_data = line.split(":", 1)
        player_data, ball_data = tracking_data.rsplit(":", 1)

        players = player_data.split(";")

        current_players = [None] * len(team_of_track)
        current_ball = None

        for index, player in enumerate(players):
            if index >= len(team_of_track):
                continue

            values = player.split(",")

            if len(values) != 3:
                continue

            x = float(values[0])
            y = float(values[1])

            if np.isfinite(x) and np.isfinite(y):
                current_players[index] = (x, y)

        ball_values = ball_data.split(",")

        if len(ball_values) == 3:
            bx = float(ball_values[0])
            by = float(ball_values[1])

            if np.isfinite(bx) and np.isfinite(by):
                current_ball = (bx, by)

        # Bola harus valid di dua frame berurutan
        if previous_ball is not None and current_ball is not None:
            valid_ball_pairs += 1

            previous_bx, previous_by = previous_ball
            current_bx, current_by = current_ball

            for index in range(len(team_of_track)):
                previous_player = previous_players[index]
                current_player = current_players[index]

                if previous_player is None or current_player is None:
                    continue

                team = int(team_of_track[index])

                if team not in stats:
                    continue

                # Jarak pemain terhadap bola pada frame sebelumnya
                previous_distance = float(np.hypot(
                    previous_player[0] - previous_bx,
                    previous_player[1] - previous_by
                ))

                # Jarak pemain terhadap bola pada frame sekarang
                current_distance = float(np.hypot(
                    current_player[0] - current_bx,
                    current_player[1] - current_by
                ))

                # Perubahan jarak:
                # Negatif = semakin dekat
                # Positif = semakin jauh
                # Nol = jarak tetap
                change = current_distance - previous_distance

                s = stats[team]

                s["changes"].append(change)
                s["previous_distances"].append(previous_distance)
                s["current_distances"].append(current_distance)

                if change < 0:
                    s["closer"] += 1
                elif change > 0:
                    s["farther"] += 1
                else:
                    s["unchanged"] += 1

                valid_player_ball_pairs += 1

        if any(
            previous_players[i] is not None
            and current_players[i] is not None
            for i in range(len(team_of_track))
        ) and previous_ball is not None and current_ball is not None:
            frames_with_valid_comparison += 1

        previous_players = current_players
        previous_ball = current_ball


def show_stats(team, s):
    print("\n" + "=" * 55)
    print(f"TEAM {team}")
    print("=" * 55)

    changes = np.frombuffer(s["changes"], dtype=np.float64)
    previous_distances = np.frombuffer(
        s["previous_distances"], dtype=np.float64
    )
    current_distances = np.frombuffer(
        s["current_distances"], dtype=np.float64
    )

    if len(changes) == 0:
        print("Tidak ada pasangan data yang valid.")
        return

    print("Jumlah pasangan:", len(changes))

    print("\nARAH PERUBAHAN JARAK")
    print("Semakin dekat:", s["closer"])
    print("Semakin jauh:", s["farther"])
    print("Jarak tetap:", s["unchanged"])

    print("\nPERUBAHAN JARAK")
    print("Rata-rata:", round(float(np.mean(changes)), 7))
    print("Median:", round(float(np.median(changes)), 7))
    print("P05:", round(float(np.percentile(changes, 5)), 7))
    print("P95:", round(float(np.percentile(changes, 95)), 7))
    print("Minimum:", round(float(np.min(changes)), 7))
    print("Maksimum:", round(float(np.max(changes)), 7))

    print("\nJARAK PEMAIN TERHADAP BOLA")
    print(
        "Rata-rata jarak sebelumnya:",
        round(float(np.mean(previous_distances)), 6)
    )
    print(
        "Rata-rata jarak sekarang:",
        round(float(np.mean(current_distances)), 6)
    )


print("=" * 55)
print("ANALISIS PERUBAHAN JARAK PEMAIN TERHADAP BOLA")
print("=" * 55)

print("Total frame:", total_frames)
print("Pasangan frame dengan bola valid:", valid_ball_pairs)
print("Pasangan pemain-bola valid:", valid_player_ball_pairs)
print("Frame dengan perbandingan valid:", frames_with_valid_comparison)

for team in [0, 1]:
    show_stats(team, stats[team])

print("\n" + "=" * 55)
print("ANALISIS SELESAI")
print("=" * 55)
