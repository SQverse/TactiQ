
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

previous_positions = [None] * len(team_of_track)

stats = {
    0: {
        "valid_pairs": 0,
        "missing_pairs": 0,
        "displacements": array("d"),
    },
    1: {
        "valid_pairs": 0,
        "missing_pairs": 0,
        "displacements": array("d"),
    },
}

total_frames = 0
frames_with_previous_data = 0
frames_without_previous_data = 0

valid_tracks_per_frame = array("d")
valid_pairs_per_frame = array("d")

# Batasan: hanya membandingkan frame berurutan.
# Tidak menghubungkan track melewati frame yang hilang.

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        total_frames += 1

        _, tracking_data = line.split(":", 1)
        player_data, _ = tracking_data.rsplit(":", 1)

        players = player_data.split(";")

        current_positions = [None] * len(team_of_track)

        valid_tracks = 0
        valid_pairs = 0

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

            current_positions[index] = (x, y)
            valid_tracks += 1

            previous = previous_positions[index]

            if previous is not None:
                dx = x - previous[0]
                dy = y - previous[1]

                displacement = float(np.hypot(dx, dy))

                team = int(team_of_track[index])

                if team in stats:
                    stats[team]["valid_pairs"] += 1
                    stats[team]["displacements"].append(displacement)

                valid_pairs += 1

            else:
                team = int(team_of_track[index])

                if team in stats:
                    stats[team]["missing_pairs"] += 1

        valid_tracks_per_frame.append(valid_tracks)
        valid_pairs_per_frame.append(valid_pairs)

        if valid_pairs > 0:
            frames_with_previous_data += 1
        else:
            frames_without_previous_data += 1

        previous_positions = current_positions


def show_distribution(values, title):
    print(f"\n{title}")

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
print("ANALISIS PERGERAKAN PEMAIN ANTAR-FRAME")
print("=" * 55)

print("Total frame:", total_frames)

print(
    "Rata-rata pemain valid per frame:",
    round(float(np.mean(valid_tracks_per_frame)), 2)
)

print(
    "Rata-rata pasangan posisi berurutan per frame:",
    round(float(np.mean(valid_pairs_per_frame)), 2)
)

print("Frame dengan pasangan posisi valid:", frames_with_previous_data)
print("Frame tanpa pasangan posisi valid:", frames_without_previous_data)

for team in [0, 1]:
    s = stats[team]

    print("\n" + "=" * 55)
    print(f"TEAM {team}")
    print("=" * 55)

    print("Pasangan posisi valid:", s["valid_pairs"])
    print("Observasi tanpa posisi sebelumnya:", s["missing_pairs"])

    show_distribution(
        s["displacements"],
        "DISTRIBUSI PERPINDAHAN POSISI ANTAR-FRAME"
    )

print("\n" + "=" * 55)
print("ANALISIS SELESAI")
print("=" * 55)
