
from pathlib import Path
from collections import defaultdict, Counter
from array import array
import sys
import numpy as np

sys.path.insert(0, "starter")
from baseline import team_of_each_track

tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")
metadata_path = Path("data/tactical/BAR-ATH_tactical_atd.xml")

team_of_track = team_of_each_track(metadata_path)

# Penyimpanan statistik yang hemat memori
stats = {
    0: {
        "total": 0,
        "x_valid": 0,
        "y_valid": 0,
        "xy_valid": 0,
        "speed_valid": 0,
        "xy_speed_valid": 0,
        "out_of_range": 0,
        "x": array("d"),
        "y": array("d"),
        "speed": array("d"),
    },
    1: {
        "total": 0,
        "x_valid": 0,
        "y_valid": 0,
        "xy_valid": 0,
        "speed_valid": 0,
        "xy_speed_valid": 0,
        "out_of_range": 0,
        "x": array("d"),
        "y": array("d"),
        "speed": array("d"),
    },
}

unknown_slots = 0
frames = 0

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        _, tracking_data = line.split(":", 1)
        player_data, _ = tracking_data.rsplit(":", 1)

        players = player_data.split(";")

        for index, player in enumerate(players):
            if index >= len(team_of_track):
                unknown_slots += 1
                continue

            team = int(team_of_track[index])

            if team not in stats:
                unknown_slots += 1
                continue

            values = player.split(",")

            if len(values) != 3:
                continue

            x = float(values[0])
            y = float(values[1])
            speed = float(values[2])

            s = stats[team]
            s["total"] += 1

            x_valid = np.isfinite(x)
            y_valid = np.isfinite(y)
            speed_valid = np.isfinite(speed)

            if x_valid:
                s["x_valid"] += 1
                s["x"].append(x)

            if y_valid:
                s["y_valid"] += 1
                s["y"].append(y)

            if speed_valid:
                s["speed_valid"] += 1
                s["speed"].append(speed)

            if x_valid and y_valid:
                s["xy_valid"] += 1

                if not (0 <= x <= 1 and 0 <= y <= 1):
                    s["out_of_range"] += 1

            if x_valid and y_valid and speed_valid:
                s["xy_speed_valid"] += 1

        frames += 1


def print_distribution(values, title):
    print(f"\n{title}")

    if len(values) == 0:
        print("Tidak ada data valid.")
        return

    data = np.frombuffer(values, dtype=np.float64)

    print("Jumlah:", len(data))
    print("Minimum:", round(float(np.min(data)), 4))
    print("Maksimum:", round(float(np.max(data)), 4))
    print("Rata-rata:", round(float(np.mean(data)), 4))
    print("Median:", round(float(np.median(data)), 4))
    print("Persentil 95:", round(float(np.percentile(data, 95)), 4))
    print("Persentil 99:", round(float(np.percentile(data, 99)), 4))


print("=" * 55)
print("ANALISIS KUALITAS TRACKING")
print("=" * 55)

print("Total frame:", frames)
print("Total track slots:", len(team_of_track))
print("Slot tidak dikenal:", unknown_slots)

for team in [0, 1]:
    s = stats[team]

    print("\n" + "=" * 55)
    print(f"TEAM {team}")
    print("=" * 55)

    print("Total observasi slot:", s["total"])
    print("Koordinat X valid:", s["x_valid"])
    print("Koordinat Y valid:", s["y_valid"])
    print("Koordinat X dan Y valid:", s["xy_valid"])
    print("Kecepatan valid:", s["speed_valid"])
    print("Koordinat dan kecepatan valid:", s["xy_speed_valid"])
    print("Koordinat di luar rentang 0-1:", s["out_of_range"])

    if s["total"] > 0:
        print(
            "Persentase koordinat lengkap:",
            round(s["xy_valid"] / s["total"] * 100, 2),
            "%"
        )

        print(
            "Persentase kecepatan tersedia:",
            round(s["speed_valid"] / s["total"] * 100, 2),
            "%"
        )

    print_distribution(s["x"], "DISTRIBUSI KOORDINAT X")
    print_distribution(s["y"], "DISTRIBUSI KOORDINAT Y")
    print_distribution(s["speed"], "DISTRIBUSI KECEPATAN")

print("\n" + "=" * 55)
print("ANALISIS SELESAI")
print("=" * 55)
