
import csv
from pathlib import Path
from collections import Counter
import math
import numpy as np

path = Path("data/processed/possession_features.csv")

expected_columns = [
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

total_rows = 0
frame_errors = []
column_errors = []

ball_counts = Counter()
both_team_counts = Counter()

distance_values = {
    "team0_nearest": [],
    "team0_second": [],
    "team1_nearest": [],
    "team1_second": [],
    "gap": [],
}

missing_values = Counter()
invalid_values = Counter()

previous_frame = None

with open(path, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    actual_columns = reader.fieldnames

    if actual_columns != expected_columns:
        column_errors.append(
            f"Kolom tidak sesuai. Ditemukan: {actual_columns}"
        )

    for row in reader:
        total_rows += 1

        # Periksa urutan frame
        frame = int(row["frame"])

        if previous_frame is not None and frame != previous_frame + 1:
            frame_errors.append((previous_frame, frame))

        previous_frame = frame

        # Periksa ketersediaan bola
        ball_valid = int(row["ball_valid"])
        ball_counts[ball_valid] += 1

        if ball_valid == 0:
            if row["ball_x"] != "" or row["ball_y"] != "":
                invalid_values["ball_without_coordinates"] += 1

        elif row["ball_x"] == "" or row["ball_y"] == "":
            invalid_values["ball_missing_coordinates"] += 1

        # Jumlah pemain terdeteksi
        count0 = int(row["team0_player_count"])
        count1 = int(row["team1_player_count"])

        if count0 < 0 or count1 < 0:
            invalid_values["negative_player_count"] += 1

        if ball_valid == 1 and count0 > 0 and count1 > 0:
            both_team_counts["both_present"] += 1

        # Periksa fitur jarak
        fields = {
            "team0_nearest": "team0_nearest_distance",
            "team0_second": "team0_second_nearest_distance",
            "team1_nearest": "team1_nearest_distance",
            "team1_second": "team1_second_nearest_distance",
            "gap": "nearest_distance_gap",
        }

        for name, field in fields.items():
            value = row[field]

            if value == "":
                missing_values[name] += 1
                continue

            number = float(value)

            if not math.isfinite(number):
                invalid_values[name] += 1
                continue

            distance_values[name].append(number)

        # Periksa konsistensi track dan jarak
        for team in [0, 1]:
            nearest_track = row[f"team{team}_nearest_track"]
            nearest_distance = row[f"team{team}_nearest_distance"]
            second_distance = row[f"team{team}_second_nearest_distance"]
            player_count = int(row[f"team{team}_player_count"])

            if player_count == 0:
                if nearest_track != "" or nearest_distance != "":
                    invalid_values[f"team{team}_empty_but_has_nearest"] += 1

            if player_count > 0:
                if nearest_track == "" or nearest_distance == "":
                    invalid_values[f"team{team}_missing_nearest"] += 1

            if nearest_distance != "" and second_distance != "":
                if float(nearest_distance) > float(second_distance):
                    invalid_values[f"team{team}_distance_order"] += 1

        # Gap hanya boleh ada jika kedua nearest tersedia
        gap = row["nearest_distance_gap"]
        d0 = row["team0_nearest_distance"]
        d1 = row["team1_nearest_distance"]

        if d0 != "" and d1 != "" and gap == "":
            invalid_values["gap_missing"] += 1

        if (d0 == "" or d1 == "") and gap != "":
            invalid_values["gap_without_both_teams"] += 1


print("=" * 60)
print("AUDIT DATASET POSSESSION FEATURES")
print("=" * 60)

print("Total baris:", total_rows)
print("Total kolom:", len(actual_columns or []))
print("Kolom sesuai:", len(column_errors) == 0)
print("Urutan frame bermasalah:", len(frame_errors))

print("\nKETERSEDIAAN BOLA")
print("Bola valid:", ball_counts[1])
print("Bola tidak valid:", ball_counts[0])

print("\nKETERSEDIAAN KEDUA TIM")
print("Kedua tim punya pemain saat bola valid:",
      both_team_counts["both_present"])

print("\nJUMLAH NILAI KOSONG")
for name, count in missing_values.items():
    print(f"{name}: {count}")

print("\nDISTRIBUSI FITUR JARAK")
for name, values in distance_values.items():
    if not values:
        print(f"{name}: tidak ada data")
        continue

    data = np.array(values, dtype=float)

    print(f"\n{name}")
    print("Jumlah:", len(data))
    print("Minimum:", round(float(np.min(data)), 8))
    print("Maksimum:", round(float(np.max(data)), 8))
    print("Rata-rata:", round(float(np.mean(data)), 8))
    print("Median:", round(float(np.median(data)), 8))

print("\nKESALAHAN DATA")
if not column_errors and not frame_errors and not invalid_values:
    print("Tidak ditemukan kesalahan struktural berdasarkan pemeriksaan ini.")
else:
    for error in column_errors:
        print(error)

    if frame_errors:
        print("Contoh urutan frame bermasalah:", frame_errors[:5])

    for name, count in invalid_values.items():
        print(f"{name}: {count}")

print("\n" + "=" * 60)
print("AUDIT SELESAI")
print("=" * 60)
