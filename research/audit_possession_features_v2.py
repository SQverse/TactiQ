
import csv
import math
from pathlib import Path
from collections import Counter

path = Path("data/processed/possession_features_v2.csv")

total_rows = 0
frame_errors = []
invalid_values = Counter()

ball_counts = Counter()
player_counts = {
    0: [],
    1: [],
}

missing_distances = Counter()
distance_values = {
    "team0_nearest": [],
    "team0_second": [],
    "team1_nearest": [],
    "team1_second": [],
    "gap": [],
}

frames_without_ball_with_players = 0
frames_without_ball_without_players = 0
frames_with_both_teams = 0

previous_frame = None

with open(path, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

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

    print("Kolom sesuai:", reader.fieldnames == expected_columns)

    for row in reader:
        total_rows += 1

        frame = int(row["frame"])

        if previous_frame is not None and frame != previous_frame + 1:
            frame_errors.append((previous_frame, frame))

        previous_frame = frame

        ball_valid = int(row["ball_valid"])
        ball_counts[ball_valid] += 1

        count0 = int(row["team0_player_count"])
        count1 = int(row["team1_player_count"])

        player_counts[0].append(count0)
        player_counts[1].append(count1)

        if count0 < 0 or count1 < 0:
            invalid_values["negative_player_count"] += 1

        if ball_valid == 0:
            if row["ball_x"] != "" or row["ball_y"] != "":
                invalid_values["ball_without_coordinates"] += 1

            if count0 > 0 or count1 > 0:
                frames_without_ball_with_players += 1
            else:
                frames_without_ball_without_players += 1

            # Jarak dan gap seharusnya kosong tanpa bola
            for field in [
                "team0_nearest_distance",
                "team0_second_nearest_distance",
                "team1_nearest_distance",
                "team1_second_nearest_distance",
                "nearest_distance_gap",
            ]:
                if row[field] != "":
                    invalid_values[f"{field}_without_ball"] += 1

        else:
            if row["ball_x"] == "" or row["ball_y"] == "":
                invalid_values["ball_missing_coordinates"] += 1

        if ball_valid == 1 and count0 > 0 and count1 > 0:
            frames_with_both_teams += 1

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
                missing_distances[name] += 1
                continue

            number = float(value)

            if not math.isfinite(number):
                invalid_values[f"{name}_not_finite"] += 1
            else:
                distance_values[name].append(number)

        for team in [0, 1]:
            nearest = row[f"team{team}_nearest_distance"]
            second = row[f"team{team}_second_nearest_distance"]
            track = row[f"team{team}_nearest_track"]
            count = int(row[f"team{team}_player_count"])

            if count == 0 and (nearest != "" or track != ""):
                invalid_values[f"team{team}_empty_but_has_nearest"] += 1

            if count > 0 and ball_valid == 1:
                if nearest == "" or track == "":
                    invalid_values[f"team{team}_missing_nearest"] += 1

            if nearest != "" and second != "":
                if float(nearest) > float(second):
                    invalid_values[f"team{team}_distance_order"] += 1

        d0 = row["team0_nearest_distance"]
        d1 = row["team1_nearest_distance"]
        gap = row["nearest_distance_gap"]

        if d0 != "" and d1 != "":
            if gap == "":
                invalid_values["gap_missing"] += 1
            elif not math.isclose(
                float(gap),
                float(d0) - float(d1),
                abs_tol=1e-7
            ):
                invalid_values["gap_calculation"] += 1

        elif gap != "":
            invalid_values["gap_without_both_teams"] += 1


print("\n" + "=" * 60)
print("AUDIT DATASET POSSESSION FEATURES V2")
print("=" * 60)

print("Total baris:", total_rows)
print("Urutan frame bermasalah:", len(frame_errors))

print("\nKETERSEDIAAN BOLA")
print("Valid:", ball_counts[1])
print("Tidak valid:", ball_counts[0])

print("\nJUMLAH PEMAIN")
for team in [0, 1]:
    values = player_counts[team]
    print(f"\nTeam {team}")
    print("Rata-rata:", round(sum(values) / len(values), 2))
    print("Minimum:", min(values))
    print("Maksimum:", max(values))
    print("Frame dengan pemain:", sum(v > 0 for v in values))

print("\nFRAME TANPA BOLA")
print("Masih memiliki pemain:", frames_without_ball_with_players)
print("Tidak memiliki pemain:", frames_without_ball_without_players)

print("\nFRAME DENGAN BOLA DAN KEDUA TIM")
print(frames_with_both_teams)

print("\nNILAI JARAK KOSONG")
for name, count in missing_distances.items():
    print(f"{name}: {count}")

print("\nDISTRIBUSI JARAK")
for name, values in distance_values.items():
    if values:
        print(
            f"{name}: jumlah={len(values)}, "
            f"mean={sum(values)/len(values):.6f}, "
            f"min={min(values):.6f}, "
            f"max={max(values):.6f}"
        )

print("\nKESALAHAN")
if not invalid_values and not frame_errors:
    print("Tidak ditemukan kesalahan berdasarkan pemeriksaan ini.")
else:
    for name, count in invalid_values.items():
        print(f"{name}: {count}")

    if frame_errors:
        print("Contoh kesalahan frame:", frame_errors[:5])

print("\n" + "=" * 60)
print("AUDIT V2 SELESAI")
print("=" * 60)
