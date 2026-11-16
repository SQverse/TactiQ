
import csv
import math
from pathlib import Path

input_path = Path("data/processed/possession_features_v2.csv")
output_path = Path("data/processed/possession_temporal_features.csv")

temporal_columns = [
    "team0_player_count_change",
    "team1_player_count_change",
    "ball_displacement",
    "team0_nearest_distance_change",
    "team1_nearest_distance_change",
    "nearest_distance_gap_change",
    "team0_nearest_track_changed",
    "team1_nearest_track_changed",
]

def number(value):
    if value == "":
        return None

    result = float(value)

    if not math.isfinite(result):
        return None

    return result


def difference(current, previous):
    if current is None or previous is None:
        return ""

    return round(current - previous, 8)


def track_changed(current, previous):
    if current == "" or previous == "":
        return ""

    return int(current != previous)


total_rows = 0
valid_ball_displacements = 0
valid_distance_changes = 0
valid_gap_changes = 0

previous = None

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(input_path, "r", newline="", encoding="utf-8") as source, \
     open(output_path, "w", newline="", encoding="utf-8") as destination:

    reader = csv.DictReader(source)

    output_columns = list(reader.fieldnames) + temporal_columns

    writer = csv.DictWriter(
        destination,
        fieldnames=output_columns
    )

    writer.writeheader()

    for row in reader:
        total_rows += 1

        # Perubahan jumlah pemain selalu dapat dihitung
        # karena jumlah pemain tidak bergantung pada bola.
        if previous is None:
            team0_count_change = ""
            team1_count_change = ""
        else:
            team0_count_change = (
                int(row["team0_player_count"])
                - int(previous["team0_player_count"])
            )

            team1_count_change = (
                int(row["team1_player_count"])
                - int(previous["team1_player_count"])
            )

        # Perpindahan bola hanya dihitung jika bola
        # tersedia pada dua frame berurutan.
        ball_displacement = ""

        current_ball_valid = row["ball_valid"] == "1"

        if previous is not None:
            previous_ball_valid = previous["ball_valid"] == "1"

            if current_ball_valid and previous_ball_valid:
                bx = number(row["ball_x"])
                by = number(row["ball_y"])

                prev_bx = number(previous["ball_x"])
                prev_by = number(previous["ball_y"])

                if None not in (bx, by, prev_bx, prev_by):
                    ball_displacement = round(
                        math.hypot(bx - prev_bx, by - prev_by),
                        8
                    )

                    valid_ball_displacements += 1

        # Perubahan jarak pemain terdekat.
        # Hanya dihitung jika kedua frame memiliki
        # koordinat bola dan jarak pemain yang valid.
        t0_distance_change = ""
        t1_distance_change = ""
        gap_change = ""

        t0_track_changed = ""
        t1_track_changed = ""

        if previous is not None:
            previous_ball_valid = previous["ball_valid"] == "1"

            if current_ball_valid and previous_ball_valid:
                for team in [0, 1]:
                    current_distance = number(
                        row[f"team{team}_nearest_distance"]
                    )

                    previous_distance = number(
                        previous[f"team{team}_nearest_distance"]
                    )

                    change = difference(
                        current_distance,
                        previous_distance
                    )

                    if team == 0:
                        t0_distance_change = change
                    else:
                        t1_distance_change = change

                    changed = track_changed(
                        row[f"team{team}_nearest_track"],
                        previous[f"team{team}_nearest_track"]
                    )

                    if team == 0:
                        t0_track_changed = changed
                    else:
                        t1_track_changed = changed

                current_gap = number(row["nearest_distance_gap"])
                previous_gap = number(previous["nearest_distance_gap"])

                gap_change = difference(current_gap, previous_gap)

                if t0_distance_change != "" or t1_distance_change != "":
                    valid_distance_changes += 1

                if gap_change != "":
                    valid_gap_changes += 1

        row.update({
            "team0_player_count_change": team0_count_change,
            "team1_player_count_change": team1_count_change,
            "ball_displacement": ball_displacement,
            "team0_nearest_distance_change": t0_distance_change,
            "team1_nearest_distance_change": t1_distance_change,
            "nearest_distance_gap_change": gap_change,
            "team0_nearest_track_changed": t0_track_changed,
            "team1_nearest_track_changed": t1_track_changed,
        })

        writer.writerow(row)

        previous = row


print("=" * 60)
print("TEMPORAL FEATURE DATASET BERHASIL DIBUAT")
print("=" * 60)

print("Total frame:", total_rows)
print("Perubahan posisi bola valid:", valid_ball_displacements)
print("Frame dengan perubahan jarak pemain:", valid_distance_changes)
print("Frame dengan perubahan gap:", valid_gap_changes)

print("Jumlah kolom:", len(output_columns))
print("Lokasi output:", output_path)

print("=" * 60)
