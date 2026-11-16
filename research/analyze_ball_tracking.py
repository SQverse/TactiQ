
from pathlib import Path
from array import array
import numpy as np

tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")

total_frames = 0
ball_x_valid = 0
ball_y_valid = 0
ball_xy_valid = 0
ball_speed_valid = 0
ball_xy_speed_valid = 0

ball_x_values = array("d")
ball_y_values = array("d")
ball_speed_values = array("d")

# Menyimpan posisi bola untuk analisis perpindahan
previous_x = None
previous_y = None

displacement_values = array("d")
consecutive_valid_pairs = 0
missing_ball_frames = 0

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        total_frames += 1

        _, tracking_data = line.split(":", 1)
        _, ball_data = tracking_data.rsplit(":", 1)

        ball = ball_data.split(",")

        if len(ball) != 3:
            previous_x = None
            previous_y = None
            missing_ball_frames += 1
            continue

        x = float(ball[0])
        y = float(ball[1])
        speed = float(ball[2])

        x_valid = np.isfinite(x)
        y_valid = np.isfinite(y)
        speed_valid = np.isfinite(speed)

        if x_valid:
            ball_x_valid += 1
            ball_x_values.append(x)

        if y_valid:
            ball_y_valid += 1
            ball_y_values.append(y)

        if speed_valid:
            ball_speed_valid += 1
            ball_speed_values.append(speed)

        if x_valid and y_valid:
            ball_xy_valid += 1

            ball_x_values_current = x
            ball_y_values_current = y

            if previous_x is not None and previous_y is not None:
                dx = x - previous_x
                dy = y - previous_y

                displacement = float(np.hypot(dx, dy))
                displacement_values.append(displacement)
                consecutive_valid_pairs += 1

            previous_x = ball_x_values_current
            previous_y = ball_y_values_current

        else:
            previous_x = None
            previous_y = None
            missing_ball_frames += 1

        if x_valid and y_valid and speed_valid:
            ball_xy_speed_valid += 1


def print_distribution(values, title):
    print(f"\n{title}")

    if len(values) == 0:
        print("Tidak ada data valid.")
        return

    data = np.frombuffer(values, dtype=np.float64)

    print("Jumlah:", len(data))
    print("Minimum:", round(float(np.min(data)), 6))
    print("Maksimum:", round(float(np.max(data)), 6))
    print("Rata-rata:", round(float(np.mean(data)), 6))
    print("Median:", round(float(np.median(data)), 6))
    print("Persentil 95:", round(float(np.percentile(data, 95)), 6))
    print("Persentil 99:", round(float(np.percentile(data, 99)), 6))


print("=" * 55)
print("ANALISIS BALL TRACKING")
print("=" * 55)

print("Total frame:", total_frames)
print("Koordinat X valid:", ball_x_valid)
print("Koordinat Y valid:", ball_y_valid)
print("Koordinat X dan Y valid:", ball_xy_valid)
print("Kecepatan valid:", ball_speed_valid)
print("Koordinat dan kecepatan valid:", ball_xy_speed_valid)
print("Frame tanpa koordinat bola:", missing_ball_frames)

if total_frames > 0:
    print(
        "Persentase bola terdeteksi:",
        round(ball_xy_valid / total_frames * 100, 2),
        "%"
    )

print_distribution(ball_x_values, "DISTRIBUSI KOORDINAT X BOLA")
print_distribution(ball_y_values, "DISTRIBUSI KOORDINAT Y BOLA")
print_distribution(ball_speed_values, "DISTRIBUSI KECEPATAN BOLA")
print_distribution(
    displacement_values,
    "DISTRIBUSI PERPINDAHAN POSISI ANTAR-FRAME"
)

print("\nJumlah pasangan frame berurutan dengan posisi bola valid:",
      consecutive_valid_pairs)

print("\n" + "=" * 55)
print("ANALISIS SELESAI")
print("=" * 55)
