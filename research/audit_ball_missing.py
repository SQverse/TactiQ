
from pathlib import Path
from collections import Counter
import math

tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")

patterns = Counter()
examples = {}

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        _, tracking_data = line.split(":", 1)
        _, ball_data = tracking_data.rsplit(":", 1)

        values = ball_data.split(",")

        if len(values) != 3:
            patterns["INVALID_FORMAT"] += 1
            continue

        x, y, speed = [
            float(value) for value in values
        ]

        x_valid = math.isfinite(x)
        y_valid = math.isfinite(y)
        speed_valid = math.isfinite(speed)

        key = (
            "VALID" if x_valid else "MISSING",
            "VALID" if y_valid else "MISSING",
            "VALID" if speed_valid else "MISSING",
        )

        patterns[key] += 1

        if key not in examples:
            examples[key] = values

print("=" * 55)
print("AUDIT MISSING DATA BALL TRACKING")
print("=" * 55)

for pattern, count in patterns.most_common():
    print(f"{pattern}: {count}")

print("\n=== CONTOH DATA SETIAP POLA ===")

for pattern, values in examples.items():
    print(f"{pattern}: {values}")

print("\nTotal pola:", sum(patterns.values()))
print("Audit selesai.")
