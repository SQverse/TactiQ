import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/possession_context_final.csv")
FPS = 25

df = pd.read_csv(INPUT)

# Klasifikasi berdasarkan ketersediaan koordinat bola.
ball_available = df["ball_valid"].astype(bool).to_numpy()

# Gunakan perubahan jumlah pemain sebagai salah satu
# indikator aktivitas tracking yang tersedia di dataset.
metrics = [
    "team0_player_count",
    "team1_player_count",
    "team0_player_count_change",
    "team1_player_count_change",
]

print("=" * 70)
print("PLAYER TRACKING BY BALL AVAILABILITY")
print("=" * 70)

groups = {
    "Ball available": ball_available,
    "Ball unavailable": ~ball_available,
}

for group_name, mask in groups.items():
    print(f"\n{'=' * 55}")
    print(group_name)
    print("Frames:", int(mask.sum()))

    for metric in metrics:
        values = df.loc[mask, metric].dropna()

        print(f"\n{metric}")
        print("Count:", len(values))
        print("Mean:", round(values.mean(), 5))
        print("Median:", round(values.median(), 5))
        print("Std:", round(values.std(), 5))
        print("Min:", round(values.min(), 5))
        print("Max:", round(values.max(), 5))

print("\nAnalysis complete.")