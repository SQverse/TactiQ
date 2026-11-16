
from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/possession_features_clean.csv")
OUTPUT = Path("data/processed/ball_outage_feature_availability.csv")

df = pd.read_csv(INPUT)

# Identifikasi frame ketika bola tidak terdeteksi
ball_missing = ~df["ball_valid"].astype(bool)

# Hitung durasi berturut-turut bola tidak terdeteksi
streak_start = (
    ball_missing
    & ~ball_missing.shift(1, fill_value=False)
)

streak_id = streak_start.cumsum()

outage_frames = (
    df.loc[ball_missing, ["frame"]]
    .copy()
)

outage_frames["outage_seconds"] = (
    outage_frames
    .groupby(streak_id[ball_missing])
    .cumcount()
    .add(1)
    / 25
)

# Kelompok durasi kehilangan bola
outage_frames["duration_group"] = pd.cut(
    outage_frames["outage_seconds"],
    bins=[0, 1, 3, 5, 10, float("inf")],
    labels=["<=1s", "1-3s", "3-5s", "5-10s", ">10s"],
    include_lowest=True
)

# Pilih kelompok fitur yang akan dianalisis
selected_features = []

for col in df.columns:
    if col in ["frame", "ball_valid"]:
        continue

    if (
        "_mean_" in col
        or "_std_" in col
        or "_age_seconds" in col
        or "_valid_count_" in col
        or "player_count" in col
    ):
        selected_features.append(col)

results = []

for group in outage_frames["duration_group"].cat.categories:

    group_indices = outage_frames.index[
        outage_frames["duration_group"] == group
    ]

    subset = df.loc[group_indices]

    for feature in selected_features:

        valid_count = int(subset[feature].notna().sum())
        total_count = len(subset)

        results.append({
            "duration_group": str(group),
            "feature": feature,
            "total_frames": total_count,
            "valid_frames": valid_count,
            "valid_percent": round(
                valid_count / total_count * 100, 2
            ) if total_count else 0
        })

result = pd.DataFrame(results)

result.to_csv(OUTPUT, index=False)

print("=" * 75)
print("BALL OUTAGE FEATURE AVAILABILITY")
print("=" * 75)

print(f"Total frames: {len(df):,}")
print(f"Ball missing frames: {ball_missing.sum():,}")
print(f"Features analyzed: {len(selected_features)}")

print("\nOUTAGE DURATION SUMMARY")
print("-" * 75)

print(
    outage_frames.groupby(
        "duration_group",
        observed=False
    ).size().to_string()
)

print("\nFEATURE AVAILABILITY")
print("-" * 75)

for group in outage_frames["duration_group"].cat.categories:

    print(f"\n[{group}]")

    display = result[
        result["duration_group"] == str(group)
    ].sort_values("valid_percent", ascending=False)

    print(
        display.head(15)[
            ["feature", "valid_percent"]
        ].to_string(index=False)
    )

print("\n" + "=" * 75)
print(f"Saved: {OUTPUT}")
print("Analysis complete.")
