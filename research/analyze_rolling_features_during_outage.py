
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/possession_features_clean.csv"
)

OUTPUT_FILE = Path(
    "data/processed/rolling_outage_analysis.csv"
)

FPS = 25

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("ROLLING FEATURES DURING BALL OUTAGE")
print("=" * 75)

df = pd.read_csv(INPUT_FILE)

required = ["frame", "ball_valid"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.sort_values("frame").reset_index(drop=True)

ball_missing = (
    pd.to_numeric(df["ball_valid"], errors="coerce")
    .fillna(0)
    .eq(0)
)

print(f"\nTotal frames: {len(df):,}")
print(f"Ball missing: {ball_missing.sum():,}")

# ============================================================
# IDENTIFY CONSECUTIVE OUTAGE EPISODES
# ============================================================

group_id = ball_missing.ne(
    ball_missing.shift(fill_value=False)
).cumsum()

outage_id = group_id.where(ball_missing)

outage_duration = (
    ball_missing.groupby(outage_id)
    .transform("sum")
    .fillna(0)
)

outage_seconds = outage_duration / FPS

conditions = [
    ball_missing & outage_seconds.le(1),
    ball_missing & outage_seconds.gt(1) & outage_seconds.le(3),
    ball_missing & outage_seconds.gt(3) & outage_seconds.le(5),
    ball_missing & outage_seconds.gt(5) & outage_seconds.le(10),
    ball_missing & outage_seconds.gt(10),
]

labels = ["<=1s", "1-3s", "3-5s", "5-10s", ">10s"]

df["outage_bucket"] = np.select(
    conditions,
    labels,
    default="not_outage"
)

print("\nOUTAGE BUCKET VALIDATION")
print(df["outage_bucket"].value_counts().to_string())

classified = df["outage_bucket"].isin(labels).sum()

assert classified == ball_missing.sum(), (
    f"Mismatch: {classified} classified frames "
    f"vs {ball_missing.sum()} missing-ball frames"
)

print(f"\nValidated outage frames: {classified:,}")

# ============================================================
# SELECT ROLLING FEATURES
# ============================================================

rolling_features = [
    col for col in df.columns
    if (
        ("_mean_" in col or "_std_" in col)
        and (col.endswith("_1s") or col.endswith("_3s"))
    )
]

if not rolling_features:
    raise ValueError(
        "No rolling mean/std features found. "
        "Check the column naming convention."
    )

print(f"\nRolling features found: {len(rolling_features)}")

for col in rolling_features:
    print(" -", col)

# ============================================================
# ANALYZE EACH OUTAGE BUCKET
# ============================================================

results = []

for bucket in labels:
    subset = df[df["outage_bucket"] == bucket]

    print("\n" + "=" * 75)
    print(f"[{bucket}]")
    print(f"Frames: {len(subset):,}")
    print("=" * 75)

    for feature in rolling_features:
        values = pd.to_numeric(
            subset[feature], errors="coerce"
        )

        valid = values.dropna()

        results.append({
            "outage_bucket": bucket,
            "feature": feature,
            "total_frames": len(subset),
            "valid_frames": len(valid),
            "valid_percent": (
                len(valid) / len(subset) * 100
                if len(subset) else 0
            ),
            "mean": valid.mean() if len(valid) else np.nan,
            "std": valid.std() if len(valid) else np.nan,
            "unique_values": valid.nunique(),
            "zero_percent": (
                (valid == 0).mean() * 100
                if len(valid) else np.nan
            ),
        })

    report = pd.DataFrame([
        row for row in results
        if row["outage_bucket"] == bucket
    ])

    if not report.empty:
        print(
            report[
                [
                    "feature",
                    "valid_percent",
                    "mean",
                    "std",
                    "unique_values",
                    "zero_percent",
                ]
            ].to_string(index=False)
        )

# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)
print(f"Saved: {OUTPUT_FILE}")
