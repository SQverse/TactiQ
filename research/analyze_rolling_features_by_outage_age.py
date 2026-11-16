
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
    "data/processed/rolling_outage_age_analysis.csv"
)

FPS = 25

BUCKETS = ["<=1s", "1-3s", "3-5s", "5-10s", ">10s"]

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("ROLLING FEATURES BY ELAPSED BALL OUTAGE TIME")
print("=" * 75)

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("frame").reset_index(drop=True)

ball_missing = (
    pd.to_numeric(df["ball_valid"], errors="coerce")
    .fillna(0)
    .eq(0)
)

print(f"Total frames: {len(df):,}")
print(f"Ball missing: {ball_missing.sum():,}")

# ============================================================
# CALCULATE ELAPSED TIME SINCE OUTAGE START
# ============================================================

run_id = ball_missing.ne(
    ball_missing.shift(fill_value=False)
).cumsum()

elapsed_frames = df.groupby(run_id).cumcount()

elapsed_seconds = (
    elapsed_frames / FPS
).where(ball_missing)

conditions = [
    ball_missing & elapsed_seconds.le(1),
    ball_missing & elapsed_seconds.gt(1) & elapsed_seconds.le(3),
    ball_missing & elapsed_seconds.gt(3) & elapsed_seconds.le(5),
    ball_missing & elapsed_seconds.gt(5) & elapsed_seconds.le(10),
    ball_missing & elapsed_seconds.gt(10),
]

df["outage_age_bucket"] = np.select(
    conditions,
    BUCKETS,
    default="not_outage"
)

# ============================================================
# VALIDATION
# ============================================================

print("\nOUTAGE AGE BUCKET VALIDATION")

print(
    df["outage_age_bucket"]
    .value_counts()
    .reindex(["not_outage"] + BUCKETS, fill_value=0)
    .to_string()
)

classified = df["outage_age_bucket"].isin(BUCKETS).sum()

assert classified == ball_missing.sum(), (
    f"Mismatch: {classified} vs {ball_missing.sum()}"
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

print(f"\nRolling features found: {len(rolling_features)}")

# ============================================================
# ANALYSIS
# ============================================================

results = []

for bucket in BUCKETS:
    subset = df[df["outage_age_bucket"] == bucket]

    print("\n" + "=" * 75)
    print(f"[{bucket}]")
    print(f"Frames: {len(subset):,}")
    print("=" * 75)

    bucket_results = []

    for feature in rolling_features:
        values = pd.to_numeric(
            subset[feature], errors="coerce"
        )

        valid = values.dropna()

        row = {
            "outage_age_bucket": bucket,
            "feature": feature,
            "total_frames": len(subset),
            "valid_frames": len(valid),
            "valid_percent": (
                len(valid) / len(subset) * 100
                if len(subset) else np.nan
            ),
            "mean": valid.mean() if len(valid) else np.nan,
            "std": valid.std() if len(valid) else np.nan,
            "unique_values": valid.nunique(),
            "zero_percent": (
                (valid == 0).mean() * 100
                if len(valid) else np.nan
            ),
        }

        results.append(row)
        bucket_results.append(row)

    report = pd.DataFrame(bucket_results)

    if len(subset):
        print(
            report[
                [
                    "feature",
                    "valid_percent",
                    "std",
                    "unique_values",
                ]
            ].to_string(index=False)
        )

# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)
print(f"Saved: {OUTPUT_FILE}")
