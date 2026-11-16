
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path("data/processed/possession_features_clean.csv")
OUTPUT_FILE = Path("data/processed/non_ball_outage_analysis.csv")

FPS = 25

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("frame").reset_index(drop=True)

ball_missing = (
    pd.to_numeric(df["ball_valid"], errors="coerce")
    .fillna(0)
    .eq(0)
)

print("=" * 75)
print("NON-BALL FEATURES DURING BALL OUTAGE")
print("=" * 75)
print(f"Total frames: {len(df):,}")
print(f"Ball missing: {ball_missing.sum():,}")

# ============================================================
# 2. CALCULATE ELAPSED OUTAGE TIME
# ============================================================

run_id = ball_missing.ne(
    ball_missing.shift(fill_value=False)
).cumsum()

elapsed_frames = df.groupby(run_id).cumcount()
elapsed_seconds = (elapsed_frames / FPS).where(ball_missing)

conditions = [
    ball_missing & elapsed_seconds.le(1),
    ball_missing & elapsed_seconds.gt(1) & elapsed_seconds.le(3),
    ball_missing & elapsed_seconds.gt(3) & elapsed_seconds.le(5),
    ball_missing & elapsed_seconds.gt(5) & elapsed_seconds.le(10),
    ball_missing & elapsed_seconds.gt(10),
]

buckets = ["<=1s", "1-3s", "3-5s", "5-10s", ">10s"]

df["outage_age_bucket"] = np.select(
    conditions,
    buckets,
    default="not_outage"
)

print("\nOUTAGE BUCKET VALIDATION")
print(df["outage_age_bucket"].value_counts())

classified = df.loc[
    df["outage_age_bucket"] != "not_outage"
].shape[0]

assert classified == ball_missing.sum()

# ============================================================
# 3. SELECT FEATURES
# ============================================================

excluded_terms = [
    "ball",
    "nearest",
    "distance",
    "displacement",
    "gap",
    "track",
]

candidate_features = {}

for col in df.columns:
    if col in ["frame", "ball_valid", "outage_age_bucket"]:
        continue

    name = col.lower()

    # Exclude features directly dependent on ball/player proximity
    if any(term in name for term in excluded_terms):
        continue

    # Player counts
    if col in [
        "team0_player_count",
        "team1_player_count",
    ]:
        candidate_features[col] = "player_structure"

    # Changes in player counts
    elif col in [
        "team0_player_count_change",
        "team1_player_count_change",
    ]:
        candidate_features[col] = "player_count_dynamics"

    # Team-specific tracking coverage
    elif (
        "valid_count" in name
        and ("team0" in name or "team1" in name)
    ):
        candidate_features[col] = "player_tracking_coverage"

    # Observation age: freshness metadata, not independent evidence
    elif "age_" in name:
        candidate_features[col] = "observation_freshness"

print("\nFEATURES SELECTED")
for category in sorted(set(candidate_features.values())):
    features = [
        col for col, group in candidate_features.items()
        if group == category
    ]
    print(f"\n{category}: {len(features)}")
    for feature in features:
        print(f"  - {feature}")

# ============================================================
# 4. ANALYZE EACH FEATURE BY OUTAGE AGE
# ============================================================

results = []

for bucket in buckets:
    subset = df[df["outage_age_bucket"] == bucket]

    print("\n" + "=" * 75)
    print(f"[{bucket}]")
    print(f"Frames: {len(subset):,}")
    print("=" * 75)

    for feature, category in candidate_features.items():
        values = pd.to_numeric(
            subset[feature], errors="coerce"
        )

        valid = values.dropna()

        valid_percent = (
            len(valid) / len(subset) * 100
            if len(subset) else 0
        )

        row = {
            "outage_age_bucket": bucket,
            "category": category,
            "feature": feature,
            "total_frames": len(subset),
            "valid_count": len(valid),
            "valid_percent": valid_percent,
            "mean": valid.mean() if len(valid) else np.nan,
            "median": valid.median() if len(valid) else np.nan,
            "std": valid.std() if len(valid) > 1 else np.nan,
            "unique_values": valid.nunique(),
        }

        results.append(row)

    output = pd.DataFrame([
        r for r in results
        if r["outage_age_bucket"] == bucket
    ])

    if not output.empty:
        print(
            output[
                [
                    "category",
                    "feature",
                    "valid_percent",
                    "mean",
                    "std",
                    "unique_values",
                ]
            ].to_string(index=False)
        )

# ============================================================
# 5. SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)
print(f"Saved: {OUTPUT_FILE}")
