
from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/possession_features_clean.csv")
OUTPUT = Path("data/processed/ball_dependency_audit.csv")

df = pd.read_csv(INPUT)

ball_valid = df["ball_valid"].astype(bool)

results = []

for col in df.columns:
    if col in ["frame", "ball_valid"]:
        continue

    missing = df[col].isna()

    valid_when_ball_detected = (
        (~missing & ball_valid).sum()
        / max(ball_valid.sum(), 1) * 100
    )

    valid_when_ball_missing = (
        (~missing & ~ball_valid).sum()
        / max((~ball_valid).sum(), 1) * 100
    )

    missing_matches_ball_missing = (
        missing == ~ball_valid
    ).mean() * 100

    missing_matches_ball_valid = (
        missing == ball_valid
    ).mean() * 100

    results.append({
        "feature": col,
        "missing_percent": round(missing.mean() * 100, 2),
        "valid_when_ball_detected_percent": round(
            valid_when_ball_detected, 2
        ),
        "valid_when_ball_missing_percent": round(
            valid_when_ball_missing, 2
        ),
        "missing_pattern_matches_ball_missing_percent": round(
            missing_matches_ball_missing, 2
        ),
        "missing_pattern_matches_ball_valid_percent": round(
            missing_matches_ball_valid, 2
        )
    })

audit = pd.DataFrame(results)

audit = audit.sort_values(
    "missing_pattern_matches_ball_missing_percent",
    ascending=False
)

audit.to_csv(OUTPUT, index=False)

print("=" * 75)
print("BALL DEPENDENCY AUDIT")
print("=" * 75)

print(f"Total frames: {len(df):,}")
print(f"Ball detected: {ball_valid.sum():,}")
print(f"Ball missing: {(~ball_valid).sum():,}")

print("\nTOP FEATURES MATCHING BALL MISSINGNESS")
print("-" * 75)

print(
    audit.head(25).to_string(index=False)
)

print("\nFEATURES AVAILABLE WHEN BALL IS MISSING")
print("-" * 75)

independent = audit[
    audit["valid_when_ball_missing_percent"] >= 95
]

print(
    independent[
        [
            "feature",
            "valid_when_ball_missing_percent",
            "missing_percent"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print(f"Saved: {OUTPUT}")
print("Audit complete.")
