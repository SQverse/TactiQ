
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path(
    "data/processed/possession_with_observation_age.csv"
)

OUTPUT = Path(
    "data/processed/feature_readiness_audit.csv"
)

print("=" * 75)
print("FEATURE READINESS AUDIT")
print("=" * 75)

df = pd.read_csv(INPUT)

print("\nDATASET OVERVIEW")
print("Rows:", len(df))
print("Columns:", len(df.columns))

results = []

for col in df.columns:

    series = df[col]

    total = len(series)
    valid = series.notna().sum()
    missing = total - valid

    valid_pct = valid / total * 100 if total else 0
    missing_pct = missing / total * 100 if total else 0

    unique = series.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(series):

        mean = series.mean()
        std = series.std()
        minimum = series.min()
        maximum = series.max()

    else:

        mean = np.nan
        std = np.nan
        minimum = np.nan
        maximum = np.nan

    results.append({
        "feature": col,
        "dtype": str(series.dtype),
        "valid_count": valid,
        "missing_count": missing,
        "valid_pct": valid_pct,
        "missing_pct": missing_pct,
        "unique_values": unique,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum
    })

audit = pd.DataFrame(results)

audit = audit.sort_values(
    "valid_pct",
    ascending=False
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

audit.to_csv(OUTPUT, index=False)

print("\nFEATURE AVAILABILITY")
print(
    audit[
        [
            "feature",
            "valid_pct",
            "missing_pct",
            "unique_values"
        ]
    ].to_string(index=False)
)

print("\nCOVERAGE SUMMARY")

print(
    "\nFeatures with 100% availability:",
    (audit["valid_pct"] == 100).sum()
)

print(
    "Features with at least 75% availability:",
    (audit["valid_pct"] >= 75).sum()
)

print(
    "Features with at least 50% availability:",
    (audit["valid_pct"] >= 50).sum()
)

print(
    "Features with less than 25% availability:",
    (audit["valid_pct"] < 25).sum()
)

print("\nSaved to:", OUTPUT)
print("\nAudit complete.")
