
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path(
    "data/processed/possession_with_observation_age.csv"
)

OUTPUT = Path(
    "data/processed/feature_correlations.csv"
)

df = pd.read_csv(INPUT)

numeric = df.select_dtypes(
    include=[np.number]
).drop(
    columns=["frame"],
    errors="ignore"
)

# Hilangkan fitur konstan
numeric = numeric.loc[
    :, numeric.nunique(dropna=True) > 1
]

correlation = numeric.corr(
    method="spearman",
    min_periods=100
)

correlation.to_csv(OUTPUT)

print("=" * 70)
print("FEATURE DEPENDENCY ANALYSIS")
print("=" * 70)

print("Numeric features:", len(numeric.columns))

print("\nHIGH CORRELATION PAIRS (ABS >= 0.90)")

pairs = []

columns = correlation.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = correlation.iloc[i, j]

        if pd.notna(value) and abs(value) >= 0.90:

            pairs.append({
                "feature_1": columns[i],
                "feature_2": columns[j],
                "correlation": value
            })

pairs_df = pd.DataFrame(pairs)

if not pairs_df.empty:

    pairs_df = pairs_df.sort_values(
        "correlation",
        key=lambda x: x.abs(),
        ascending=False
    )

    print(pairs_df.to_string(index=False))

else:
    print("No high-correlation pairs found.")

print("\nTotal high-correlation pairs:", len(pairs))

print("\nSaved to:", OUTPUT)
print("\nAnalysis complete.")
