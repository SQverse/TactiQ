
import pandas as pd
import numpy as np

INPUT = "data/processed/feature_correlations.csv"

corr = pd.read_csv(INPUT, index_col=0)

pairs = []

columns = corr.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = corr.iloc[i, j]

        if pd.notna(value) and abs(value) >= 0.90:
            pairs.append({
                "feature_1": columns[i],
                "feature_2": columns[j],
                "correlation": value
            })

result = pd.DataFrame(pairs)

result["abs_correlation"] = (
    result["correlation"].abs()
)

result = result.sort_values(
    "abs_correlation",
    ascending=False
)

print("=" * 75)
print("TOP 40 FEATURE CORRELATIONS")
print("=" * 75)

print(
    result.head(40)[
        ["feature_1", "feature_2", "correlation"]
    ].to_string(index=False)
)

print("\nNEGATIVE CORRELATIONS")

negative = result[
    result["correlation"] < -0.90
]

print(
    negative.head(20)[
        ["feature_1", "feature_2", "correlation"]
    ].to_string(index=False)
)

print("\nTotal high-correlation pairs:", len(result))
