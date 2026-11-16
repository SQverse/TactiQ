
import pandas as pd
import numpy as np

INPUT = "data/processed/possession_with_observation_age.csv"
CORR = "data/processed/feature_correlations.csv"

df = pd.read_csv(INPUT)
corr = pd.read_csv(CORR, index_col=0)

pairs = []

columns = corr.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = corr.iloc[i, j]

        if pd.notna(value) and abs(value) >= 0.9999:

            a = df[columns[i]]
            b = df[columns[j]]

            valid = a.notna() & b.notna()

            if valid.sum() == 0:
                continue

            identical = np.isclose(
                a[valid].to_numpy(dtype=float),
                b[valid].to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-12
            ).all()

            pairs.append({
                "feature_1": columns[i],
                "feature_2": columns[j],
                "correlation": value,
                "valid_rows": int(valid.sum()),
                "identical_values": identical
            })

result = pd.DataFrame(pairs)

print("=" * 75)
print("REDUNDANT FEATURE VERIFICATION")
print("=" * 75)

if result.empty:
    print("No pairs found.")
else:
    print(result.to_string(index=False))

    print("\nIDENTICAL PAIRS:",
          result["identical_values"].sum())

    print("NON-IDENTICAL PAIRS:",
          (~result["identical_values"]).sum())

print("\nAnalysis complete.")
