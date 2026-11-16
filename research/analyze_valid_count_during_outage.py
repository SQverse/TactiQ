
from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/possession_features_clean.csv")
OUTPUT = Path("data/processed/valid_count_outage_analysis.csv")

df = pd.read_csv(INPUT)

ball_missing = ~df["ball_valid"].astype(bool)

streak_start = (
    ball_missing
    & ~ball_missing.shift(1, fill_value=False)
)

streak_id = streak_start.cumsum()

outage = df.loc[ball_missing].copy()

outage["outage_seconds"] = (
    outage.groupby(streak_id[ball_missing])
    .cumcount()
    .add(1) / 25
)

outage["duration_group"] = pd.cut(
    outage["outage_seconds"],
    bins=[0, 1, 3, 5, 10, float("inf")],
    labels=["<=1s", "1-3s", "3-5s", "5-10s", ">10s"],
    include_lowest=True
)

count_columns = [
    col for col in df.columns
    if "valid_count" in col
]

results = []

for group in outage["duration_group"].cat.categories:

    subset = outage[
        outage["duration_group"] == group
    ]

    for col in count_columns:

        values = subset[col].dropna()

        results.append({
            "duration_group": str(group),
            "feature": col,
            "frames": len(subset),
            "mean": values.mean() if len(values) else None,
            "median": values.median() if len(values) else None,
            "minimum": values.min() if len(values) else None,
            "maximum": values.max() if len(values) else None,
            "zero_percent": (
                (values == 0).mean() * 100
                if len(values) else None
            )
        })

result = pd.DataFrame(results)

result.to_csv(OUTPUT, index=False)

print("=" * 75)
print("VALID COUNT DURING BALL OUTAGE")
print("=" * 75)

for group in outage["duration_group"].cat.categories:

    print(f"\n[{group}]")

    subset = result[
        result["duration_group"] == str(group)
    ]

    print(
        subset[
            [
                "feature",
                "mean",
                "median",
                "zero_percent"
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 75)
print(f"Saved: {OUTPUT}")
print("Analysis complete.")
