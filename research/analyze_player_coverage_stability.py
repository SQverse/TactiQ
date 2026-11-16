
import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/player_coverage_during_outage.csv"

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("frame").reset_index(drop=True)

teams = ["team0_player_count", "team1_player_count"]

for team in teams:
    df[f"{team}_change"] = df[team].diff().abs()

    df[f"{team}_coverage_change"] = (
        df[f"{team}_change"]
        / (24 if team.startswith("team0") else 22)
        * 100
    )

print("=" * 75)
print("PLAYER COVERAGE STABILITY")
print("=" * 75)

for bucket in ["not_outage", "<=1s", "1-3s", "3-5s", "5-10s", ">10s"]:
    subset = df[df["outage_age_bucket"] == bucket]

    print(f"\n[{bucket}]")
    print("Frames:", len(subset))

    for team in teams:
        change_col = f"{team}_change"
        coverage_change_col = f"{team}_coverage_change"

        values = subset[change_col].dropna()
        coverage_values = subset[coverage_change_col].dropna()

        print(f"\n{team}")
        print("Mean absolute count change:", round(values.mean(), 4))
        print("Median absolute count change:", round(values.median(), 4))
        print("P95 absolute count change:", round(values.quantile(0.95), 4))
        print("Mean coverage change (%):", round(coverage_values.mean(), 4))
        print("Unique count changes:", sorted(values.unique().tolist()))
