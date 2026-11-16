
import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/player_coverage_during_outage.csv"

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("frame").reset_index(drop=True)

# Identify continuous ball-availability episodes
ball_missing = ~df["ball_valid"].astype(bool)

episode_id = ball_missing.ne(
    ball_missing.shift(fill_value=False)
).cumsum()

df["episode_id"] = episode_id

# Only compare adjacent frames within the same episode
same_episode = df["episode_id"].eq(
    df["episode_id"].shift()
)

same_frame_sequence = df["frame"].diff().eq(1)

valid_pair = same_episode & same_frame_sequence

print("=" * 75)
print("PLAYER COUNT STABILITY - V2")
print("=" * 75)

teams = [
    ("team0_player_count", 24),
    ("team1_player_count", 22),
]

for column, total_slots in teams:

    change = df[column].diff().abs()

    # Exclude transitions across outage boundaries
    change = change.where(valid_pair)

    df[f"{column}_change"] = change

    print(f"\n{'=' * 60}")
    print(column)
    print(f"{'=' * 60}")

    for bucket in [
        "not_outage", "<=1s", "1-3s",
        "3-5s", "5-10s", ">10s"
    ]:

        subset = df[
            df["outage_age_bucket"] == bucket
        ]

        values = subset[
            f"{column}_change"
        ].dropna()

        if len(values) == 0:
            continue

        changed = values[values > 0]

        print(f"\n[{bucket}]")
        print("Valid adjacent pairs:", len(values))
        print("Changed pairs:", len(changed))
        print(
            "Change rate (%):",
            round(len(changed) / len(values) * 100, 4)
        )
        print("Mean absolute change:", round(values.mean(), 4))
        print("Median:", values.median())
        print("P95:", values.quantile(0.95))
        print("P99:", values.quantile(0.99))
        print("Maximum:", values.max())

        print("Change distribution:")
        print(
            values.value_counts()
            .sort_index()
            .to_string()
        )

print("\nANALYSIS COMPLETE")
