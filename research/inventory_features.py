
from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/possession_features_clean.csv")
OUTPUT = Path("data/processed/feature_inventory.csv")

df = pd.read_csv(INPUT)

def categorize_feature(name):
    n = name.lower()

    # Identitas frame
    if n == "frame":
        return "Frame Metadata"

    # Kualitas dan ketersediaan observasi
    if (
        "age_" in n
        or "valid_count" in n
        or "coverage" in n
        or n == "ball_valid"
    ):
        return "Observation Quality"

    # Fitur statistik rolling
    if any(x in n for x in ["rolling", "_mean_", "_std_"]):
        return "Temporal Context"

    # Fitur perubahan antar-frame
    if any(x in n for x in [
        "change",
        "displacement",
        "track_changed"
    ]):
        return "Temporal Dynamics"

    # Fitur posisi dan jarak
    if any(x in n for x in [
        "ball_x",
        "ball_y",
        "ball_speed",
        "nearest_distance",
        "distance_gap"
    ]):
        return "Spatial / Ball Tracking"

    # Jumlah pemain
    if "player_count" in n:
        return "Team Structure"

    # Fitur lain terkait tim
    if "team0" in n or "team1" in n:
        return "Team-related Features"

    return "Other / Needs Review"


def describe_feature(name):
    n = name.lower()

    if n == "frame":
        return "Urutan frame dalam pertandingan."

    if n == "ball_valid":
        return "Indikator ketersediaan koordinat bola."

    if "age_frames" in n:
        return "Usia observasi terakhir dalam satuan frame."

    if "age_seconds" in n:
        return "Usia observasi terakhir dalam satuan detik."

    if "valid_count" in n:
        return "Jumlah observasi valid dalam jendela waktu."

    if "coverage" in n:
        return "Informasi cakupan atau ketersediaan tracking."

    if "nearest_distance" in n:
        return "Jarak pemain terdekat terhadap bola atau perubahan jaraknya."

    if "distance_gap" in n:
        return "Selisih jarak terdekat antara kedua tim."

    if "ball_displacement" in n:
        return "Perpindahan posisi bola antar-observasi."

    if "ball_x" in n or "ball_y" in n:
        return "Koordinat posisi bola."

    if "ball_speed" in n:
        return "Kecepatan bola berdasarkan data tracking."

    if "player_count" in n:
        return "Jumlah pemain yang terdeteksi dari masing-masing tim."

    if "track_changed" in n:
        return "Indikator pergantian track pemain terdekat."

    if "change" in n:
        return "Perubahan nilai fitur dibandingkan observasi sebelumnya."

    if "_mean_" in n:
        return "Rata-rata fitur dalam jendela waktu tertentu."

    if "_std_" in n:
        return "Standar deviasi fitur dalam jendela waktu tertentu."

    return "Perlu pemeriksaan lebih lanjut."


records = []

for column in df.columns:
    series = df[column]

    non_null = int(series.notna().sum())
    missing = int(series.isna().sum())

    sample_values = (
        series.dropna()
        .head(3)
        .astype(str)
        .tolist()
    )

    records.append({
        "feature": column,
        "category": categorize_feature(column),
        "description": describe_feature(column),
        "dtype": str(series.dtype),
        "non_null": non_null,
        "missing": missing,
        "missing_percent": round(
            missing / len(df) * 100, 2
        ),
        "unique_values": int(series.nunique()),
        "sample_values": " | ".join(sample_values)
    })

inventory = pd.DataFrame(records)

inventory.to_csv(OUTPUT, index=False)

print("=" * 70)
print("FEATURE INVENTORY")
print("=" * 70)

print(f"Total rows    : {len(df):,}")
print(f"Total features: {len(df.columns)}")
print()

summary = (
    inventory.groupby("category")
    .agg(
        feature_count=("feature", "count"),
        average_missing=("missing_percent", "mean")
    )
    .sort_values("feature_count", ascending=False)
)

print(summary.round(2).to_string())

print("\nFEATURE DETAILS")
print("-" * 70)

for category, group in inventory.groupby("category", sort=False):
    print(f"\n[{category}]")

    for _, row in group.iterrows():
        print(
            f"- {row['feature']} "
            f"({row['missing_percent']}% missing)"
        )

print("\n" + "=" * 70)
print(f"Inventory saved to: {OUTPUT}")
print("Analysis complete.")
