from collections import Counter, defaultdict
import numpy as np
from starter.load_smart_tagging import load

events = load()

print("TOTAL EVENTS:", len(events))

HOME = 1
AWAY = 2

def smooth(codes, window):
    if window <= 0:
        return np.where(
            codes == HOME, "H",
            np.where(codes == AWAY, "A", "D")
        )

    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)

    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")

    return labels

# 1. Jumlah event per kode
print("\n=== EVENT COUNTS ===")
for code, count in Counter(e["code"] for e in events).most_common():
    print(f"{code:<35} {count:>4}")

# 2. Durasi setiap kode
durations = defaultdict(list)

for event in events:
    duration = event["end_s"] - event["start_s"]
    durations[event["code"]].append(duration)

print("\n=== DURATION PER CODE ===")
for code, values in durations.items():
    avg = sum(values) / len(values)
    print(
        f"{code:<35} "
        f"count={len(values):>3} "
        f"avg={avg:>6.2f}s "
        f"min={min(values):>6.2f}s "
        f"max={max(values):>6.2f}s"
    )

# 3. Jumlah event per tim
print("\n=== TEAM COUNTS ===")
team_counts = Counter(
    event["tags"].get("Team", "N/A")
    for event in events
)

for team, count in team_counts.items():
    print(f"{team:<25} {count:>4}")

# 4. Contoh 10 event pertama
print("\n=== FIRST 10 EVENTS ===")
for event in events[:10]:
    print({
        "code": event["code"],
        "start_frame": event["start_frame"],
        "end_frame": event["end_frame"],
        "team": event["tags"].get("Team"),
        "half": event["tags"].get("Half"),
        "type": event["tags"].get("Type"),
    })
    
# 5. Check overlapping events
print("\n=== OVERLAPPING EVENTS ===")

events_sorted = sorted(events, key=lambda e: e["start_frame"])

overlaps = []

for i, first in enumerate(events_sorted):
    for second in events_sorted[i + 1:]:
        if second["start_frame"] >= first["end_frame"]:
            break

        overlaps.append((
            first["code"],
            second["code"],
            first["tags"].get("Team"),
            second["tags"].get("Team"),
        ))

print("Total overlapping pairs:", len(overlaps))

pair_counts = Counter(
    (a, b) for a, b, _, _ in overlaps
)

for (a, b), count in pair_counts.most_common(15):
    print(f"{a:<32} + {b:<32} {count:>4}")

print("\n=== CROSS-TEAM OVERLAPS ===")

cross_team = [
    item for item in overlaps
    if item[2] not in (None, "N/A")
    and item[3] not in (None, "N/A")
    and item[2] != item[3]
]

print("Cross-team overlapping pairs:", len(cross_team))

# 6. Frame-level team overlap analysis

print("\n=== FRAME-LEVEL TEAM OVERLAP ===")

team_intervals = {
    "Barcelona": [],
    "Atletic Club": []
}

for event in events:
    team = event["tags"].get("Team")

    if team in team_intervals:
        team_intervals[team].append(
            (event["start_frame"], event["end_frame"])
        )

barcelona_frames = set()
athletic_frames = set()

for start, end in team_intervals["Barcelona"]:
    barcelona_frames.update(range(start, end))

for start, end in team_intervals["Atletic Club"]:
    athletic_frames.update(range(start, end))

both = barcelona_frames & athletic_frames

print("Barcelona annotated frames:", len(barcelona_frames))
print("Atletic Club annotated frames:", len(athletic_frames))
print("Both teams annotated:", len(both))

print(
    "Overlap ratio:",
    round(len(both) / max(1, len(barcelona_frames | athletic_frames)) * 100, 2),
    "%"
)

# 7. Team consistency per event code

print("\n=== TEAM CONSISTENCY PER CODE ===")

code_teams = {}

for event in events:
    code = event["code"]
    team = event["tags"].get("Team")

    if team in ("Barcelona", "Atletic Club"):
        code_teams.setdefault(code, Counter())
        code_teams[code][team] += 1

for code, counts in sorted(code_teams.items()):
    total = sum(counts.values())

    print(
        f"{code:<35} "
        f"Barcelona={counts['Barcelona']:>3} "
        f"Athletic={counts['Atletic Club']:>3} "
        f"total={total:>3}"
    )
    
# 8. Inspect selected event details

selected_codes = [
    "BUILD UP",
    "PROGRESSION",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION",
    "SET PIECES",
]

print("\n=== SELECTED EVENT DETAILS ===")

for code in selected_codes:
    print(f"\n--- {code} ---")

    selected = [
        event for event in events
        if event["code"] == code
    ]

    for event in selected[:5]:
        print({
            "start_s": event["start_s"],
            "end_s": event["end_s"],
            "team": event["tags"].get("Team"),
            "half": event["tags"].get("Half"),
            "type": event["tags"].get("Type"),
        })
        
from collections import defaultdict

print("\n=== PAIR VALIDATION ===")

pairs = [
    ("BUILD UP", "DEFENDING IN ATTACKING THIRD"),
    ("PROGRESSION", "DEFENDING IN MIDDLE THIRD"),
    ("ATTACKING TRANSITION", "DEFENSIVE TRANSITION"),
]

for code_a, code_b in pairs:
    events_a = [e for e in events if e["code"] == code_a]
    events_b = [e for e in events if e["code"] == code_b]

    matched = 0
    same_team = 0
    opposite_team = 0

    for a in events_a:
        for b in events_b:
            same_time = (
                abs(a["start_s"] - b["start_s"]) < 0.05
                and abs(a["end_s"] - b["end_s"]) < 0.05
            )

            if same_time:
                matched += 1

                if a["tags"].get("Team") == b["tags"].get("Team"):
                    same_team += 1
                else:
                    opposite_team += 1

    print(f"\n{code_a} vs {code_b}")
    print("Matched:", matched)
    print("Same team:", same_team)
    print("Opposite team:", opposite_team)
        
print("\n=== ATTACKING SIGNAL COVERAGE ===")

attacking_codes = [
    "BUILD UP",
    "PROGRESSION",
    "ATTACKING TRANSITION",
]

team_frames = {
    "Barcelona": set(),
    "Atletic Club": set(),
}

for event in events:
    if event["code"] not in attacking_codes:
        continue

    team = event["tags"].get("Team")

    if team not in team_frames:
        continue

    start = round(event["start_s"] * 25)
    end = round(event["end_s"] * 25)

    for frame in range(start, end + 1):
        team_frames[team].add(frame)

barcelona = team_frames["Barcelona"]
athletic = team_frames["Atletic Club"]

print("Barcelona attacking frames:", len(barcelona))
print("Athletic attacking frames:", len(athletic))
print("Both teams signal:", len(barcelona & athletic))
print("Total unique frames:", len(barcelona | athletic))     

import pandas as pd
from pathlib import Path

print("\n=== SMART TAGGING VS BASELINE 115 ===")

baseline_path = Path("evaluation/results/baseline_smooth115.csv")

baseline = pd.read_csv(baseline_path)

print("Baseline columns:", baseline.columns.tolist())
print("Baseline rows:", len(baseline))
print(baseline.head())   

print("\n=== BASELINE LABEL DISTRIBUTION ===")

print(baseline["label"].value_counts())

print("\n=== SMART TAGGING VS BASELINE AGREEMENT ===")

for team in ["Barcelona", "Atletic Club"]:
    frames = team_frames[team]

    labels = baseline.loc[
        baseline["frame"].isin(frames), "label"
    ]

    print(f"\n{team}")
    print("Total tagged frames:", len(frames))
    print("Baseline labels:")
    print(labels.value_counts())
    
print("\n=== DISAGREEMENT ANALYSIS ===")

for team, expected_label in [
    ("Barcelona", "H"),
    ("Atletic Club", "A"),
]:
    frames = team_frames[team]

    selected = baseline[
        baseline["frame"].isin(frames)
    ]

    disagreement = selected[
        selected["label"] != expected_label
    ]

    print(f"\n{team}")
    print("Disagreement frames:", len(disagreement))
    print("Disagreement labels:")
    print(disagreement["label"].value_counts())

    print("Sample disagreement frames:")
    print(disagreement.head(10))
    
print("\n=== DISAGREEMENT INTERVALS ===")

for team, expected_label in [
    ("Barcelona", "H"),
    ("Atletic Club", "A"),
]:
    frames = team_frames[team]

    selected = baseline[
        baseline["frame"].isin(frames)
        & (baseline["label"] != expected_label)
    ]["frame"].tolist()

    intervals = []

    if selected:
        start = previous = selected[0]

        for frame in selected[1:]:
            if frame == previous + 1:
                previous = frame
            else:
                intervals.append((start, previous))
                start = previous = frame

        intervals.append((start, previous))

    intervals.sort(
        key=lambda interval: interval[1] - interval[0],
        reverse=True
    )

    print(f"\n{team}")
    print("Number of disagreement intervals:", len(intervals))
    print("Longest 10 intervals:")

    for start, end in intervals[:10]:
        print(
            f"Frames {start}-{end} | "
            f"Duration: {(end-start+1)/25:.2f} seconds"
        )
        
print("\n=== INSPECT LONGEST DISAGREEMENT ===")

check_start = 11499
check_end = 12021

for event in events:
    event_start = round(event["start_s"] * 25)
    event_end = round(event["end_s"] * 25)

    if event_start <= check_end and event_end >= check_start:
        print({
            "code": event["code"],
            "start_frame": event_start,
            "end_frame": event_end,
            "team": event["tags"].get("Team"),
            "half": event["tags"].get("Half"),
            "type": event["tags"].get("Type"),
        })

print("\nBaseline labels in this interval:")

segment = baseline[
    (baseline["frame"] >= check_start)
    & (baseline["frame"] <= check_end)
]

print(segment["label"].value_counts())        

print("\n=== BASELINE TIMELINE ===")

check_start = 11499
check_end = 12021

timeline_start = check_start - 150
timeline_end = check_end + 150

timeline = baseline[
    (baseline["frame"] >= timeline_start)
    & (baseline["frame"] <= timeline_end)
].copy()

timeline["group"] = (
    timeline["label"] != timeline["label"].shift()
).cumsum()

for _, group in timeline.groupby("group"):
    start = group["frame"].iloc[0]
    end = group["frame"].iloc[-1]
    label = group["label"].iloc[0]

    print({
        "start": start,
        "end": end,
        "label": label,
        "duration_seconds": round((end - start + 1) / 25, 2)
    })
    
print("\n=== RAW LABEL CHECK ===")

raw_path = Path("evaluation/results/baseline_raw.csv")

raw = pd.read_csv(raw_path)

check_start = 11499
check_end = 12021

raw_interval = raw[
    (raw["frame"] >= check_start)
    & (raw["frame"] <= check_end)
]

print(raw_interval["label"].value_counts())

print("\nPersentase label mentah:")
print(
    raw_interval["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n=== RAW LABEL TIMELINE ===")

raw_timeline = raw_interval.copy()

raw_timeline["group"] = (
    raw_timeline["label"] != raw_timeline["label"].shift()
).cumsum()

for _, group in raw_timeline.groupby("group"):
    start = group["frame"].iloc[0]
    end = group["frame"].iloc[-1]
    label = group["label"].iloc[0]

    print({
        "start": int(start),
        "end": int(end),
        "label": label,
        "duration_seconds": round((end - start + 1) / 25, 2)
    })
    
print("\n=== LOCAL SMOOTHING COMPARISON ===")

for window in [25, 50, 75, 100, 115, 125]:
    
    smoothed = smooth(
        raw["label"].map({"H": HOME, "A": AWAY, "D": 0}).to_numpy(),
        window
    )

    interval_labels = smoothed[
    (raw["frame"].to_numpy() >= 38499) &
    (raw["frame"].to_numpy() <= 38823)
    ]

    print(
        f"Window {window}: "
        f"H={np.sum(interval_labels == 'H')}, "
        f"A={np.sum(interval_labels == 'A')}, "
        f"D={np.sum(interval_labels == 'D')}"
    )
    

print("\n=== SECOND INTERVAL SMART TAGGING ===")

check_start = 38499
check_end = 38823

for code in [
    "BUILD UP",
    "PROGRESSION",
    "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD"
]:
    print(f"\n{code}:")

    for event in events:
        if event["code"] != code:
            continue

        start = event["start_frame"]
        end = event["end_frame"]

        if start <= check_end and end >= check_start:
            print({
                "start": start,
                "end": end,
                "team": event["tags"].get("Team"),
                "code": event["code"]
            })