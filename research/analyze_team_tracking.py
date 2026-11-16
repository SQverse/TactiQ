
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, "starter")
from baseline import team_of_each_track

metadata_path = Path("data/tactical/BAR-ATH_tactical_atd.xml")
tracking_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")

team_of_track = team_of_each_track(metadata_path)

print("=== TRACK METADATA ===")
print("Total track slots:", len(team_of_track))

team_distribution = Counter(team_of_track)

for team, count in team_distribution.items():
    print(f"Team code {team}: {count} slots")

observations = Counter()
frames = 0

with open(tracking_path, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue

        _, tracking_data = line.split(":", 1)
        player_data, _ = tracking_data.rsplit(":", 1)
        players = player_data.split(";")

        for index, player in enumerate(players):
            if player.split(",")[0] == "NaN":
                continue

            if index >= len(team_of_track):
                observations["unknown"] += 1
                continue

            team = team_of_track[index]

            if team == 0:
                observations["team_0"] += 1
            elif team == 1:
                observations["team_1"] += 1
            else:
                observations["unknown"] += 1

        frames += 1

print("\n=== TOTAL PLAYER OBSERVATIONS ===")
print("Frames analyzed:", frames)
print("Team 0 observations:", observations["team_0"])
print("Team 1 observations:", observations["team_1"])
print("Unknown observations:", observations["unknown"])

print("\n=== AVERAGE PLAYERS PER FRAME ===")

if frames:
    print("Team 0:", round(observations["team_0"] / frames, 2))
    print("Team 1:", round(observations["team_1"] / frames, 2))
    print("Unknown:", round(observations["unknown"] / frames, 2))
