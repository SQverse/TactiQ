
from pathlib import Path
from collections import Counter
from statistics import mean, median

file_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")

player_distribution = Counter()
ball_distribution = Counter()

players_with_ball = []
players_without_ball = []

total_frames = 0

with open(file_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        frame_number, tracking_data = line.split(":", 1)
        player_data, ball_data = tracking_data.rsplit(":", 1)

        players = player_data.split(";")
        ball = ball_data.split(",")

        player_count = sum(
            1 for player in players
            if player.split(",")[0] != "NaN"
        )

        ball_available = ball[0] != "NaN"

        total_frames += 1
        player_distribution[player_count] += 1
        ball_distribution[ball_available] += 1

        if ball_available:
            players_with_ball.append(player_count)
        else:
            players_without_ball.append(player_count)

all_counts = [
    count
    for count, frequency in player_distribution.items()
    for _ in range(frequency)
]

print("=== GENERAL STATISTICS ===")
print("Total frames:", total_frames)
print("Average players:", round(mean(all_counts), 2))
print("Median players:", median(all_counts))

print("\n=== PLAYER TRACKING QUALITY ===")
print("Frames with exactly 22 players:",
      player_distribution[22])

print("Frames with fewer than 11 players:",
      sum(v for k, v in player_distribution.items() if k < 11))

print("Frames with more than 22 players:",
      sum(v for k, v in player_distribution.items() if k > 22))

print("\n=== BALL AVAILABILITY ===")
print("Ball available:", ball_distribution[True])
print("Ball unavailable:", ball_distribution[False])

print("\n=== PLAYERS WHEN BALL IS AVAILABLE ===")
print("Average:", round(mean(players_with_ball), 2))
print("Median:", median(players_with_ball))

print("\n=== PLAYERS WHEN BALL IS UNAVAILABLE ===")
print("Average:", round(mean(players_without_ball), 2))
print("Median:", median(players_without_ball))
