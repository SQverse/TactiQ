from pathlib import Path

file_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")

print("File exists:", file_path.exists())

with open(file_path, "r") as file:
    for i in range(3):
        line = file.readline()

        print(f"\n=== LINE {i + 1} ===")
        print(line[:1000])
