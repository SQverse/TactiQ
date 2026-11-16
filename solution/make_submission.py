"""One command from the provided data to a validated submission file.

    python solution/make_submission.py --team <team_name>

writes   submission/<team_name>.csv        <- this exact file is what you upload
prints   its SHA-256, so a later run can be compared byte for byte (RULES.md, rule 3)

It does three things, in order, and stops at the first one that fails:
  1. runs solution/possession_hmm.py on the data in ./data (or $MTAH_DATA_DIR);
  2. runs evaluation/validate_submission.py on the file it just wrote -- the same checks the
     official scorer applies, so a file that passes here is not dropped there;
  3. prints the hash and the row count.

It does NOT score the file against any ground truth and has no option to. The public five minutes
are only for checking that the pipeline is wired correctly (python evaluation/run_evaluation.py,
after copying the CSV into evaluation/results/); they are not a tuning set -- see RULES.md and the
organisers' clarification of 2026-11-04.

--team must be the team name from your registration form: SUBMISSION.md asks for <team_name>.csv.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAMES, GAME  # noqa: E402
from possession_hmm import shown_path  # noqa: E402

N_ROWS = GAMES[GAME]["n_video_frames"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, env=None):
    """Run a child process, echo its output, and return its exit code."""
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    sys.stdout.write(result.stdout)
    sys.stdout.flush()
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the file becomes <team>.csv")
    parser.add_argument("--out", default=os.path.join(ROOT, "submission"), help="output folder")
    parser.add_argument("--no-cache", action="store_true", help="re-read the ATD instead of .cache/")
    args = parser.parse_args(argv)

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]*$", args.team):
        print("FAIL  team name {!r}: use letters, digits, space, '_', '-' or '.' only".format(args.team))
        return 2

    pipeline = [sys.executable, os.path.join(HERE, "possession_hmm.py"),
                "--name", args.team, "--out", args.out]
    if args.no_cache:
        pipeline.append("--no-cache")

    print("[1/3] generating the possession file")
    if run(pipeline) != 0:
        print("FAIL  the pipeline did not finish; nothing was validated")
        return 1

    path = os.path.join(args.out, args.team + ".csv")
    print("\n[2/3] validating {}".format(shown_path(path)))
    check = [sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path]
    if run(check) != 0:
        print("\nFAIL  the validator rejected the file. Do NOT upload it.")
        return 1

    with open(path) as handle:
        rows = sum(1 for _ in handle) - 1
    if rows != N_ROWS:
        print("FAIL  {} data rows, expected {}".format(rows, N_ROWS))
        return 1

    print("\n[3/3] done")
    print("  file    {}".format(path))
    print("  rows    {}".format(rows))
    print("  sha256  {}".format(sha256(path)))
    print("\nUpload exactly this file. If you regenerate it, compare the sha256 above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
