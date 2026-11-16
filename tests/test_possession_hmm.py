"""Self-checks for solution/possession_hmm.py. No match data needed; run with

    python tests/test_possession_hmm.py

Part 1 checks the pieces on tiny inputs. Part 2 builds a synthetic match (static players, a ball that
visits one player at a time, dead-ball spells, dropouts) in a temporary data folder and runs the whole
command line against it, then validates and scores the output with the repository's own tools.
The synthetic match is made up; passing says the code is wired correctly, not that it is good.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "solution"))
import possession_hmm as ph  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

PITCH = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]


def test_shown_path_survives_a_different_drive():
    """Windows: relpath raises ValueError across drives (C: temp folder, D: repository)."""
    real = os.path.relpath

    def across_drives(path, start=os.curdir):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    os.path.relpath = across_drives
    try:
        assert ph.shown_path("C:\\Temp\\x.csv") == "C:\\Temp\\x.csv"
    finally:
        os.path.relpath = real
    assert ph.shown_path(os.path.join(ROOT, "a", "b.csv")) == os.path.join("a", "b.csv")


def test_run_lengths():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert list(ph.run_lengths(mask)) == [0, 2, 2, 0, 3, 3, 3, 0, 0, 1]
    assert list(ph.run_lengths(np.zeros(0, dtype=bool))) == []


def test_viterbi_ignores_a_blip_and_keeps_a_real_change():
    fps = 25
    scores = np.zeros((600, 3))
    scores[:300, ph.H] = 0.5
    scores[:300, ph.A] = -0.5
    scores[300:, ph.H] = -0.5
    scores[300:, ph.A] = 0.5
    scores[100:110, ph.H], scores[100:110, ph.A] = -0.5, 0.5   # a 10-frame contradiction
    path = ph.viterbi(scores, ph.transition_penalties(fps))
    assert set(path[:300]) == {ph.H}, "blip was not smoothed away"
    assert set(path[300:]) == {ph.A}, "real change was lost"


def test_extract_features_parses_the_atd_line_format():
    team_of_track = np.array([ph.HOME, ph.AWAY], dtype=np.int8)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "atd.txt")
        with open(path, "w") as handle:
            handle.write("1:0.5,0.5,1.0;0.6,0.5,1.0:0.5,0.5,3.0\n")      # ball on the home player
            handle.write("2:NaN,NaN,NaN;0.6,0.5,1.0:NaN,NaN,NaN\n")      # ball missing
            handle.write("3:0.5,0.5,1.0;NaN,NaN,NaN:0.9,0.5,0.0\n")      # no away player tracked
        f = ph.extract_features(path, team_of_track, 4, PITCH)
    assert f["d_home"][0] == 0.0 and abs(f["d_away"][0] - 0.1 * PITCH[0]) < 1e-3
    assert np.isnan(f["ball_xn"][1]) and f["w_home"][1] == 0.0
    assert np.isnan(f["d_away"][2]) and f["ball_speed"][2] == 0.0
    assert abs(f["ball_speed"][0] - 3.0 / 3.6) < 1e-6, "raw ball speed is km/h and must be read as m/s"
    assert np.isnan(f["ball_xn"][3])  # frame beyond the file stays missing


# ---------------------------------------------------------------------------------------------
# Synthetic match
# ---------------------------------------------------------------------------------------------

def players():
    """22 static players on a 6 x 4 grid (two cells left empty); home on the left, away on the right."""
    cells = [((col + 0.5) / 6.0, (row + 0.5) / 4.0) for col in range(6) for row in range(4)]
    home = [c for c in cells if c[0] < 0.5][:11]
    away = [c for c in cells if c[0] > 0.5][:11]
    return home, away


def build_match(folder):
    """Write tactical ATD metadata + raw into folder/tactical and return the true label array."""
    home, away = players()
    group = lambda xy: "{:.5f},{:.5f},1.0".format(*xy)  # noqa: E731
    player_field = ";".join(group(p) for p in home + away)

    rng = np.random.RandomState(0)
    truth = np.full(N_FRAMES, "X", dtype="<U1")
    owner_xy = {}            # frame -> ball (x, y, speed) or None
    for half, first, last in half_video_blocks(GAME):
        frame, side = first, 0
        while frame <= last:
            end = min(last, frame + 15 * 25 - 1)               # a 15 s possession
            truth[frame:end + 1] = "H" if side == 0 else "A"
            who = (home if side == 0 else away)[int(rng.randint(11))]
            for t in range(frame, end + 1):
                owner_xy[t] = (who[0] + 0.8 / PITCH[0], who[1], 5.0)
            frame = end + 1
            if frame <= last and rng.rand() < 0.5:             # a 10 s stoppage, ball over the line
                end = min(last, frame + 10 * 25 - 1)
                truth[frame:end + 1] = "D"
                for t in range(frame, end + 1):
                    owner_xy[t] = (0.5, 1.03, 0.0)
                frame = end + 1
            side = 1 - side

    os.makedirs(os.path.join(folder, "tactical"))
    stem = os.path.join(folder, "tactical", "BAR-ATH_tactical")
    with open(stem + "_atd.xml", "w") as handle:
        handle.write("<root><Teams><Team id='T1'><Name>Home FC</Name></Team>"
                     "<Team id='T2'><Name>Away FC</Name></Team></Teams><Players>")
        handle.write("".join("<Player teamId='T1'/>" for _ in home))
        handle.write("".join("<Player teamId='T2'/>" for _ in away))
        handle.write("</Players></root>")
    with open(stem + "_atd.txt", "w") as handle:
        for frame in range(N_FRAMES):
            ball = owner_xy.get(frame)
            # 10 % of the frames lose the ball (a fixed pattern, not random, so the test is stable)
            if ball is None or frame % 10 == 3:
                text = "NaN,NaN,NaN"
            else:
                text = "{:.5f},{:.5f},{:.1f}".format(*ball)
            handle.write("{}:{}:{}\n".format(frame + 1, player_field, text))
    return truth


def test_full_command_line_on_a_synthetic_match():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        truth = build_match(data_dir)
        env = dict(os.environ, MTAH_DATA_DIR=data_dir)
        cmd = [sys.executable, os.path.join(ROOT, "solution", "possession_hmm.py"),
               "--name", "synthetic", "--out", out_dir, "--no-cache"]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
        assert run.returncode == 0, run.stdout + run.stderr
        csv = os.path.join(out_dir, "synthetic.csv")

        check = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), csv],
                               capture_output=True, text=True, cwd=ROOT)
        assert check.returncode == 0, check.stdout + check.stderr

        labels = np.array([line.split(",")[1].strip() for line in open(csv).read().splitlines()[1:]])
        assert len(labels) == N_FRAMES
        graded = truth != "X"
        accuracy = float(np.mean(labels[graded] == truth[graded]))
        assert not np.any(labels[~graded] != "X"), "frames outside the halves must be X"
        assert accuracy > 0.95, "accuracy on the synthetic match was only {:.3f}".format(accuracy)

        spans = ph.possessions(labels, 25)
        true_spans = ph.possessions(truth, 25)
        assert abs(len(spans) - len(true_spans)) <= 2, (len(spans), len(true_spans))
        print("  synthetic match: accuracy {:.3f}, {} possessions found vs {} true".format(
            accuracy, len(spans), len(true_spans)))


def test_make_submission_validates_and_is_byte_identical_across_runs():
    """The one-command runner: file named after the team, validator passes, two runs hash the same."""
    import hashlib
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        build_match(data_dir)
        env = dict(os.environ, MTAH_DATA_DIR=data_dir)
        script = os.path.join(ROOT, "solution", "make_submission.py")
        digests = []
        for attempt in range(2):
            cmd = [sys.executable, script, "--team", "Team_X", "--out", out_dir, "--no-cache"]
            run = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
            assert run.returncode == 0, run.stdout + run.stderr
            assert "sha256" in run.stdout
            with open(os.path.join(out_dir, "Team_X.csv"), "rb") as handle:
                digests.append(hashlib.sha256(handle.read()).hexdigest())
        assert digests[0] == digests[1], "two runs on the same data produced different files"

        bad = subprocess.run([sys.executable, script, "--team", "../evil"], env=env,
                             capture_output=True, text=True, cwd=ROOT)
        assert bad.returncode == 2, "an unsafe team name must be refused"


def test_output_is_deterministic():
    f = {name: np.full(500, np.nan, dtype=np.float32) for name in ph.FEATURE_NAMES}
    f["w_home"] = np.zeros(500, dtype=np.float32)
    f["w_away"] = np.zeros(500, dtype=np.float32)
    f["ball_xn"][:] = 0.4
    f["ball_yn"][:] = 0.5
    f["ball_speed"][:] = 4.0
    f["d_home"][:], f["d_away"][:] = 1.0, 9.0
    f["w_home"][:], f["w_away"][:] = 0.6, 0.0
    scores = ph.emission_scores(f, 25)
    penalty = ph.transition_penalties(25)
    assert np.array_equal(ph.viterbi(scores, penalty), ph.viterbi(scores, penalty))
    assert set(ph.viterbi(scores, penalty)) == {ph.H}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok   {}".format(name))
    print("all checks passed")
