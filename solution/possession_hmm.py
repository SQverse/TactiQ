"""Possession labelling with a three-state Viterbi decoder (H / A / D).

The baseline asks one question per frame ("who is nearest the ball?") and then repairs the answer
with a majority vote. This does the opposite: it scores each frame softly, then decodes the whole
half at once with a penalty for every change of state, so a possession is something the model
*builds* rather than something that survives smoothing.

    pip install -r requirements-solution.txt
    python solution/possession_hmm.py --name <team_name>
    python evaluation/validate_submission.py evaluation/results/<team_name>.csv
    python evaluation/run_evaluation.py

numpy only. The output is deterministic: no randomness anywhere, so the same data gives the same
file, byte for byte (RULES.md, rule 3).

How it works
------------
1. Features, per video frame, from the ATD alone: where the ball is, the distance to the nearest
   home and away player, a soft "ownership" weight per team, and whether the ball is off the pitch,
   motionless, or missing.
2. Emission scores, one per state per frame (higher = more likely):
     H / A  the team whose players are close to the ball, weighted by how close anyone is at all
            (a ball in the air, far from everybody, says nothing and scores zero)
     D      the ball is off the pitch, or has not moved for seconds, or has been missing for a long
            stretch
3. Viterbi over {H, A, D}, per half. Changing H <-> A or entering / leaving D costs a fixed penalty,
   expressed as "seconds of unambiguous evidence needed to change". That penalty sets the shortest
   possession the model can detect; it does not delay the boundary, because decoding looks at the
   whole half and puts a change where the evidence for it actually begins.
4. Frames outside the two halves are written as X; they are never graded.

Parameters
----------
Every number in PARAMS is a physical or footballing prior (metres, seconds), chosen before looking
at any label. NONE was tuned on the public ground truth or on any hand label of this match --
RULES.md forbids it and EVALUATION.md shows why it misleads. Change them only for a reason you can
state in the technical document.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import (AWAY, FIELD_SEP, GROUP_SEP, HOME, MISSING,  # noqa: E402
                      team_names, team_of_each_track, write_submission)

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_DIR = os.path.join(ROOT, ".cache")

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = dict(
    # --- who is controlling the ball ---------------------------------------------------------
    tau_m=2.5,          # distance scale of a player's claim on the ball: weight = exp(-d / tau)
    near_m=3.0,         # nearest player this close (or closer): full confidence in the owner
    far_m=8.0,          # nearest player this far (or farther): ball is loose / in the air, no evidence
    logodds_clip=1.5,   # log-odds of home vs away weight is clipped to +/- this, then scaled to +/-1
    team_weight=1.0,    # H and A get +/- team_weight / 2, so a clear call is worth 1.0 per frame

    # --- dead ball -----------------------------------------------------------------------------
    oob_bonus=1.5,      # ball tracked outside the pitch lines: very likely dead
    oob_margin=0.002,   # normalised units (~0.2 m); the ball on the line is still in play
    still_speed=0.5,    # m/s; below this the ball counts as motionless
    still_seconds=3.0,  # ...but only after it has been motionless this long
    still_bonus=0.8,
    gap_start_s=4.0,    # ball missing for less than this says nothing about a stoppage
    gap_end_s=8.0,      # ...and for more than this, gap_bonus applies in full
    gap_bonus=0.1,

    # --- how reluctant the model is to change its mind -----------------------------------------
    ha_switch_s=2.0,    # H <-> A: seconds of unambiguous evidence needed
    dead_switch_s=2.5,  # live <-> D
)


# ---------------------------------------------------------------------------------------------
# 1. Features
# ---------------------------------------------------------------------------------------------

FEATURE_NAMES = ("ball_xn", "ball_yn", "ball_speed", "d_home", "d_away", "w_home", "w_away")


def _fields(group):
    """The first three comma-separated numbers of an ATD group, as floats."""
    return [float(v) for v in group.split(",")[:3]]


FEATURE_VERSION = 2  # bump when extract_features changes, so a stale cache is never reused


def extract_features(raw_path, team_of_track, n_frames, pitch, tau_m=PARAMS["tau_m"]):
    """Per-video-frame features from the raw ATD. NaN (or 0 for weights) where the ball is missing.
    ball_speed is in m/s (the raw file stores km/h).

    Same parsing as starter/baseline.py: a line is "<frame_id>:<players>:<ball>", frame_id counts
    from 1 and video frames from 0, coordinates are normalised and scaled to metres per axis.
    """
    length, width = pitch
    nan = lambda: np.full(n_frames, np.nan, dtype=np.float32)  # noqa: E731
    ball_xn, ball_yn, ball_speed, d_home, d_away = nan(), nan(), nan(), nan(), nan()
    w_home = np.zeros(n_frames, dtype=np.float32)
    w_away = np.zeros(n_frames, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue

            bx, by, bspeed = _fields(parts[2])
            # The raw file carries speed in km/h (the metadata's "speed_kph" channel); kloppy's
            # m/s in DATA.md is a conversion it applies on load. This module reads the raw file,
            # so convert here, once, and every threshold below is in m/s.
            ball_xn[frame], ball_yn[frame], ball_speed[frame] = bx, by, bspeed / 3.6
            bx_m, by_m = bx * length, by * width

            dh = da = math.inf
            wh = wa = 0.0
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING:
                    continue
                px, py, _ = _fields(group)
                dist = math.hypot(px * length - bx_m, py * width - by_m)
                weight = math.exp(-dist / tau_m)
                if team_of_track[slot] == HOME:
                    wh += weight
                    dh = min(dh, dist)
                else:
                    wa += weight
                    da = min(da, dist)
            d_home[frame] = dh if dh != math.inf else np.nan
            d_away[frame] = da if da != math.inf else np.nan
            w_home[frame], w_away[frame] = wh, wa

    return dict(zip(FEATURE_NAMES, (ball_xn, ball_yn, ball_speed, d_home, d_away, w_home, w_away)))


def load_or_extract(raw_path, team_of_track, n_frames, pitch, use_cache=True):
    """extract_features, cached in .cache/ (git-ignored) keyed on the raw file's size and mtime.

    The cache holds frame-level data derived from the match, so it must never be committed or
    published -- RULES.md. .gitignore excludes it.
    """
    stat = os.stat(raw_path)
    key = np.array([stat.st_size, int(stat.st_mtime), n_frames, PARAMS["tau_m"], FEATURE_VERSION],
                   dtype=np.float64)
    path = os.path.join(CACHE_DIR, "features.npz")
    if use_cache and os.path.isfile(path):
        cached = np.load(path)
        if np.array_equal(cached["key"], key):
            print("features: loaded from cache")
            return {name: cached[name] for name in FEATURE_NAMES}

    print("features: reading {} ...".format(os.path.basename(raw_path)))
    feats = extract_features(raw_path, team_of_track, n_frames, pitch)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, key=key, **feats)
    return feats


def slice_features(feats, first, last):
    """The features of video frames first..last inclusive."""
    return {name: values[first:last + 1] for name, values in feats.items()}


# ---------------------------------------------------------------------------------------------
# 2. Emission scores
# ---------------------------------------------------------------------------------------------

def run_lengths(mask):
    """For every True element, the length of the maximal run of True it sits in; 0 elsewhere."""
    out = np.zeros(len(mask), dtype=np.int64)
    if len(mask) == 0:
        return out
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:end] = end - start
    return out


def emission_scores(f, fps, params=PARAMS):
    """(n, 3) array of per-frame scores for H, A, D. Higher means more likely; only differences
    between the three columns on one frame matter."""
    n = len(f["ball_xn"])
    visible = ~np.isnan(f["ball_xn"])
    scores = np.zeros((n, 3), dtype=np.float64)

    # -- who has it ----------------------------------------------------------------------------
    log_odds = np.log((f["w_home"] + 1e-6) / (f["w_away"] + 1e-6))
    log_odds = np.clip(log_odds, -params["logodds_clip"], params["logodds_clip"]) / params["logodds_clip"]

    nearest = np.fmin(f["d_home"], f["d_away"])  # fmin ignores NaN unless both are NaN
    span = params["far_m"] - params["near_m"]
    confidence = np.clip((params["far_m"] - np.nan_to_num(nearest, nan=np.inf)) / span, 0.0, 1.0)

    team = np.where(visible, confidence * log_odds, 0.0)
    scores[:, H] = 0.5 * params["team_weight"] * team
    scores[:, A] = -0.5 * params["team_weight"] * team

    # -- dead ball -----------------------------------------------------------------------------
    margin = params["oob_margin"]
    off_pitch = visible & ((f["ball_xn"] < -margin) | (f["ball_xn"] > 1 + margin) |
                           (f["ball_yn"] < -margin) | (f["ball_yn"] > 1 + margin))
    scores[off_pitch, D] += params["oob_bonus"]

    slow = visible & (np.nan_to_num(f["ball_speed"], nan=np.inf) < params["still_speed"])
    still = run_lengths(slow) >= params["still_seconds"] * fps
    scores[still, D] += params["still_bonus"]

    gap_s = run_lengths(~visible) / float(fps)
    ramp = np.clip((gap_s - params["gap_start_s"]) / (params["gap_end_s"] - params["gap_start_s"]), 0.0, 1.0)
    scores[:, D] += np.where(~visible, params["gap_bonus"] * ramp, 0.0)
    return scores


# ---------------------------------------------------------------------------------------------
# 3. Decoding
# ---------------------------------------------------------------------------------------------

def transition_penalties(fps, params=PARAMS):
    """3x3 cost of moving from state i to state j. Staying is free. One unit of cost is one frame
    of a perfectly unambiguous H-vs-A call, hence 'seconds * fps'."""
    ha = params["ha_switch_s"] * fps
    dead = params["dead_switch_s"] * fps
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = ha
    cost[H, D] = cost[A, D] = cost[D, H] = cost[D, A] = dead
    return cost


def viterbi(scores, penalty):
    """Most likely state sequence: maximise sum(scores) - sum(penalty of each change).

    scores is (n, k), penalty is (k, k) with penalty[i, j] the cost of going from i to j. Ties go to
    the lower state index, so the result is fully deterministic.
    """
    n, k = scores.shape
    best = scores[0].copy()
    back = np.zeros((n, k), dtype=np.int8)
    columns = np.arange(k)
    for t in range(1, n):
        candidate = best[:, None] - penalty        # [from, to]
        origin = candidate.argmax(axis=0)
        back[t] = origin
        best = candidate[origin, columns] + scores[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = best.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def label_match(feats, n_frames, fps, blocks, params=PARAMS):
    """Full-length array of H / A / D, with X outside the graded halves."""
    labels = np.full(n_frames, "X", dtype="<U1")
    penalty = transition_penalties(fps, params)
    for half, first, last in blocks:
        part = slice_features(feats, first, last)
        path = viterbi(emission_scores(part, fps, params), penalty)
        labels[first:last + 1] = LABELS[path]
    return labels


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------

def shown_path(path, start=ROOT):
    """Path relative to start for printing, or the path itself when that is impossible.

    On Windows os.path.relpath raises ValueError when the two paths are on different drives
    (for example a file in a temporary folder on C: and a repository on D:).
    """
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def possessions(labels, fps):
    """[(label, start, length)] of maximal H or A runs, in frames. Dead ball is never a possession."""
    out, start = [], None
    for i in range(len(labels) + 1):
        current = labels[i] if i < len(labels) else None
        if start is not None and current != labels[start]:
            out.append((labels[start], start, i - start))
            start = None
        if start is None and current in ("H", "A"):
            start = i
    return out


def report(labels, fps, blocks):
    """Sanity numbers a football analyst can judge without any ground truth: possession counts and
    durations per half. A match has dozens to a couple of hundred, mean around 10-20 s."""
    for half, first, last in blocks:
        part = labels[first:last + 1]
        runs = possessions(part, fps)
        seconds = [length / float(fps) for _, _, length in runs]
        share = {k: 100.0 * np.mean(part == k) for k in "HAD"}
        print("  half {}: {:>3} possessions, mean {:>5.1f} s, median {:>5.1f} s | H {:.0f}%  A {:.0f}%  D {:.0f}%".format(
            half, len(runs), np.mean(seconds) if seconds else 0.0, np.median(seconds) if seconds else 0.0,
            share["H"], share["A"], share["D"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="possession_hmm",
                        help="file name stem; use your team name for the submission (<team_name>.csv)")
    parser.add_argument("--out", default=RESULTS_DIR, help="folder to write into")
    parser.add_argument("--no-cache", action="store_true", help="re-read the ATD instead of .cache/")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    fps = GAMES[GAME]["fps"]
    blocks = half_video_blocks(GAME)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    feats = load_or_extract(raw_path, team_of_each_track(meta_path), n_frames,
                            GAMES[GAME]["pitch"], use_cache=not args.no_cache)

    labels = label_match(feats, n_frames, fps, blocks)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote {}".format(shown_path(path)))
    report(labels, fps, blocks)
    print("\nNext:  python evaluation/validate_submission.py {}".format(shown_path(path)))
    print("       python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
