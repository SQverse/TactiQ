# Technical Document — More than a Hack 2027, Phase 1

**Team:** TactiQ
**Members:** Abid Syauqi Rabbani, Bagas Aryasatya Pratama, Salman Alfarisy
**Code:** <https://github.com/SQverse/TactiQ>
**Submission file:** `TactiQ.csv` (SHA-256 `ee25243fdab86bdcd28b4af106140331a995c6415b3984e91db10c1c6d684498`)
**Date:** 16 November 2026

---

## 1. Approach

We label every frame `H`, `A` or `D` by decoding each half with a **three-state Viterbi model** over
soft, per-frame evidence computed from the automatic tracking (ATD) alone.

**Why this rather than the baseline.** The organisers' baseline decides each frame independently
("who is nearest the ball?") and repairs the result with a majority vote. The ranking metric
(`spellF1`) scores whole possessions, so a possession should be built from evidence that persists,
not merely survive smoothing. A decoder that charges a fixed price for every change of state does
that: a change is accepted only when the evidence for it outweighs the price, and it is placed where
that evidence begins, because the whole half is decoded at once rather than through a sliding window.

### Per-frame evidence

- **Who controls the ball.** For each team, the sum of `exp(-d / tau)` over its tracked players,
  where `d` is the distance to the ball in metres (`tau` = 2.5 m). The clipped log-ratio of the two
  sums, scaled to [-1, 1], is the team signal. It is multiplied by a confidence that falls linearly
  from 1 to 0 as the nearest player of either team goes from 3 m to 8 m away: a ball far from
  everybody (in the air, or lost by the tracker) says nothing about who has it.
- **Dead ball.** Evidence for `D` is added when the ball is tracked outside the pitch lines, when it
  has been motionless (below 0.5 m/s) for at least 3 s, and, weakly, while the ball has been missing
  for more than 4 s (full weight at 8 s).
- **Missing ball.** With no ball there is no team evidence, so the decoder keeps its current state.
  A short occlusion inside a possession therefore does not split it.

### Decoding

Viterbi over {`H`, `A`, `D`}, per half, maximising the summed evidence minus the cost of each change
of state. `H` ↔ `A` costs the equivalent of 2 s of unambiguous evidence; entering or leaving `D`
costs 2.5 s. Frames outside the two graded halves are written `X`, which is never scored. The
decoder never abstains inside a half.

## 2. Assets used

Only the tactical tracking data (ATD) is used; nothing else from the provided bundle enters the
method.

| Asset | Used | How |
|---|---|---|
| Tactical ATD (tracking with ball): `atd.xml` and `atd.txt` | Yes | The only input: player and ball positions, ball speed, and the team of each track slot |
| Smart Tagging | No | A possible source of restart evidence for `D`; not implemented |
| Homography | No | The pitch coordinates in the ATD were sufficient |
| Tactical and broadcast video | No | Not used by the algorithm |
| Broadcast ATD and homography | No | Not synchronised to the reference timeline |

**Notes on the ATD.** The first team in the metadata is the home side (Barcelona, as `DATA.md`
states). The raw speed column is in km/h (`speed_kph` in the metadata) and is converted to m/s.
Coordinates are normalised and scaled per axis to metres (105.3 m × 68.0 m, from `games.py`).

**Exploratory analysis (not used by the method).** The repository's `research/` folder contains a
separate analysis of the ATD: ball outages and what other signals remain available during them,
player-track continuity and coverage, and a per-frame feature table (nearest and second-nearest
player per team, rolling statistics over 1 s and 3 s, observation age). None of it feeds the
submitted file; it is the groundwork for the next version (section 5).

## 3. Training data

**None.** The method has no learned component, and no labelled data of any kind is used: none from
the provided match and none from any other. Nothing was trained or fitted.

Every value in the parameter table of the code (`PARAMS` in `solution/possession_hmm.py`) is a
physical or footballing prior (distances in metres, durations in seconds, one speed in m/s), set
before the method was run on the provided match. None was fitted, tuned or calibrated on the public
ground truth, on the full-match ground truth, on the video, or on any other data about this match.
The public five minutes were used only to check that the pipeline runs and that the file is well
formed.

## 4. Possession rules implemented

- Only `H` and `A` count as possession. `D` is a stoppage, never a possession.
- **Loose balls.** Possession stays with the last team while the ball is loose, in the air or
  untracked: no team evidence moves the decoder, and a change costs 2 s of evidence.
- **Challenges and touches.** A change of possession requires the opposite team to show sustained
  control; a few frames of proximity are not enough. This is our proxy for "controlled" possession:
  a touch or a deflection does not flip the state, a sustained spell near the ball does.
- **Dead ball** is inferred from the ball leaving the pitch, from sustained stillness, and from long
  tracking gaps.
- **Restarts** are not modelled separately: the model leaves `D` when the ball moves and a team
  controls it again.
- Dead-ball ownership is not asked for and is not attempted.

## 5. Limitations and next steps

- **Dead ball is the weak class.** Inferring `D` from position and speed alone misses stoppages
  where the ball stays on the pitch, and reacts slowly after a restart. Conversely, a long tracking
  gap pushes the model into `D` and, with no team evidence while the ball stays missing, it can stay
  there for a long time.
- **Aerial balls and crowded boxes.** The ball is untracked on roughly 41% of graded frames, often
  exactly when possession changes. The decoder then holds its previous state, which delays or misses
  some hand-overs. As a result the model produces fewer and longer possessions than a real match
  contains.
- **Collective proximity.** The team signal sums over all nearby players, so a team that crowds the
  ball area is favoured even when it does not control the ball.
- **Tracklets, not players.** The 46 track slots swap and vanish, so the method uses only collective
  proximity, never a player's identity or history.
- **Fixed priors.** Because nothing was tuned on this match, the thresholds are not optimal for it.
  With external labelled matches we would tune the distance scales and switch penalties there.
- **Next steps.** (1) Use the outage analysis in `research/` — signals that survive while the ball
  is missing, such as player movement and the observation-age features — as team evidence during
  gaps. (2) Normalise the team signal so it is not a head count. (3) Use Smart Tagging set pieces as
  evidence for `D` and for the team that restarts.

## 6. Reproduction

With the data bundle in `data/` (or `MTAH_DATA_DIR` pointing at it) and Python 3.8 or newer, run
from the repository root:

```bash
pip install -r requirements-solution.txt
python tests/test_possession_hmm.py          # optional; synthetic data only
python solution/make_submission.py --team "TactiQ"
```

This writes `submission/TactiQ.csv`, validates it with the organisers' validator and prints its
SHA-256. On Windows, `jalankan.bat` runs the same steps.

The method has no randomness: two runs on the same data and the same dependency versions give an
identical file, which the repository's tests check. The submitted file was generated on Windows with
Python 3.10, numpy 2.2.6 and pandas 2.3.3 (SHA-256 above), and was regenerated byte-identically from
the raw ATD, with the feature cache disabled (`--no-cache`), on Windows with Python 3.13.5,
numpy 2.2.5 and pandas 2.2.3. Full instructions are in `SOLUTION.md`.

## 7. Development note

The code was written with the help of an AI coding assistant (Claude, Anthropic). It is plain numpy,
fully contained in the repository, and can be run and inspected without any external service.
