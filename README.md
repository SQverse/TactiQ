# TactiQ — More than a Hack 2027, Phase 1: Ball Possession

Team TactiQ's submission to [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
(Metrica Sports, FC Barcelona and Mobile World Capital Barcelona).

**The task:** for every one of the 199,319 video frames of one LaLiga match, say who has the ball:
`H` (home), `A` (away), `D` (dead ball) or `X` (not answered).

**Our method:** a three-state Viterbi decoder over per-frame evidence from the automatic tracking
(ATD) alone. Each frame is scored softly — how close each team's players are to the ball, whether the
ball is off the pitch, still, or missing — and each half is decoded at once with a fixed cost for
every change of state, so a possession is built from evidence that persists. No training, no
labelled data, numpy only, fully deterministic.

| | |
|---|---|
| Technical document | [docs/TactiQ.md](docs/TactiQ.md) |
| Reproduction | [SOLUTION.md](SOLUTION.md) |
| Submitted file | [submission/TactiQ.csv](submission/TactiQ.csv) — SHA-256 `ee25243f…d684498` |

## Quick start

Put the organisers' data bundle in `data/` (see [data/README.md](data/README.md)), then from the
repository root:

```bash
pip install -r requirements-solution.txt
python tests/test_possession_hmm.py                # self-checks, synthetic data only
python solution/make_submission.py --team "TactiQ"
```

On Windows, double-click `jalankan.bat` to run the same steps.

## Repository layout

```
solution/
  possession_hmm.py        features, emission scores and the Viterbi decoder
  make_submission.py       generate -> validate -> print SHA-256
tests/
  test_possession_hmm.py   unit tests and an end-to-end run on a synthetic match
submission/TactiQ.csv      the submitted file
docs/TactiQ.md             technical document
research/                  exploratory analysis of the tracking data (not used by the solution)
jalankan.bat               Windows one-click runner

evaluation/                organisers' scorer, validator and public 5-minute ground truth
starter/                   organisers' baseline, loaders and notebooks
games.py                   organisers' match constants and frame alignment
data/                      match data goes here; not in git
```

Our own files are `solution/`, `tests/`, `submission/`, `docs/`, `research/`, `jalankan.bat`,
`requirements-solution.txt`, `SOLUTION.md` and this README; `.gitignore` is the organisers' with a
few lines added at the end. Everything else is the organisers' starter kit, unchanged. Its original
README is [STARTER_KIT.md](STARTER_KIT.md); the challenge rules are in [CHALLENGE.md](CHALLENGE.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md) and
[RULES.md](RULES.md).

## Data

The match data is licensed for this competition only and is **not** in this repository. `data/`,
`.cache/` and every intermediate table are git-ignored. The starter notebooks are committed without
outputs; clear their outputs before committing if you run them, because the outputs contain match
data.

## Licence

The organisers' starter kit is MIT-licensed (see [LICENSE](LICENSE)). The match data is not covered
by it.
