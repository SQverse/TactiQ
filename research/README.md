# research/ — exploratory analysis of the tracking data

These scripts are a team member's exploration of the automatic tracking (ATD), written alongside the
solution. **None of them is used by `solution/`, and none of them affects `submission/TactiQ.csv`.**
They are kept here as the record of what we looked at, and as the starting point for the next
version of the model (see "Limitations and next steps" in the technical document).

None of them reads a ground truth, public or private.

## What is here

| prefix | what it does |
|---|---|
| `build_*` | a per-frame feature table built from the ATD, in stages (below) |
| `analyze_*` | how the tracking behaves: ball outages, player-track continuity and coverage, spatial structure, movement while the ball is missing |
| `audit_*` | checks on the feature tables: row count (199,319), frame order, duplicates, coverage, readiness |
| `inspect_*`, `inventory_features.py` | quick looks at the raw assets and a categorised list of all features |
| `verify_redundant_features.py`, `remove_verified_redundant_features.py` | find exactly duplicated features (correlation >= 0.9999, confirmed value by value) and drop them |

## The feature pipeline

Run from the repository root (paths are relative to it), with the match data in `data/`:

```bash
python research/build_possession_features_v2.py     # -> data/processed/possession_features_v2.csv
python research/build_temporal_features.py          # -> possession_temporal_features.csv
python research/build_context_features.py           # -> possession_context_features.csv
python research/build_context_coverage_features.py  # -> possession_context_final.csv
python research/build_observation_age_features.py   # -> possession_with_observation_age.csv
python research/analyze_feature_dependencies.py      # -> feature_correlations.csv
python research/verify_redundant_features.py         # check duplicates value by value
python research/remove_verified_redundant_features.py  # -> possession_features_clean.csv
```

Per frame: ball position and visibility, player count per team, nearest and second-nearest player
of each team and their distances, the gap between the teams, frame-to-frame changes, rolling mean
and standard deviation over 1 s and 3 s (past frames only), valid-observation counts per window,
and the age of the last valid observation of each signal.

Everything is written to `data/processed/`, which is git-ignored: these tables are derived from
the match data and must not be published.

## Notes

- `inspect_smart_tagging.py` refers to `evaluation/results/baseline_smooth115.csv`, a baseline
  variant from a window sweep that is not committed; regenerate it with a ±115-frame window in
  `starter/baseline.py` if you need it.
- Comments in the scripts are in Indonesian.
