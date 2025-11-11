# Musician Solver

An optimized Haskell implementation of the *Musician* deduction game featuring both composer and performer roles. The performer uses a hybrid information-theoretic strategy with two-ply lookahead to identify any valid three-pitch chord drawn from 21 possible pitches.

## Game Overview

- **Pitches:** 21 combinations of notes `A..G` and octaves `1..3`.
- **Chord:** Any ordered set of three distinct pitches (1,330 total targets).
- **Feedback tuple:** `(correct pitches, correct notes, correct octaves)` with exact matches removed before counting partial matches.

## Key Modules

- `Proj2.hs` – Core solver exported as a `Proj2` module with:
- `toPitch` / `Pitch` parsing & printing helpers.
- `feedback` kernel computing answer tuples via strict note/octave histograms.
- `initialGuess` returning the canonical opener `[(A,1),(D,2),(G,3)]`.
- `nextGuess` orchestrating candidate pruning and strategy hand-off.

## Guessing Strategy

1. **Candidate pruning:** Maintain remaining legal targets by filtering on exact feedback consistency.
2. **Hybrid scoring:** For each candidate guess `g`, build a histogram of possible feedback outcomes against the current candidate set `S` and score via:
   - Expected remaining candidates `E`.
   - Worst-case bucket size `W`.
   - Information entropy `H`.
   The solver minimizes `(E, W, -H)` lexicographically.
3. **Two-ply lookahead:** When `|S| ≤ 50`, evaluate the top four one-ply guesses more deeply by simulating a best-response guess for each feedback outcome and minimizing the expected size of the subsequent candidate set.

## Performance Summary

Profiling was performed over all 1,330 targets with GHC `-O2`, `-prof`, and `-fprof-auto`.

| Metric                   | Result                       |
| -------------------------| -----------------------------|
| Average guesses per game | **4.27** (median 4, max 7)   |
| Total runtime            | **252 s** (~189 ms per game) |
| Total allocations        | **1,095 GB** (profiling run) |
| Max residency            | **683 KB**                   |
| GC overhead              | **0.0%** (99.7% productivity)|

### Hotspots

- `feedback` and sub-components (`nOnly`, `oOnly`, `exacts`) dominate (~77% of time), as expected for the solver’s core loop.
- Strict folds, bang patterns, and `Map.Strict` ensure low allocation pressure.

### Future Work

- Memoize feedback results for repeated pair evaluations (estimated 10–20% speedup).
- Replace list-based histograms with unboxed arrays for tighter loops.
- Parallelize histogram scoring for multi-core speedups.
- Tune the two-ply activation threshold adaptively.

## Building & Running

```bash
ghc -O2 Proj2.hs
./Proj2
```

### Profiling Commands

```bash
ghc -O2 -prof -fprof-auto -rtsopts Proj2.hs -o Musician
./Musician +RTS -p -s -RTS
```

## Repository Files

- `Proj2.hs` – Solver implementation.
- `PROFILING_REPORT.md` – Full profiling narrative with tables, methodology, and recommendations.
- `README.md` – Project overview (this document).

## Credits

Authored by Quynh Anh Ha.


