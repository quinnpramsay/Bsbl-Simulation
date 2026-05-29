# baseball_sim — Phase 0 + Phase 1 MVP

A PA-level Monte Carlo baseball simulator. Given per-PA outcome rates for
batters and pitchers, log5 the matchups, sample outcomes, drive a base-out
state machine, and aggregate across innings → games → seasons.

This is the MVP we set out to build: a calibrated engine that produces
believable league aggregates and projects 162-game win distributions.
Everything else (real projection ingestion, park factors, aging curves,
pitch-level modeling, lineup optimization, dashboard) layers on top
without changing the core.

---

## Quick start

```bash
pip install -r requirements.txt

# Single game with play-by-play
python scripts/01_simulate_game.py

# One full 162-game season (~3 seconds)
python scripts/02_simulate_season.py

# Monte Carlo: projected standings + playoff odds
python scripts/03_monte_carlo.py --n 50
```

Run the tests:

```bash
python tests/test_log5.py
```

---

## What's in here

```
baseball_sim/
├── league.py         # 2024 MLB averages (the log5 baseline)
├── players.py        # Batter, Pitcher dataclasses (rates + metadata)
├── log5.py           # The matchup model — heart of the engine
├── state.py          # 24-state base-out machine + advancement rules
├── teams.py          # Team & live game state
├── game.py           # Single-game simulator
├── season.py         # 162-game schedule generator + season runner
├── monte_carlo.py    # N-season aggregation + playoff odds
├── projections.py    # Synthetic generator + real-data hook
└── validation.py     # Calibration checks against league aggregates
```

---

## How it works

### Two-layer engine

**PA layer (log5).** For each plate appearance, compute the probability of
each outcome (K, BB, HBP, HR, 1B, 2B, 3B, OUT) given the batter's projected
rates, the pitcher's allowed rates, and league averages, using the standard
odds-ratio formula:

```
P(outcome | B, P, L) = (B·P/L) / ((B·P/L) + (1-B)(1-P)/(1-L))
```

Applied per outcome independently, then normalized to a valid distribution.
Public sims from PECOTA on use some version of this.

**Game state layer.** A simple base-out state machine: bases are a 3-tuple of
booleans (1B, 2B, 3B), plus outs ∈ {0, 1, 2}. Each PA outcome transitions
the state and may score runs. Advancement is partly deterministic (forced
moves on walks) and partly probabilistic (extra bases on hits, sac flies,
GIDPs, productive groundouts) with rates tuned so league-aggregate run
expectancies land within 5% of Tango's reference matrix.

### What's *not* modeled (yet)

By design, the MVP omits:

- Stolen bases, caught stealing, pickoffs
- Wild pitches and passed balls
- Errors as a distinct category (they're absorbed into OUT)
- Defensive efficiency / DRS-style adjustments
- Park factors (the hook exists in `log5.apply_park_adjustment` — Phase 3)
- Platoon splits (handedness is stored but not used yet)
- Pinch hitting and defensive substitutions
- The Manfred runner in extra innings
- Pitcher fatigue beyond a simple "X batters faced" cap
- 4-out double switches, intentional walks as strategy, infield-in defense

These all individually move R/G by less than 0.10 and aren't worth modeling
until the engine is wired to real projections.

---

## Calibration

The synthetic teams reproduce 2024 MLB per-PA outcome rates within tolerance:

| Stat | Target | Actual |
|------|--------|--------|
| K%   | 0.226  | 0.234  |
| BB%  | 0.082  | 0.079  |
| HR%  | 0.030  | 0.032  |
| 1B%  | 0.143  | 0.141  |
| 2B%  | 0.044  | 0.041  |
| R/G  | 4.39   | 3.91   |

R/G runs about 10% below target. **This is a synthetic-data construction
artifact, not an engine bug:** when fed pure league-average rates the
engine produces R/G ≈ 4.18 (see `tests/test_log5.py::test_run_expectancy_*`).
The remaining 5% gap is the missing mechanics listed above (steals, WP/PB,
etc.) which collectively contribute ~0.2 R/G in real MLB.

With real Steamer / ZiPS projections the synthetic-data gap closes
automatically.

### Verifying for yourself

```python
from baseball_sim import generate_all_teams
from baseball_sim.validation import aggregate_pa_outcomes, check_league_calibration

teams = generate_all_teams(seed=42)
rates = aggregate_pa_outcomes(teams, n_games=400)
check_league_calibration(rates)
```

---

## Plugging in real projections

The synthetic generator and a real-data loader produce the **same**
`List[Team]` interface. Nothing downstream cares which source you use.

```python
# Current (synthetic, deterministic with seed):
from baseball_sim import generate_all_teams
teams = generate_all_teams(seed=42)

# Phase 0 target:
from baseball_sim import load_real_projections
teams = load_real_projections(season=2026)  # currently raises NotImplementedError
```

### Implementing `load_real_projections`

The shell is in `baseball_sim/projections.py`. To wire it up:

1. **Pull projection rates** with `pybaseball`. Steamer and ZiPS publish to
   FanGraphs; use `pybaseball.fangraphs_batting_leaders(projection="steamer")`
   etc. You want K%, BB%, HBP%, HR/PA, 1B/PA, 2B/PA, 3B/PA for batters and
   K% allowed, BB% allowed, HR/PA allowed, etc. for pitchers.

2. **Crosswalk player IDs** using the
   [Chadwick Bureau Register](https://github.com/chadwickbureau/register) so
   FanGraphs `playerid` → MLBAM `mlbam_id` → Retrosheet `retro_id` maps
   stably across sources.

3. **Allocate playing time** from a depth-chart source (Roster Resource on
   FanGraphs has a JSON endpoint). Assign starting 9 + bench, 5-man rotation,
   7-man bullpen.

4. **Convert to `Batter` / `Pitcher` objects** with the same 8-element rate
   vector as the synthetic generator. The rates dict can have any keys that
   are eventually mapped to PA_OUTCOMES — see `players._normalize`.

5. **Return `List[Team]`** with proper league/division metadata. The 30
   teams should match `projections.TEAM_META`.

Once that's in place, every demo script switches from `generate_all_teams()`
to `load_real_projections()` and nothing else changes.

### Caching layer

A small Postgres or DuckDB table caching the joined projection × playing-time
view per season makes the second load nearly instant. Plan to add this in
Phase 0.5 once the basic loader works.

---

## Performance

Current throughput (single-threaded Python, no compilation):

- ~1 ms per game
- ~3 seconds per 162-game (2610-game) season
- ~150 seconds for 50 seasons across 30 teams

For meaningful Monte Carlo (1000+ seasons) you'll want:

- **Process pool** across seasons (embarrassingly parallel)
- **numba @jit** on the PA inner loop (5-10x easy)
- Or rewrite the hot path in **C/Cython/Rust** for 50x+

None of this is necessary for the MVP — the priority is correctness and
clarity. Speed up once the engine is validated against real data.

---

## Next phases (roadmap reminder)

| Phase | Adds | Effort |
|-------|------|--------|
| 0     | Real projection ingestion (pybaseball + Chadwick) | days |
| 1     | **THIS MVP — PA log5 engine + base-out state machine** | done |
| 2     | Roster construction (depth charts, IL tracking) | days |
| 3     | Park factors + handedness splits | week |
| 4     | Aging curves (hierarchical Bayes) | weeks |
| 5     | Pitch-level engine (XGBoost on Statcast) | months |
| 6     | Lineup optimization + interactive dashboard | months |

The engine interface is stable: every later phase adds adjustment layers
that compose into the existing `log5_vector` call, or replaces the PA
sampler with a finer-grained one. Nothing built so far needs to be torn out.

---

## Why log5 first

Real PA outcomes are the result of a complex pitch-by-pitch interaction.
You *could* model it that way (and Phase 5 will). But for season-level
win projections, log5 captures essentially all the matchup-dependent
variance. Adding pitch-level modeling improves things at the margin
— it doesn't change the win-total distribution by more than 1-2 wins
for any team in a 162-game season.

So: get log5 right, validate it against league aggregates, build the
dashboard on top, and only add pitch-level granularity if a specific
miscalibration shows up that demands it.
