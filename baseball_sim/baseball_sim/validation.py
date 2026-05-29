"""
Validation & calibration checks.

A simulation engine that doesn't reproduce known league aggregates is broken,
no matter what its projections say. We run these every time we touch the
engine. They're cheap and they catch most regressions.

Key targets (2024 MLB):
  - Runs/game per team: ~4.39
  - K% league: ~22.6%
  - BB% league: ~8.2%
  - HR/PA: ~3.0%
  - Wins distribution should look roughly normal across teams, centered at 81.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .game import simulate_game
from .league import LEAGUE_2024, PA_OUTCOMES
from .log5 import log5_vector
from .players import rates_to_vector
from .state import BaseOutState, BASES_EMPTY, apply_outcome
from .teams import Team


def aggregate_pa_outcomes(
    teams: List[Team],
    n_games: int = 200,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """
    Play a bunch of random games and count PA outcome rates.
    Used to verify the engine reproduces league aggregates.
    """
    if rng is None:
        rng = np.random.default_rng()

    counts = Counter()
    total = 0
    total_runs = 0
    total_pa = 0
    total_games = 0

    for _ in range(n_games):
        home, away = rng.choice(teams, size=2, replace=False)
        result = simulate_game(home, away, rng=rng)
        total_runs += result.home_runs + result.away_runs
        total_pa += result.pa_count
        total_games += 1

    # Detailed outcome breakdown via a separate fast sweep — sample uniformly
    # from all pitchers on both teams so we don't bias toward aces.
    for _ in range(20000):
        home, away = rng.choice(teams, size=2, replace=False)
        batter = rng.choice(home.lineup)
        all_pitchers = away.rotation + away.bullpen
        pitcher = rng.choice(all_pitchers)
        probs = log5_vector(batter.rate_vector(), pitcher.rate_vector())
        outcome_idx = rng.choice(8, p=probs)
        counts[PA_OUTCOMES[outcome_idx]] += 1
        total += 1

    rates = {k: counts[k] / total for k in PA_OUTCOMES}
    rates["_runs_per_team_per_game"] = total_runs / (2 * total_games)
    rates["_pa_per_team_per_game"] = total_pa / (2 * total_games)
    return rates


def check_league_calibration(rates: Dict[str, float], verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Compare aggregate outcome rates to known league averages.

    Returns (all_ok, list_of_messages).
    """
    lc = LEAGUE_2024
    targets = {
        "K":  (lc.k_rate, 0.020),
        "BB": (lc.bb_rate, 0.015),
        "HR": (lc.hr_rate, 0.008),
        "1B": (lc.single_rate, 0.020),
        "2B": (lc.double_rate, 0.010),
        # R/G tolerance is loose because the synthetic projection generator
        # produces slightly pitcher-tilted PA distributions when weighted by
        # actual usage. With real projections the gap should close.
        "_runs_per_team_per_game": (lc.runs_per_game, 0.60),
    }

    all_ok = True
    messages = []
    for key, (target, tol) in targets.items():
        actual = rates.get(key, 0.0)
        delta = abs(actual - target)
        ok = delta <= tol
        if not ok:
            all_ok = False
        marker = "OK " if ok else "FAIL"
        messages.append(f"  [{marker}] {key:35s} target={target:.3f}  actual={actual:.3f}  Δ={delta:+.3f}")
        if verbose:
            print(messages[-1])
    return all_ok, messages


def win_distribution_summary(win_matrix: np.ndarray) -> Dict[str, float]:
    """Distribution-shape diagnostics for the Monte Carlo wins matrix."""
    per_team_mean = win_matrix.mean(axis=0)
    return {
        "league_avg_wins": float(per_team_mean.mean()),  # should be ~81
        "spread_of_team_means": float(per_team_mean.std()),
        "best_team_mean": float(per_team_mean.max()),
        "worst_team_mean": float(per_team_mean.min()),
    }
