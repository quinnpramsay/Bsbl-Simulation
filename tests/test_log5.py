"""
Unit tests for the log5 engine and state machine.

Run with: python -m pytest tests/ -v
Or directly: python tests/test_log5.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from baseball_sim.league import LEAGUE_2024, PA_OUTCOMES, league_rates_as_dict
from baseball_sim.log5 import log5_vector
from baseball_sim.players import Batter, Pitcher, rates_to_vector
from baseball_sim.state import (
    BASES_EMPTY,
    BaseOutState,
    apply_outcome,
    apply_home_run,
    apply_triple,
    apply_walk_or_hbp,
)


def test_log5_average_vs_average_returns_league():
    """League-avg batter vs league-avg pitcher should give league rates."""
    league_vec = rates_to_vector(league_rates_as_dict())
    result = log5_vector(league_vec, league_vec, league_vec)
    assert np.allclose(result, league_vec, atol=0.001), \
        f"Expected league rates, got {result}"


def test_log5_output_sums_to_one():
    """log5 output must be a valid probability distribution."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        b = rng.dirichlet(np.ones(8))
        p = rng.dirichlet(np.ones(8))
        result = log5_vector(b, p)
        assert abs(result.sum() - 1.0) < 1e-9
        assert (result >= 0).all()


def test_log5_high_k_batter_vs_high_k_pitcher_increases_k():
    """A high-K batter facing a high-K pitcher should K more often than league avg."""
    league_vec = rates_to_vector(league_rates_as_dict())
    high_k_b = league_vec.copy()
    high_k_b[0] = 0.35  # K
    high_k_b /= high_k_b.sum()
    high_k_p = league_vec.copy()
    high_k_p[0] = 0.32
    high_k_p /= high_k_p.sum()
    result = log5_vector(high_k_b, high_k_p)
    assert result[0] > league_vec[0] * 1.5, \
        f"Expected K rate > {league_vec[0]*1.5:.3f}, got {result[0]:.3f}"


def test_home_run_clears_bases():
    """HR with bases loaded scores 4 runs and clears bases."""
    state = BaseOutState(bases=(True, True, True), outs=1)
    new_state, runs = apply_home_run(state)
    assert runs == 4
    assert new_state.bases == BASES_EMPTY
    assert new_state.outs == 1  # outs unchanged


def test_triple_scores_all_runners():
    """Triple with two on scores both runners, batter to 3rd."""
    state = BaseOutState(bases=(True, True, False), outs=0)
    new_state, runs = apply_triple(state)
    assert runs == 2
    assert new_state.bases == (False, False, True)


def test_walk_with_bases_loaded_scores_one():
    """Walk forces in one run when bases loaded."""
    state = BaseOutState(bases=(True, True, True), outs=2)
    new_state, runs = apply_walk_or_hbp(state)
    assert runs == 1
    assert new_state.bases == (True, True, True)


def test_walk_with_runner_on_first_only():
    """Walk with only 1B occupied: runners on 1st and 2nd."""
    state = BaseOutState(bases=(True, False, False), outs=0)
    new_state, runs = apply_walk_or_hbp(state)
    assert runs == 0
    assert new_state.bases == (True, True, False)


def test_strikeout_just_adds_out():
    """K adds an out, bases unchanged."""
    state = BaseOutState(bases=(False, True, False), outs=1)
    k_idx = PA_OUTCOMES.index("K")
    rng = np.random.default_rng(0)
    new_state, runs = apply_outcome(k_idx, state, rng)
    assert runs == 0
    assert new_state.outs == 2
    assert new_state.bases == (False, True, False)


def test_three_outs_ends_inning():
    """Inning must end when outs reach 3."""
    state = BaseOutState(bases=BASES_EMPTY, outs=2)
    k_idx = PA_OUTCOMES.index("K")
    rng = np.random.default_rng(0)
    new_state, _ = apply_outcome(k_idx, state, rng)
    assert new_state.is_inning_over()


def test_run_expectancy_bases_empty_zero_outs():
    """
    Run expectancy from (empty, 0 outs) using league rates should be ~0.45-0.50.
    Real MLB is 0.481 (Tango 2010-15). We allow a wider tolerance because
    we don't model SB/CS, WP/PB, errors.
    """
    rng = np.random.default_rng(1234)
    probs = np.array([0.226, 0.082, 0.012, 0.030, 0.143, 0.044, 0.004, 0.459])
    probs /= probs.sum()

    total = 0
    n = 20000
    for _ in range(n):
        state = BaseOutState(BASES_EMPTY, 0)
        runs = 0
        while not state.is_inning_over():
            cum = probs.cumsum()
            idx = int(cum.searchsorted(rng.random()))
            if idx >= 8:
                idx = 7
            state, scored = apply_outcome(idx, state, rng)
            runs += scored
        total += runs
    re = total / n
    assert 0.40 < re < 0.55, f"RE(empty,0) = {re:.3f}, expected ~0.481"


def test_simulate_game_produces_winner():
    """A simulated game must end with a clear winner."""
    from baseball_sim import generate_all_teams, simulate_game

    teams = generate_all_teams(seed=1)
    rng = np.random.default_rng(7)
    result = simulate_game(teams[0], teams[1], rng=rng)
    assert result.home_runs != result.away_runs
    assert result.home_won == (result.home_runs > result.away_runs)
    assert result.innings >= 9
    assert result.pa_count > 50


def test_season_wins_sum_to_total_games():
    """Wins + losses across all teams must equal 2 * total games."""
    from baseball_sim import generate_all_teams, simulate_season

    teams = generate_all_teams(seed=2)
    rng = np.random.default_rng(99)
    # Use just a few teams for speed
    season = simulate_season(teams[:6], rng=rng)
    total_w = sum(season.wins.values())
    total_l = sum(season.losses.values())
    assert total_w == total_l
    assert total_w + total_l == 2 * season.total_games


if __name__ == "__main__":
    # Simple manual runner so this works without pytest installed
    import inspect
    tests = [
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)
