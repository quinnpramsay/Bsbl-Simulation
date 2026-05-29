"""
Single game simulator.

Top-level flow:
  for inning in 1..9 (plus extras until tiebreak):
    play_half_inning(away batting)
    if not game_decided: play_half_inning(home batting)

Each half inning loops PAs until 3 outs:
  - get current batter & pitcher
  - compute matchup probabilities via log5
  - sample a PA outcome
  - apply to base-out state, accumulate runs
  - advance batting order, increment pitcher BF
  - check pitcher-change rules

For MVP we ignore: pinch hitting, defensive substitutions, the Manfred runner,
intentional walks as a strategic decision, mound visits, and most strategic
management. None of these meaningfully move season-level win totals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .log5 import log5_vector
from .state import BaseOutState, apply_outcome, BASES_EMPTY
from .teams import LiveTeamState, Team
from .players import Pitcher


@dataclass
class GameResult:
    home_team_id: str
    away_team_id: str
    home_runs: int
    away_runs: int
    innings: int
    home_won: bool
    pa_count: int = 0
    # Optional play-by-play log for debugging / single-game inspection
    log: List[str] = field(default_factory=list)


def _choose_next_pitcher(
    pitching_team: LiveTeamState,
    inning: int,
) -> Pitcher:
    """
    Pick the next pitcher. Heuristics:
      - Starter pitches first inning(s)
      - Replaced when batters_faced exceeds typical_batters
      - Then cycle through bullpen
    """
    team = pitching_team.team

    if pitching_team.current_pitcher is None:
        # First pitcher of the game: starter
        return team.rotation[0] if team.rotation else team.bullpen[0]

    cur = pitching_team.current_pitcher
    bf = pitching_team.batters_faced_by_current_pitcher

    # Pull starter after he hits his stamina cap, or once past inning 5
    if cur.role == "SP":
        if bf >= cur.typical_batters or inning >= 7:
            # Bring in first available unused reliever
            for rp in team.bullpen:
                if rp.player_id not in pitching_team.pitchers_used:
                    return rp
            # Fallback: re-use someone (shouldn't happen in 9 innings)
            return team.bullpen[0]
        return cur

    # Current pitcher is RP: pull after ~5 BF (typical reliever outing)
    if bf >= cur.typical_batters:
        for rp in team.bullpen:
            if rp.player_id not in pitching_team.pitchers_used:
                return rp
        return cur  # ran out of RPs, keep him in
    return cur


def _play_half_inning(
    batting: LiveTeamState,
    pitching: LiveTeamState,
    inning: int,
    rng: np.random.Generator,
    log_pbp: bool = False,
    log: Optional[List[str]] = None,
) -> tuple[int, int]:
    """
    Play one half inning. Returns (runs_scored, pa_count).
    """
    state = BaseOutState(bases=BASES_EMPTY, outs=0)
    runs = 0
    pa_count = 0

    while not state.is_inning_over():
        # Pitcher change check at start of each PA
        next_p = _choose_next_pitcher(pitching, inning)
        if next_p is not pitching.current_pitcher:
            pitching.set_pitcher(next_p)

        batter = batting.current_batter()
        pitcher = pitching.current_pitcher

        # Compute matchup distribution
        probs = log5_vector(batter.rate_vector(), pitcher.rate_vector())

        # Fast multinomial sample: cumulative sum + uniform draw + searchsorted
        # Faster than rng.choice(8, p=probs) by ~3-5x in tight loops.
        cum = probs.cumsum()
        outcome_idx = int(cum.searchsorted(rng.random()))
        if outcome_idx >= 8:  # numerical safety
            outcome_idx = 7

        # Apply to base-out state
        state, scored = apply_outcome(outcome_idx, state, rng)
        runs += scored

        # Bookkeeping
        batting.advance_batter()
        pitching.record_pa()
        pa_count += 1

        if log_pbp and log is not None:
            from .league import PA_OUTCOMES
            outcome = PA_OUTCOMES[outcome_idx]
            log.append(
                f"  {batter.name} vs {pitcher.name}: {outcome}"
                f" -> bases={state.bases} outs={state.outs} +{scored}R"
            )

    return runs, pa_count


def simulate_game(
    home_team: Team,
    away_team: Team,
    rng: Optional[np.random.Generator] = None,
    log_pbp: bool = False,
) -> GameResult:
    """
    Simulate one full game between two teams.

    Home team bats bottom of inning. Game goes to extras if tied after 9.
    """
    if rng is None:
        rng = np.random.default_rng()

    home_state = LiveTeamState(team=home_team)
    away_state = LiveTeamState(team=away_team)

    # Initialize starters
    home_state.set_pitcher(home_team.rotation[0])
    away_state.set_pitcher(away_team.rotation[0])

    home_runs = 0
    away_runs = 0
    total_pa = 0
    log: List[str] = []
    inning = 1
    max_innings = 15  # safety cap

    while inning <= max_innings:
        if log_pbp:
            log.append(f"\n=== Inning {inning} (top) ===")
        r, pa = _play_half_inning(away_state, home_state, inning, rng, log_pbp, log)
        away_runs += r
        total_pa += pa

        # Bottom of inning: skip only if home leads after 9+ and is the home team batting last
        skip_bottom = (
            inning >= 9
            and home_runs > away_runs
            # never skip; trailing home team needs at-bats
        )
        # Standard rule: walk-off skip only when home team is winning entering bottom 9
        # Already encoded above.

        if not skip_bottom:
            if log_pbp:
                log.append(f"=== Inning {inning} (bot) ===")
            r, pa = _play_half_inning(home_state, away_state, inning, rng, log_pbp, log)
            home_runs += r
            total_pa += pa

        # After 9+ innings, end if not tied
        if inning >= 9 and home_runs != away_runs:
            break

        inning += 1

    home_won = home_runs > away_runs
    return GameResult(
        home_team_id=home_team.team_id,
        away_team_id=away_team.team_id,
        home_runs=home_runs,
        away_runs=away_runs,
        innings=inning,
        home_won=home_won,
        pa_count=total_pa,
        log=log,
    )
