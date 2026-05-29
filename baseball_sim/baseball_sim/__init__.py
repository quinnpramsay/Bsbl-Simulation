"""
baseball_sim: Phase 0 + Phase 1 MVP

PA-level log5 simulation engine with base-out state machine, game and season
simulators, and a Monte Carlo runner.

Public surface:
  - League constants (LEAGUE_2024, PA_OUTCOMES)
  - Player models (Batter, Pitcher)
  - Team model (Team, LiveTeamState)
  - log5 matchup function
  - simulate_game(), simulate_season(), run_monte_carlo()
  - generate_all_teams() for synthetic data
"""
from .game import GameResult, simulate_game
from .league import LEAGUE_2024, PA_OUTCOMES, LeagueContext
from .log5 import log5_vector, matchup_probs
from .monte_carlo import MonteCarloResult, run_monte_carlo
from .players import Batter, Pitcher
from .projections import generate_all_teams, load_real_projections
from .season import SeasonResult, simulate_season
from .teams import LiveTeamState, Team

__all__ = [
    "LEAGUE_2024",
    "PA_OUTCOMES",
    "LeagueContext",
    "Batter",
    "Pitcher",
    "Team",
    "LiveTeamState",
    "log5_vector",
    "matchup_probs",
    "GameResult",
    "simulate_game",
    "SeasonResult",
    "simulate_season",
    "MonteCarloResult",
    "run_monte_carlo",
    "generate_all_teams",
    "load_real_projections",
]
