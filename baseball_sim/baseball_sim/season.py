"""
Schedule generation and season simulator.

For the MVP we generate a simplified schedule where every pair of teams plays
6 games (3 home, 3 away). That gives each team 29 * 6 = 174 games, slightly
more than the real 162 but it keeps the schedule symmetric and avoids
strength-of-schedule artifacts. Easy to swap in a real schedule later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .game import GameResult, simulate_game
from .teams import Team


GAMES_PER_PAIRING = 6  # 3 home + 3 away
# Total games per team = (n_teams - 1) * GAMES_PER_PAIRING


def build_schedule(teams: List[Team], rng: np.random.Generator) -> List[Tuple[Team, Team]]:
    """
    Build a round-robin schedule. Returns list of (home_team, away_team) pairs.

    Each unordered pair plays GAMES_PER_PAIRING games, split evenly home/away.
    The list is shuffled so games aren't clustered.
    """
    games: List[Tuple[Team, Team]] = []
    half = GAMES_PER_PAIRING // 2
    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1 :]:
            for _ in range(half):
                games.append((t1, t2))  # t1 home
                games.append((t2, t1))  # t2 home
    rng.shuffle(games)
    return games


@dataclass
class SeasonResult:
    wins: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, int] = field(default_factory=dict)
    runs_scored: Dict[str, int] = field(default_factory=dict)
    runs_allowed: Dict[str, int] = field(default_factory=dict)
    total_pa: int = 0
    total_games: int = 0

    def win_pct(self, team_id: str) -> float:
        w = self.wins.get(team_id, 0)
        l = self.losses.get(team_id, 0)
        return w / (w + l) if (w + l) > 0 else 0.0

    def projected_162(self, team_id: str) -> float:
        return self.win_pct(team_id) * 162


def simulate_season(
    teams: List[Team],
    rng: Optional[np.random.Generator] = None,
    schedule: Optional[List[Tuple[Team, Team]]] = None,
) -> SeasonResult:
    """Simulate one full season. Returns standings + run totals."""
    if rng is None:
        rng = np.random.default_rng()
    if schedule is None:
        schedule = build_schedule(teams, rng)

    result = SeasonResult()
    for t in teams:
        result.wins[t.team_id] = 0
        result.losses[t.team_id] = 0
        result.runs_scored[t.team_id] = 0
        result.runs_allowed[t.team_id] = 0

    for home, away in schedule:
        gr = simulate_game(home, away, rng=rng)
        if gr.home_won:
            result.wins[home.team_id] += 1
            result.losses[away.team_id] += 1
        else:
            result.wins[away.team_id] += 1
            result.losses[home.team_id] += 1
        result.runs_scored[home.team_id] += gr.home_runs
        result.runs_scored[away.team_id] += gr.away_runs
        result.runs_allowed[home.team_id] += gr.away_runs
        result.runs_allowed[away.team_id] += gr.home_runs
        result.total_pa += gr.pa_count
        result.total_games += 1

    return result
