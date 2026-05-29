"""
Monte Carlo runner: simulate many seasons, aggregate distributions.

Outputs per team:
  - mean wins
  - 5th / 50th / 95th percentile of wins
  - playoff odds (top 3 in division per league = 6 teams)
  - division win odds
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .season import SeasonResult, build_schedule, simulate_season
from .teams import Team


@dataclass
class MonteCarloResult:
    n_seasons: int
    team_ids: List[str]
    win_matrix: np.ndarray  # shape (n_seasons, n_teams), normalized to 162
    runs_scored_matrix: np.ndarray
    runs_allowed_matrix: np.ndarray
    playoff_appearances: Dict[str, int] = field(default_factory=dict)
    division_titles: Dict[str, int] = field(default_factory=dict)

    def summary_table(self, teams: List[Team]) -> List[dict]:
        rows = []
        team_by_id = {t.team_id: t for t in teams}
        for i, tid in enumerate(self.team_ids):
            t = team_by_id[tid]
            wins = self.win_matrix[:, i]
            rs = self.runs_scored_matrix[:, i]
            ra = self.runs_allowed_matrix[:, i]
            rows.append({
                "team_id": tid,
                "name": t.name,
                "league": t.league,
                "division": t.division,
                "mean_W": float(np.mean(wins)),
                "p5_W": float(np.percentile(wins, 5)),
                "p50_W": float(np.percentile(wins, 50)),
                "p95_W": float(np.percentile(wins, 95)),
                "mean_RS": float(np.mean(rs)),
                "mean_RA": float(np.mean(ra)),
                "playoff_pct": 100.0 * self.playoff_appearances.get(tid, 0) / self.n_seasons,
                "division_pct": 100.0 * self.division_titles.get(tid, 0) / self.n_seasons,
            })
        return rows


def _playoff_teams(season: SeasonResult, teams: List[Team]) -> tuple[set, set]:
    """
    Determine playoff teams and division winners for a given simulated season.

    Simplified: top team per division wins the division (3 per league). Top 3
    remaining teams by win% per league get wild cards. 6 playoff teams per
    league = 12 total.
    """
    by_league: Dict[str, List[Team]] = {"AL": [], "NL": []}
    for t in teams:
        by_league[t.league].append(t)

    playoff_ids = set()
    div_winners = set()

    for league_teams in by_league.values():
        # Group by division
        by_div: Dict[str, List[Team]] = {}
        for t in league_teams:
            by_div.setdefault(t.division, []).append(t)

        wild_card_pool = []
        for div_teams in by_div.values():
            # Sort by win percentage descending
            ranked = sorted(div_teams, key=lambda t: season.win_pct(t.team_id), reverse=True)
            div_winners.add(ranked[0].team_id)
            playoff_ids.add(ranked[0].team_id)
            wild_card_pool.extend(ranked[1:])

        # Top 3 wild cards by win% across league
        wc_ranked = sorted(wild_card_pool, key=lambda t: season.win_pct(t.team_id), reverse=True)
        for t in wc_ranked[:3]:
            playoff_ids.add(t.team_id)

    return playoff_ids, div_winners


def run_monte_carlo(
    teams: List[Team],
    n_seasons: int = 1000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> MonteCarloResult:
    """Run N season simulations and aggregate."""
    if rng is None:
        rng = np.random.default_rng()

    n_teams = len(teams)
    team_ids = [t.team_id for t in teams]
    team_idx = {tid: i for i, tid in enumerate(team_ids)}

    win_matrix = np.zeros((n_seasons, n_teams), dtype=np.float64)
    rs_matrix = np.zeros((n_seasons, n_teams), dtype=np.float64)
    ra_matrix = np.zeros((n_seasons, n_teams), dtype=np.float64)
    playoff_counts: Dict[str, int] = {tid: 0 for tid in team_ids}
    division_counts: Dict[str, int] = {tid: 0 for tid in team_ids}

    for s in range(n_seasons):
        season = simulate_season(teams, rng=rng)
        for tid in team_ids:
            i = team_idx[tid]
            win_matrix[s, i] = season.projected_162(tid)  # normalize to 162 G
            rs_matrix[s, i] = season.runs_scored[tid]
            ra_matrix[s, i] = season.runs_allowed[tid]

        playoff_ids, div_ids = _playoff_teams(season, teams)
        for tid in playoff_ids:
            playoff_counts[tid] += 1
        for tid in div_ids:
            division_counts[tid] += 1

        if verbose and (s + 1) % max(1, n_seasons // 10) == 0:
            print(f"  Season {s + 1}/{n_seasons} complete")

    return MonteCarloResult(
        n_seasons=n_seasons,
        team_ids=team_ids,
        win_matrix=win_matrix,
        runs_scored_matrix=rs_matrix,
        runs_allowed_matrix=ra_matrix,
        playoff_appearances=playoff_counts,
        division_titles=division_counts,
    )
