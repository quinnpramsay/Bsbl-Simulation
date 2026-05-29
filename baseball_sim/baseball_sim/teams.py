"""
Teams and rosters.

A Team has:
  - 9 batters in a fixed lineup order (DH universal, so AL/NL both bat 9)
  - 5 starting pitchers (rotation)
  - 7 relief pitchers (bullpen)
  - park factors (placeholder for Phase 3)

A LiveTeamState tracks within-game state: current batting spot and pitcher.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .players import Batter, Pitcher


@dataclass
class Team:
    team_id: str
    name: str
    league: str  # "AL" or "NL"
    division: str  # "East", "Central", "West"
    lineup: List[Batter] = field(default_factory=list)  # 9 batters in order
    rotation: List[Pitcher] = field(default_factory=list)  # 5 SP
    bullpen: List[Pitcher] = field(default_factory=list)  # 7 RP
    bench: List[Batter] = field(default_factory=list)  # backup batters

    def starter_for_game(self, game_index: int) -> Pitcher:
        """Pick the rotation slot for a given game (round-robin)."""
        if not self.rotation:
            raise ValueError(f"{self.name} has no rotation")
        return self.rotation[game_index % len(self.rotation)]


@dataclass
class LiveTeamState:
    """Mutable within-game state for one team."""

    team: Team
    batting_spot: int = 0  # 0-8
    current_pitcher: Optional[Pitcher] = None
    batters_faced_by_current_pitcher: int = 0
    pitchers_used: List[str] = field(default_factory=list)

    def current_batter(self) -> Batter:
        return self.team.lineup[self.batting_spot]

    def advance_batter(self):
        self.batting_spot = (self.batting_spot + 1) % 9

    def set_pitcher(self, p: Pitcher):
        self.current_pitcher = p
        self.batters_faced_by_current_pitcher = 0
        self.pitchers_used.append(p.player_id)

    def record_pa(self):
        self.batters_faced_by_current_pitcher += 1
