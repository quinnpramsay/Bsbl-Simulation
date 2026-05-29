"""
Player models.

Batters and pitchers are characterized by their projected per-PA outcome rates.
This is the minimum information the log5 engine needs. Everything else (age,
handedness, splits, pitch-type tendencies, etc.) can layer on top.

For the MVP we keep things minimal: name, position, handedness, and an
8-element rate vector that should approximately sum to 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .league import PA_OUTCOMES


def _normalize(rates: Dict[str, float]) -> Dict[str, float]:
    """Ensure rates sum to 1.0 (small projection drift is fine)."""
    total = sum(rates.values())
    if total <= 0:
        raise ValueError("Rate vector must have positive sum")
    return {k: v / total for k, v in rates.items()}


def rates_to_vector(rates: Dict[str, float]) -> np.ndarray:
    """Convert a dict of rates into a numpy vector in PA_OUTCOMES order."""
    return np.array([rates[k] for k in PA_OUTCOMES], dtype=np.float64)


@dataclass
class Batter:
    """A position player with projected per-PA outcome rates."""

    player_id: str
    name: str
    position: str
    bats: str  # "L", "R", or "S" (switch)
    rates: Dict[str, float] = field(default_factory=dict)
    projected_pa: int = 600  # expected playing time

    def __post_init__(self):
        self.rates = _normalize(self.rates)
        self._rate_vec = rates_to_vector(self.rates)  # cached

    def rate_vector(self) -> np.ndarray:
        return self._rate_vec


@dataclass
class Pitcher:
    """A pitcher with projected per-PA outcome rates (rates they ALLOW)."""

    player_id: str
    name: str
    role: str  # "SP" or "RP"
    throws: str  # "L" or "R"
    rates: Dict[str, float] = field(default_factory=dict)
    projected_ip: float = 180.0  # expected innings
    # Stamina parameters
    typical_pitches: int = 95  # pitches before fatigue hits hard
    typical_batters: int = 25  # batters faced (proxy for stamina)

    def __post_init__(self):
        self.rates = _normalize(self.rates)
        self._rate_vec = rates_to_vector(self.rates)  # cached

    def rate_vector(self) -> np.ndarray:
        return self._rate_vec
