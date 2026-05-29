"""
Log5 / odds-ratio matchup model.

This is the heart of the PA engine. Given a batter's per-PA outcome rates
and a pitcher's per-PA outcome rates, plus league averages, we compute the
expected probability distribution over PA outcomes for that specific matchup.

The classic log5 formula (Bill James / Tom Tango) for a single outcome:

    P(outcome | B, P) = (B * P / L) / ((B * P / L) + (1 - B)(1 - P)/(1 - L))

We apply this per outcome (independently), then normalize. This is a small
approximation — the outcomes aren't truly independent — but normalization
papers over it and the result is remarkably accurate in practice. Every public
sim from PECOTA to OOTP uses some version of this.

For external adjustments (park, weather, handedness), apply *before* log5 by
multiplying the relevant batter/pitcher rates by a small factor. See
`apply_park_adjustment` for an example.
"""
from __future__ import annotations

import numpy as np

from .league import LEAGUE_2024, LeagueContext, league_rates_as_dict
from .players import Batter, Pitcher, rates_to_vector


# Precomputed league rate vector (avoids dict lookups in hot loops)
_LEAGUE_VEC = rates_to_vector(league_rates_as_dict(LEAGUE_2024))


def log5_vector(
    batter_rates: np.ndarray,
    pitcher_rates: np.ndarray,
    league_rates: np.ndarray = _LEAGUE_VEC,
) -> np.ndarray:
    """
    Vectorized log5 across all PA outcome categories.

    Args:
        batter_rates: shape (8,) batter's per-PA outcome rates
        pitcher_rates: shape (8,) pitcher's per-PA outcome rates allowed
        league_rates: shape (8,) league average per-PA rates

    Returns:
        shape (8,) normalized matchup probability vector summing to 1.0
    """
    # Odds ratio form: numerator = B * P / L, denominator adds (1-B)(1-P)/(1-L)
    # Rates are guaranteed > 0 and < 1 after normalization (no clip needed)
    num = (batter_rates * pitcher_rates) / league_rates
    denom = num + ((1 - batter_rates) * (1 - pitcher_rates)) / (1 - league_rates)
    raw = num / denom
    return raw / raw.sum()


def matchup_probs(
    batter: Batter,
    pitcher: Pitcher,
    league: LeagueContext = LEAGUE_2024,
) -> np.ndarray:
    """High-level convenience wrapper around log5_vector."""
    league_vec = rates_to_vector(league_rates_as_dict(league))
    return log5_vector(batter.rate_vector(), pitcher.rate_vector(), league_vec)


def apply_park_adjustment(
    rates: np.ndarray,
    park_factors: dict,
) -> np.ndarray:
    """
    Apply park factors as multiplicative adjustments.

    park_factors is a dict like {"HR": 1.08, "1B": 0.99, ...} where 1.00
    means neutral. Applied before log5. The MVP doesn't use this yet but
    the hook is here for Phase 3.
    """
    from .league import PA_OUTCOMES

    factors = np.array(
        [park_factors.get(k, 1.0) for k in PA_OUTCOMES], dtype=np.float64
    )
    return rates * factors
