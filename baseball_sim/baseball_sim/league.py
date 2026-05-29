"""
League context: average rates used as the baseline for log5.

These are approximate 2024 MLB league rates per plate appearance. Update these
each season (or pull dynamically) so log5 always normalizes against the
current run environment.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueContext:
    """Per-PA league average rates. Must sum to ~1.0 (we normalize anyway)."""

    # Plate appearance outcome rates
    k_rate: float = 0.226       # strikeouts
    bb_rate: float = 0.082      # walks (incl IBB folded in)
    hbp_rate: float = 0.012     # hit by pitch
    hr_rate: float = 0.030      # home runs
    single_rate: float = 0.143  # singles
    double_rate: float = 0.044  # doubles
    triple_rate: float = 0.004  # triples
    out_rate: float = 0.459     # in-play outs (groundouts, flyouts, lineouts)

    # Aggregate game-level constants (for sanity checks)
    runs_per_game: float = 4.39  # per team, 2024
    pa_per_game: float = 38.0    # per team
    avg_game_length_innings: float = 9.0


LEAGUE_2024 = LeagueContext()

# Outcome categories used everywhere downstream. Order matters because we
# build numpy probability vectors in this exact order.
PA_OUTCOMES = ["K", "BB", "HBP", "HR", "1B", "2B", "3B", "OUT"]


def league_rates_as_dict(ctx: LeagueContext = LEAGUE_2024) -> dict:
    """Return the league rate vector keyed by PA_OUTCOMES."""
    return {
        "K": ctx.k_rate,
        "BB": ctx.bb_rate,
        "HBP": ctx.hbp_rate,
        "HR": ctx.hr_rate,
        "1B": ctx.single_rate,
        "2B": ctx.double_rate,
        "3B": ctx.triple_rate,
        "OUT": ctx.out_rate,
    }
