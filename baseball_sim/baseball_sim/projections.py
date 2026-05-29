"""
Projection generation.

Two modes:
  1. SYNTHETIC (used here): deterministic-with-seed generator that produces
     30 MLB teams with realistic per-PA outcome distributions. Useful for
     development before real data ingestion is wired up.
  2. REAL (placeholder): load_real_projections() — a stub demonstrating where
     a FanGraphs/Steamer/ZiPS adapter would slot in. Returns the same Team
     objects with the same interface, so nothing downstream cares which
     source produced them.

Team-level talent variance is introduced via a per-team `talent` z-score in
roughly N(0, 0.15). Positive talent skews batters toward more power & contact
and pitchers toward more Ks / fewer walks & HRs.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .league import LEAGUE_2024
from .players import Batter, Pitcher
from .teams import Team


# ---- Real MLB team metadata (30 teams) ----
TEAM_META = [
    # AL East
    ("BAL", "Baltimore Orioles", "AL", "East"),
    ("BOS", "Boston Red Sox", "AL", "East"),
    ("NYY", "New York Yankees", "AL", "East"),
    ("TBR", "Tampa Bay Rays", "AL", "East"),
    ("TOR", "Toronto Blue Jays", "AL", "East"),
    # AL Central
    ("CWS", "Chicago White Sox", "AL", "Central"),
    ("CLE", "Cleveland Guardians", "AL", "Central"),
    ("DET", "Detroit Tigers", "AL", "Central"),
    ("KCR", "Kansas City Royals", "AL", "Central"),
    ("MIN", "Minnesota Twins", "AL", "Central"),
    # AL West
    ("HOU", "Houston Astros", "AL", "West"),
    ("LAA", "Los Angeles Angels", "AL", "West"),
    ("ATH", "Athletics", "AL", "West"),
    ("SEA", "Seattle Mariners", "AL", "West"),
    ("TEX", "Texas Rangers", "AL", "West"),
    # NL East
    ("ATL", "Atlanta Braves", "NL", "East"),
    ("MIA", "Miami Marlins", "NL", "East"),
    ("NYM", "New York Mets", "NL", "East"),
    ("PHI", "Philadelphia Phillies", "NL", "East"),
    ("WSN", "Washington Nationals", "NL", "East"),
    # NL Central
    ("CHC", "Chicago Cubs", "NL", "Central"),
    ("CIN", "Cincinnati Reds", "NL", "Central"),
    ("MIL", "Milwaukee Brewers", "NL", "Central"),
    ("PIT", "Pittsburgh Pirates", "NL", "Central"),
    ("STL", "St. Louis Cardinals", "NL", "Central"),
    # NL West
    ("ARI", "Arizona Diamondbacks", "NL", "West"),
    ("COL", "Colorado Rockies", "NL", "West"),
    ("LAD", "Los Angeles Dodgers", "NL", "West"),
    ("SDP", "San Diego Padres", "NL", "West"),
    ("SFG", "San Francisco Giants", "NL", "West"),
]


# ---- Population-level skill distributions ----
# These are the standard deviations of per-PA rates across MLB regulars.
# Means come from LEAGUE_2024 (i.e. league averages).

BATTER_SD = {
    "K":  0.045,
    "BB": 0.020,
    "HBP": 0.005,
    "HR": 0.013,
    "1B": 0.018,
    "2B": 0.010,
    "3B": 0.003,
    # OUT is residual after normalization
}

PITCHER_SD = {
    "K":  0.045,
    "BB": 0.020,
    "HBP": 0.004,
    "HR": 0.010,
    "1B": 0.014,
    "2B": 0.008,
    "3B": 0.002,
}


def _sample_batter_rates(
    rng: np.random.Generator,
    talent_z: float = 0.0,
) -> dict:
    """
    Draw a realistic batter rate vector.

    talent_z > 0 → better hitter (more power & contact, fewer Ks)
    talent_z < 0 → worse hitter
    """
    lc = LEAGUE_2024

    # Sign convention: which direction does positive talent push each stat?
    talent_push = {
        "K":  -1.0 * talent_z * 0.030,   # fewer Ks
        "BB": +0.5 * talent_z * 0.015,   # more walks
        "HBP": 0.0,
        "HR": +1.0 * talent_z * 0.010,   # more power
        "1B": +0.5 * talent_z * 0.010,
        "2B": +0.5 * talent_z * 0.008,
        "3B": 0.0,
    }

    means = {
        "K":  lc.k_rate,
        "BB": lc.bb_rate,
        "HBP": lc.hbp_rate,
        "HR": lc.hr_rate,
        "1B": lc.single_rate,
        "2B": lc.double_rate,
        "3B": lc.triple_rate,
    }

    rates = {}
    for key in means:
        val = means[key] + talent_push[key] + rng.normal(0, BATTER_SD[key])
        rates[key] = max(val, 0.001)

    # OUT is the residual (must be positive)
    rates["OUT"] = max(1.0 - sum(rates.values()), 0.20)
    return rates


def _sample_pitcher_rates(
    rng: np.random.Generator,
    talent_z: float = 0.0,
) -> dict:
    """
    Draw realistic per-PA rates ALLOWED by a pitcher.

    talent_z > 0 → better pitcher (more Ks, fewer BBs, fewer HRs allowed)
    """
    lc = LEAGUE_2024

    talent_push = {
        "K":  +1.0 * talent_z * 0.030,   # more Ks
        "BB": -0.8 * talent_z * 0.015,   # fewer walks
        "HBP": 0.0,
        "HR": -1.0 * talent_z * 0.010,   # fewer HRs
        "1B": -0.4 * talent_z * 0.010,
        "2B": -0.4 * talent_z * 0.008,
        "3B": 0.0,
    }

    means = {
        "K":  lc.k_rate,
        "BB": lc.bb_rate,
        "HBP": lc.hbp_rate,
        "HR": lc.hr_rate,
        "1B": lc.single_rate,
        "2B": lc.double_rate,
        "3B": lc.triple_rate,
    }

    rates = {}
    for key in means:
        val = means[key] + talent_push[key] + rng.normal(0, PITCHER_SD[key])
        rates[key] = max(val, 0.001)

    rates["OUT"] = max(1.0 - sum(rates.values()), 0.20)
    return rates


POSITIONS = ["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "DH"]


def _make_batter(rng: np.random.Generator, team_id: str, position: str, idx: int, team_talent: float) -> Batter:
    # Star players (top of lineup) are slightly better than bottom-of-order
    lineup_talent_bonus = (4 - idx) * 0.05  # spots 0-3 are stronger
    talent_z = team_talent + rng.normal(0, 0.6) + lineup_talent_bonus
    rates = _sample_batter_rates(rng, talent_z=talent_z)
    bats = rng.choice(["L", "R", "S"], p=[0.30, 0.62, 0.08])
    return Batter(
        player_id=f"{team_id}_B{idx:02d}",
        name=f"{team_id} Batter {idx + 1}",
        position=position,
        bats=str(bats),
        rates=rates,
        projected_pa=int(rng.normal(580, 60)),
    )


def _make_pitcher(rng: np.random.Generator, team_id: str, role: str, idx: int, team_talent: float) -> Pitcher:
    # Aces of the rotation a bit better than #5 starters
    # Role bonuses are kept modest so the weighted-by-PA average pitcher faced
    # stays near 0 talent, preserving league-mean PA outcomes in actual games.
    role_bonus = 0.0
    if role == "SP" and idx == 0:
        role_bonus = 0.20  # ace
    elif role == "SP" and idx == 1:
        role_bonus = 0.10
    elif role == "SP" and idx >= 3:
        role_bonus = -0.10  # back-end starters
    elif role == "RP" and idx < 2:
        role_bonus = 0.15  # high-leverage RPs (closer, setup)
    elif role == "RP" and idx >= 5:
        role_bonus = -0.15  # mop-up RPs

    talent_z = team_talent + rng.normal(0, 0.6) + role_bonus
    rates = _sample_pitcher_rates(rng, talent_z=talent_z)
    throws = rng.choice(["L", "R"], p=[0.30, 0.70])
    typical_bf = 25 if role == "SP" else 5
    typical_pitches = 95 if role == "SP" else 18
    return Pitcher(
        player_id=f"{team_id}_{role}{idx:02d}",
        name=f"{team_id} {role} {idx + 1}",
        role=role,
        throws=str(throws),
        rates=rates,
        projected_ip=180.0 if role == "SP" else 65.0,
        typical_batters=typical_bf,
        typical_pitches=typical_pitches,
    )


def _team_talent_for(team_id: str) -> float:
    """
    Deterministic team-level talent z-score. Roughly approximates 2024 team
    quality so the demo produces a believable standings shape. These are not
    real projections — just a plausible spread.

    Scale is intentionally moderate (±0.5) so 162-game win spreads come out
    around 60-100 wins like real MLB rather than 38-120.
    """
    talents = {
        "LAD": +0.50, "ATL": +0.45, "NYY": +0.40, "HOU": +0.35, "PHI": +0.40,
        "BAL": +0.30, "TBR": +0.25, "MIL": +0.25, "ARI": +0.20, "SEA": +0.25,
        "TEX": +0.15, "MIN": +0.15, "TOR": +0.15, "STL": +0.10, "BOS": +0.10,
        "SDP": +0.30, "NYM": +0.20, "CLE": +0.25, "DET": +0.15, "KCR": +0.15,
        "CHC": +0.05, "SFG": +0.05, "CIN": +0.05, "PIT": -0.05, "LAA": -0.15,
        "MIA": -0.25, "WSN": -0.20, "ATH": -0.30, "CWS": -0.45, "COL": -0.35,
    }
    return talents.get(team_id, 0.0)


def generate_synthetic_team(team_id: str, name: str, league: str, division: str, rng: np.random.Generator) -> Team:
    talent = _team_talent_for(team_id)

    lineup = [_make_batter(rng, team_id, POSITIONS[i], i, talent) for i in range(9)]
    rotation = [_make_pitcher(rng, team_id, "SP", i, talent) for i in range(5)]
    bullpen = [_make_pitcher(rng, team_id, "RP", i, talent) for i in range(7)]
    bench = [_make_batter(rng, team_id, "BENCH", i + 9, talent - 0.3) for i in range(4)]

    return Team(
        team_id=team_id,
        name=name,
        league=league,
        division=division,
        lineup=lineup,
        rotation=rotation,
        bullpen=bullpen,
        bench=bench,
    )


def generate_all_teams(seed: int = 42) -> List[Team]:
    """Generate all 30 MLB teams with synthetic projections (seeded for reproducibility)."""
    rng = np.random.default_rng(seed)
    return [generate_synthetic_team(tid, name, lg, div, rng) for tid, name, lg, div in TEAM_META]


def load_real_projections(season: int = 2026) -> List[Team]:  # pragma: no cover
    """
    Stub: load real projections from FanGraphs / Steamer / ZiPS.

    Implementation when ready:
      - Use pybaseball to pull projections by player_id
      - Map to Chadwick crosswalk for stable IDs
      - Convert rate stats (K%, BB%, HR/PA, etc.) into the 8-element vector
      - Use depth charts to allocate PA / IP per player
      - Return List[Team]

    For now this raises so callers see the boundary clearly.
    """
    raise NotImplementedError(
        "Real projection loader not yet implemented. "
        "Wire pybaseball + Chadwick crosswalk here. "
        "See README for Phase 0 instructions."
    )
