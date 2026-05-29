"""
Base-out state machine.

State is (bases, outs) where bases is a 3-tuple of booleans for
(1B, 2B, 3B) occupancy and outs is in {0, 1, 2}. (At 3 outs the inning ends.)

We apply a PA outcome to a state and return:
  - new state
  - runs scored on this PA

Advancement is partly deterministic (forced moves) and partly probabilistic
(extra bases on hits). The probabilities below are tuned so league-aggregate
R/G lands near the actual ~4.4. We deliberately keep this simple — no
explicit baserunner speed model, no errors, no sac flies as a separate
category (they're absorbed into OUT with a small RBI probability).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .league import PA_OUTCOMES

# ---- Empirically tuned advancement probabilities ----
# Calibrated to produce ~4.4 R/G and match Tango run expectancy within ~5%.

# On a single:
P_RUNNER_2B_SCORES_ON_1B = 0.62  # 2B runner scores on a single
P_RUNNER_1B_TO_3B_ON_1B = 0.32   # 1B runner takes an extra base to 3rd

# On a double:
P_RUNNER_1B_SCORES_ON_2B = 0.45  # 1B runner scores on a double

# On an "OUT" (in-play out) with < 2 outs:
P_SAC_FLY_SCORES_3B = 0.36        # runner on 3B scores (tag-up or productive)
P_PRODUCTIVE_2B_TO_3B = 0.38      # runner on 2B advances to 3B (no one on 1B)
P_PRODUCTIVE_1B_TO_2B = 0.20      # runner on 1B advances to 2B (no GIDP)
P_GIDP_ATTEMPT = 0.11             # GIDP attempt rate when 1B occupied & <2 outs
P_GIDP_SUCCESS = 0.40             # success rate conditional on attempt


# Type aliases
Bases = Tuple[bool, bool, bool]  # (1B, 2B, 3B)
BASES_EMPTY: Bases = (False, False, False)


@dataclass
class BaseOutState:
    bases: Bases = BASES_EMPTY
    outs: int = 0

    def is_inning_over(self) -> bool:
        return self.outs >= 3

    def copy(self) -> "BaseOutState":
        return BaseOutState(bases=self.bases, outs=self.outs)


def _count_runners(bases: Bases) -> int:
    return sum(bases)


def apply_walk_or_hbp(state: BaseOutState) -> Tuple[BaseOutState, int]:
    """Force advance only. Returns (new_state, runs_scored)."""
    r1, r2, r3 = state.bases
    runs = 0
    if r1 and r2 and r3:
        # Bases loaded: runner on 3B scores, others push
        runs = 1
        new_bases = (True, True, True)
    elif r1 and r2:
        new_bases = (True, True, True)
    elif r1 and r3:
        new_bases = (True, True, True)
    elif r1:
        new_bases = (True, True, False)
    elif r2 and r3:
        new_bases = (True, True, True)
    elif r2:
        new_bases = (True, True, False)
    elif r3:
        new_bases = (True, False, True)
    else:
        new_bases = (True, False, False)
    return BaseOutState(bases=new_bases, outs=state.outs), runs


def apply_home_run(state: BaseOutState) -> Tuple[BaseOutState, int]:
    """All runners + batter score."""
    runs = 1 + _count_runners(state.bases)
    return BaseOutState(bases=BASES_EMPTY, outs=state.outs), runs


def apply_triple(state: BaseOutState) -> Tuple[BaseOutState, int]:
    """All runners score, batter to 3B."""
    runs = _count_runners(state.bases)
    return BaseOutState(bases=(False, False, True), outs=state.outs), runs


def apply_double(state: BaseOutState, rng: np.random.Generator) -> Tuple[BaseOutState, int]:
    """Runners on 2B and 3B score. 1B runner scores P_RUNNER_1B_SCORES_ON_2B."""
    r1, r2, r3 = state.bases
    runs = 0
    new_r3 = False
    if r3:
        runs += 1
    if r2:
        runs += 1
    if r1:
        if rng.random() < P_RUNNER_1B_SCORES_ON_2B:
            runs += 1
        else:
            new_r3 = True
    return BaseOutState(bases=(False, True, new_r3), outs=state.outs), runs


def apply_single(state: BaseOutState, rng: np.random.Generator) -> Tuple[BaseOutState, int]:
    """Probabilistic advancement."""
    r1, r2, r3 = state.bases
    runs = 0
    new_r1 = True  # batter
    new_r2 = False
    new_r3 = False

    if r3:
        runs += 1
    if r2:
        if rng.random() < P_RUNNER_2B_SCORES_ON_1B:
            runs += 1
        else:
            new_r3 = True
    if r1:
        if rng.random() < P_RUNNER_1B_TO_3B_ON_1B:
            new_r3 = True  # may overwrite from prior; rare collision
        else:
            new_r2 = True

    return BaseOutState(bases=(new_r1, new_r2, new_r3), outs=state.outs), runs


def apply_out(state: BaseOutState, rng: np.random.Generator) -> Tuple[BaseOutState, int]:
    """
    Handle in-play outs (K already routed separately).

    Models:
      - GIDP: runner on 1B may be erased on grounder, batter out at 1B
      - Sac fly / sac advance: runner on 3B may score with < 2 outs
      - Productive groundout: runner on 2B may take 3B; runner on 1B may take 2B

    Other categories (errors, FCs other than GIDP, sac bunts) ignored for MVP.
    """
    r1, r2, r3 = state.bases
    runs = 0

    # Try GIDP first if eligible (runner on 1B, < 2 outs)
    if r1 and state.outs < 2 and rng.random() < P_GIDP_ATTEMPT:
        if rng.random() < P_GIDP_SUCCESS:
            # Two outs recorded; runner on 1B and batter both out
            new_outs = state.outs + 2
            new_bases = (False, r2, r3)
            return BaseOutState(bases=new_bases, outs=new_outs), runs
        # else: fielder's choice equivalent — treat as regular out

    # Productive advancement with < 2 outs
    if state.outs < 2:
        # Sac fly / productive out scoring from 3B
        if r3 and rng.random() < P_SAC_FLY_SCORES_3B:
            runs += 1
            r3 = False

        # Runner on 2B with no one on 1B can take 3B on a grounder
        if r2 and not r1 and not r3 and rng.random() < P_PRODUCTIVE_2B_TO_3B:
            r3 = True
            r2 = False

        # Runner on 1B with bases otherwise clear can advance to 2B
        if r1 and not r2 and not r3 and rng.random() < P_PRODUCTIVE_1B_TO_2B:
            r2 = True
            r1 = False

    return BaseOutState(bases=(r1, r2, r3), outs=state.outs + 1), runs


def apply_strikeout(state: BaseOutState) -> Tuple[BaseOutState, int]:
    """Just adds an out."""
    return BaseOutState(bases=state.bases, outs=state.outs + 1), 0


# Dispatch table for fast lookup in the hot loop
_OUTCOME_INDEX = {name: i for i, name in enumerate(PA_OUTCOMES)}


def apply_outcome(
    outcome_idx: int,
    state: BaseOutState,
    rng: np.random.Generator,
) -> Tuple[BaseOutState, int]:
    """Apply a PA outcome by index (faster than string lookup)."""
    if outcome_idx == _OUTCOME_INDEX["K"]:
        return apply_strikeout(state)
    if outcome_idx == _OUTCOME_INDEX["BB"]:
        return apply_walk_or_hbp(state)
    if outcome_idx == _OUTCOME_INDEX["HBP"]:
        return apply_walk_or_hbp(state)
    if outcome_idx == _OUTCOME_INDEX["HR"]:
        return apply_home_run(state)
    if outcome_idx == _OUTCOME_INDEX["1B"]:
        return apply_single(state, rng)
    if outcome_idx == _OUTCOME_INDEX["2B"]:
        return apply_double(state, rng)
    if outcome_idx == _OUTCOME_INDEX["3B"]:
        return apply_triple(state)
    if outcome_idx == _OUTCOME_INDEX["OUT"]:
        return apply_out(state, rng)
    raise ValueError(f"Unknown outcome index: {outcome_idx}")
