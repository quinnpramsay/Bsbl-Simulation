#!/usr/bin/env python3
"""
Demo 1: Simulate a single game with detailed play-by-play.

Shows how the PA engine and state machine compose into a full game.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from baseball_sim import generate_all_teams, simulate_game


def main():
    teams = generate_all_teams(seed=42)
    by_id = {t.team_id: t for t in teams}

    home = by_id["LAD"]
    away = by_id["NYY"]

    rng = np.random.default_rng(123)
    result = simulate_game(home, away, rng=rng, log_pbp=True)

    print(f"\n{'='*60}")
    print(f"  {away.name} @ {home.name}")
    print(f"{'='*60}")

    # Show first inning and a couple highlights, then the final
    print("\n  PLAY-BY-PLAY (first 2 innings):")
    inning_breaks_seen = 0
    for line in result.log:
        if line.startswith("\n=== Inning") or line.startswith("=== Inning"):
            if "Inning 3" in line:
                break
        print(line)

    print(f"\n  ... [{result.innings} innings total, {result.pa_count} PAs] ...")

    print(f"\n  FINAL: {away.name} {result.away_runs} — {home.name} {result.home_runs}")
    print(f"  Winner: {away.name if not result.home_won else home.name}")


if __name__ == "__main__":
    main()
