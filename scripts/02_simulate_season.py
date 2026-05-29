#!/usr/bin/env python3
"""
Demo 2: Simulate one full season and print standings.

This is one Monte Carlo draw — a single deterministic instance of how the
season could play out given these projections.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
from baseball_sim import generate_all_teams, simulate_season


def main():
    teams = generate_all_teams(seed=42)
    rng = np.random.default_rng(2026)

    print("Simulating one 162-game season ...")
    t0 = time.time()
    season = simulate_season(teams, rng=rng)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({season.total_games} games, {season.total_pa} PAs)")
    print(f"Throughput: {season.total_pa / elapsed:.0f} PAs/sec")

    # Print standings grouped by division
    print(f"\n{'='*70}")
    print("  STANDINGS (this season)")
    print(f"{'='*70}")

    for league in ["AL", "NL"]:
        for division in ["East", "Central", "West"]:
            div_teams = [t for t in teams if t.league == league and t.division == division]
            div_teams.sort(key=lambda t: season.win_pct(t.team_id), reverse=True)
            print(f"\n  {league} {division}")
            print(f"  {'Team':30s} {'W':>4s}-{'L':>4s}  {'Pct':>5s}  {'RS':>5s}  {'RA':>5s}  {'162-pace W':>11s}")
            for t in div_teams:
                w = season.wins[t.team_id]
                l = season.losses[t.team_id]
                pct = season.win_pct(t.team_id)
                rs = season.runs_scored[t.team_id]
                ra = season.runs_allowed[t.team_id]
                pace = season.projected_162(t.team_id)
                print(f"  {t.name:30s} {w:4d}-{l:4d}  {pct:5.3f}  {rs:5d}  {ra:5d}  {pace:11.1f}")


if __name__ == "__main__":
    main()
