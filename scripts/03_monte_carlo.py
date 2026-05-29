#!/usr/bin/env python3
"""
Demo 3: Run a Monte Carlo across N seasons, output win distributions
and playoff odds. This is the headline output of the MVP.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import argparse
import numpy as np
from baseball_sim import generate_all_teams, run_monte_carlo
from baseball_sim.validation import win_distribution_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of seasons to simulate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    teams = generate_all_teams(seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)

    print(f"Running Monte Carlo: {args.n} seasons × {len(teams)} teams ...")
    t0 = time.time()
    mc = run_monte_carlo(teams, n_seasons=args.n, rng=rng, verbose=True)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/args.n*1000:.0f}ms per season)")

    rows = mc.summary_table(teams)

    # Print by league + division
    print(f"\n{'='*100}")
    print(f"  PROJECTED STANDINGS ({args.n} season Monte Carlo)")
    print(f"{'='*100}")

    for league in ["AL", "NL"]:
        for division in ["East", "Central", "West"]:
            div = [r for r in rows if r["league"] == league and r["division"] == division]
            div.sort(key=lambda r: r["mean_W"], reverse=True)
            print(f"\n  {league} {division}")
            print(f"  {'Team':30s} {'mean W':>7s} {'p5':>5s} {'p95':>5s} {'RS':>5s} {'RA':>5s}  {'Div%':>5s}  {'Playoff%':>8s}")
            for r in div:
                print(f"  {r['name']:30s} {r['mean_W']:7.1f} "
                      f"{r['p5_W']:5.0f} {r['p95_W']:5.0f} "
                      f"{r['mean_RS']:5.0f} {r['mean_RA']:5.0f}  "
                      f"{r['division_pct']:5.1f}  {r['playoff_pct']:8.1f}")

    # Distribution summary
    print(f"\n{'='*100}")
    print("  CALIBRATION CHECK")
    print(f"{'='*100}")
    summary = win_distribution_summary(mc.win_matrix)
    print(f"  League average wins per team: {summary['league_avg_wins']:.1f}  (target: 81.0)")
    print(f"  Spread of team means (std):    {summary['spread_of_team_means']:.2f}  (real MLB ~9-11)")
    print(f"  Best team projected wins:      {summary['best_team_mean']:.1f}")
    print(f"  Worst team projected wins:     {summary['worst_team_mean']:.1f}")


if __name__ == "__main__":
    main()
