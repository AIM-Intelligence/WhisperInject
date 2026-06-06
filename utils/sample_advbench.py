#!/usr/bin/env python3
"""
sample_advbench.py
------------------
Performs stratified sampling from the AdvBench dataset with statistical guarantees.

Uses finite-population-corrected (FPC) sample size calculation to determine
how many prompts to sample for a given confidence level and margin of error.

Usage
-----
# Sample with 80% confidence, ±15% margin (default)
python utils/sample_advbench.py --csv data/advbench_full.csv --output data/sample.csv

# Sample with 95% confidence, ±10% margin
python utils/sample_advbench.py --csv data/advbench_full.csv --output data/sample.csv \
    --confidence 0.95 --margin 0.10

# Exclude already-tested prompts
python utils/sample_advbench.py --csv data/advbench_full.csv --output data/sample.csv \
    --exclude-csv data/tested_prompts.csv
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from tabulate import tabulate


# Z-values for common confidence levels
CONF_Z = {
    0.80: 1.282,
    0.85: 1.440,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def fpc_n(N: int, p: float, e: float, z: float) -> int:
    """
    Finite-population-corrected sample size for a single proportion.

    Args:
        N: Population size
        p: Expected proportion (use 0.5 for max uncertainty)
        e: Margin of error (e.g., 0.15 for ±15%)
        z: Z-value for confidence level

    Returns:
        Required sample size
    """
    num = z**2 * p * (1 - p) * N
    denom = e**2 * (N - 1) + z**2 * p * (1 - p)
    return math.ceil(num / denom)


def allocate(df: pd.DataFrame, n_tot: int) -> dict:
    """
    Proportional allocation with floor+leftover to ensure Σn_h = n_tot exactly.

    Args:
        df: DataFrame with 'stratum' column
        n_tot: Total sample size to allocate

    Returns:
        Dict mapping stratum -> sample size
    """
    sizes = df["stratum"].value_counts()  # N_h for each stratum
    props = sizes / len(df)
    raw = props * n_tot

    # Floor every allocation, remember fractional parts
    alloc = {h: math.floor(r) for h, r in raw.items()}
    leftover = n_tot - sum(alloc.values())

    # Give +1s to strata with largest fractional parts
    frac_sorted = sorted(
        raw.items(), key=lambda x: x[1] - math.floor(x[1]), reverse=True
    )
    for h, _ in frac_sorted[:leftover]:
        alloc[h] += 1

    # Clamp to stratum population
    alloc = {h: min(alloc[h], sizes[h]) for h in sizes.index}
    return alloc


def stratified_sample(df: pd.DataFrame, n_tot: int, seed: int) -> tuple:
    """
    Perform stratified random sampling.

    Args:
        df: DataFrame with 'stratum' column
        n_tot: Total sample size
        seed: Random seed for reproducibility

    Returns:
        (sample_df, allocation_dict)
    """
    alloc = allocate(df, n_tot)
    parts = [
        df[df["stratum"] == h].sample(n=n_h, random_state=seed, replace=False)
        for h, n_h in alloc.items()
        if n_h > 0
    ]
    return pd.concat(parts).reset_index(drop=True), alloc


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling from AdvBench with statistical guarantees."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV with 'prompt' and 'stratum' columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("advbench_sample.csv"),
        help="Output CSV path (default: advbench_sample.csv)",
    )
    parser.add_argument(
        "--exclude-csv",
        type=Path,
        help="CSV with prompts to exclude (e.g., already tested)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.85,
        choices=[0.80, 0.85, 0.90, 0.95, 0.99],
        help="Confidence level (default: 0.80)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Margin of error, e.g., 0.15 for ±15%% (default: 0.15)",
    )
    parser.add_argument(
        "--prop",
        type=float,
        default=0.5,
        help="Expected proportion - use 0.5 for max uncertainty (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Override: sample exactly N items (ignores confidence/margin)",
    )
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} prompts from {args.csv}")

    # Exclude already-tested prompts
    if args.exclude_csv:
        exclude_df = pd.read_csv(args.exclude_csv)
        print(
            f"Loaded {len(exclude_df)} items to exclude from {args.exclude_csv}")

        if "prompt" in df.columns and "prompt" in exclude_df.columns:
            df = df[~df["prompt"].isin(exclude_df["prompt"])]
            print(f"After exclusion: {len(df)} prompts remaining")
        else:
            print("⚠️  Warning: 'prompt' column not found in one of the CSV files")

    if len(df) == 0:
        print("❌ No items left to sample after exclusion!")
        return

    # Ensure stratum column exists
    if "stratum" not in df.columns:
        print(f"⚠️  Warning: 'stratum' column not found in {args.csv}")
        print(f"   Available columns: {list(df.columns)}")
        df["stratum"] = "default"
        print("   Creating single stratum for all data")

    # Calculate sample size
    z = CONF_Z.get(args.confidence)
    if z is None:
        raise ValueError(f"Unsupported confidence level: {args.confidence}")

    if args.n:
        n = min(args.n, len(df))
        print(f"\n📊 Using override: n = {n}")
    else:
        n = fpc_n(len(df), p=args.prop, e=args.margin, z=z)
        print(f"\n📊 Population N = {len(df)}")
        print(
            f"   Target sample n = {n} "
            f"(CL={args.confidence*100:.0f}%, ±{args.margin*100:.1f}%, p={args.prop})"
        )

    # Perform stratified sampling
    sample_df, alloc = stratified_sample(df, n_tot=n, seed=args.seed)

    # Show allocation table
    alloc_table = [
        [stratum, df[df["stratum"] == stratum].shape[0], alloc.get(stratum, 0)]
        for stratum in sorted(alloc.keys())
    ]
    print("\n📋 Allocation by stratum:")
    print(tabulate(alloc_table, headers=[
          "Stratum", "Population", "Sample"], tablefmt="github"))

    # Save sample
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.output, index=False)
    print(f"\n✅ Saved {len(sample_df)} prompts → {args.output}")


if __name__ == "__main__":
    main()
