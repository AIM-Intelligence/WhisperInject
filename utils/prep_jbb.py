#!/usr/bin/env python3
"""
prep_jbb.py
-----------
Prepares the JailbreakBench JBB-Behaviors dataset for large-scale experiments.

- Loads JailbreakBench/JBB-Behaviors from HuggingFace
- Maps Goal → prompt, Category → stratum
- Outputs CSV compatible with sample_advbench.py and batch_attack.py

Dataset Info:
- 100 harmful behaviors, 100 benign counterparts
- 10 balanced categories (10 prompts each)
- Categories: Harassment/Discrimination, Malware/Hacking, Physical harm,
  Economic harm, Fraud/Deception, Disinformation, Sexual/Adult content,
  Privacy, Expert advice, Government decision-making

Usage
-----
# Export all harmful behaviors (default)
python utils/prep_jbb.py --out-csv data/jbb_harmful.csv

# Export benign behaviors
python utils/prep_jbb.py --split benign --out-csv data/jbb_benign.csv

# Sample N per category (e.g., 5 per category = 50 total)
python utils/prep_jbb.py --out-csv data/jbb_sample.csv --per-category 5

# Include target prefix column
python utils/prep_jbb.py --out-csv data/jbb_with_targets.csv --include-target

# Show distribution only (no file output)
python utils/prep_jbb.py
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser(
        description="Prepare JBB-Behaviors dataset for experiments."
    )
    parser.add_argument(
        "--split",
        default="harmful",
        choices=["harmful", "benign"],
        help="Dataset split to use (default: harmful)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Output CSV path (e.g., data/jbb_harmful.csv)",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        help="Sample N prompts per category (default: all)",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Include 'target' column with expected response prefix",
    )
    parser.add_argument(
        "--include-behavior",
        action="store_true",
        help="Include 'behavior' column with specific behavior type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading JailbreakBench/JBB-Behaviors ({args.split} split)...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors",
                      "behaviors", split=args.split)

    # Convert to DataFrame
    df = pd.DataFrame({
        "prompt": ds["Goal"],
        "stratum": ds["Category"],
        "target": ds["Target"],
        "behavior": ds["Behavior"],
        "source": ds["Source"],
    })

    total = len(df)
    print(f"Loaded {total} prompts.")

    # Sample per category if requested
    if args.per_category:
        n_per = args.per_category
        max_per = df["stratum"].value_counts().min()
        if n_per > max_per:
            print(
                f"⚠️  Warning: Requested {n_per} per category, but smallest category has {max_per}")
            n_per = max_per

        sampled_parts = []
        for stratum in df["stratum"].unique():
            stratum_df = df[df["stratum"] == stratum]
            sampled_parts.append(
                stratum_df.sample(
                    n=n_per, random_state=args.seed, replace=False)
            )
        df = pd.concat(sampled_parts).reset_index(drop=True)
        print(f"Sampled {n_per} per category → {len(df)} total prompts")

    # Display frequency table
    if not args.quiet:
        counts = Counter(df["stratum"])
        table = [
            [k, v, f"{100 * v / len(df):5.2f}%"]
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        ]
        print("\n📊 Category Distribution:\n")
        print(tabulate(table, headers=[
              "Category", "Count", "Percent"], tablefmt="github"))

        # Show source distribution
        source_counts = Counter(df["source"])
        source_table = [
            [k, v, f"{100 * v / len(df):5.2f}%"]
            for k, v in sorted(source_counts.items(), key=lambda x: -x[1])
        ]
        print("\n📚 Source Distribution:\n")
        print(tabulate(source_table, headers=[
              "Source", "Count", "Percent"], tablefmt="github"))

    # Prepare output columns
    output_cols = ["prompt", "stratum"]
    if args.include_target:
        output_cols.append("target")
    if args.include_behavior:
        output_cols.append("behavior")

    # Save dataset to CSV
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df = df[output_cols]
        out_df.to_csv(args.out_csv, index=False)
        print(f"\n✅ Dataset saved to: {args.out_csv}")
        print(f"   Columns: {', '.join(output_cols)}")
        print(f"   Rows: {len(out_df)}")
    elif not args.quiet:
        print("\n💡 Use --out-csv to save the dataset to a CSV file.")

    # Show sample prompts
    if not args.quiet:
        print("\n📝 Sample prompts:")
        for i, row in df.head(3).iterrows():
            print(f"  [{row['stratum']}] {row['prompt'][:80]}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
