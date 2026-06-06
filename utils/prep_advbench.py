#!/usr/bin/env python3
"""
prep_advbench.py
----------------
Prepares the AdvBench dataset for large-scale experiments.

- Loads walledai/AdvBench from HuggingFace OR from a local CSV
- Tags each prompt with a harm stratum (category) using regex rules
- Outputs CSV with: prompt, stratum

Usage
-----
# From HuggingFace (requires login for gated dataset)
python utils/prep_advbench.py --out-csv data/advbench_full.csv

# From local CSV (if you already have the data)
python utils/prep_advbench.py --from-csv /path/to/existing.csv --out-csv data/advbench.csv

# Specific split from HuggingFace
python utils/prep_advbench.py --split train --out-csv data/advbench_train.csv

# Show distribution only (no file output)
python utils/prep_advbench.py --from-csv data/existing.csv
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tabulate import tabulate


# --------------------------------------------------------------------- #
# Strata (harm category) definitions
# --------------------------------------------------------------------- #
STRATA_PATTERNS = {
    "violence": [
        r"\b(kill|murder|assassinat(e|ion)|bomb|violence|bombs|disrupt(?:ing)?|weapon|weapons|explosive|shoot|gun|firearm|terrorist|massacre|assault|hit-and-run)\b",
        r"\b(physical harm|hijack.*car|plant a bomb|poison\w* (a )?(person|someone|individual|people|water supply)|sabotage|mass shooting|terrorist attack|violent crime)\b",
        r"\b(disrupt.*\b(systems?|infrastructures?)|eco-terrorism|harm animals|start a fire|abduction|torture)\b",
        r"\b(interfere with.*\b(systems?|infrastructures?))\b",
    ],
    "illicit_cyber": [
        r"\b(hack|phish|malware|virus|ransomware|backdoor|crack(?:ing)? passwords?|passwords|keylogger|spyware|botnet|DDoS|spoof|cyber attack)\b",
        r"\b(steal.*(data|information|credentials))|(exploit.*vulnerabilit)|(hijack.*(car|drone|system|website|webcam))\b",
        r"\b(bypass security|disable antivirus|secure network|government database|spy on people|(?:track|monitor|spy on|surveil)\b.*\b(location|whereabouts|movements)|collects personal data)\b",
        r"\b(leaks confidential data|accesses and manipulates.*data|crash a computer system|denies access to a website|deletes files from.*computer)\b",
        r"(tutorial|guide|instructions|automates).*(hacking|virus|cyberattack)",
        r"\b(spam|harvest.*data|circumvent.*(censorship|surveillance|digital rights management))\b",
    ],
    "harmful_speech": [
        r"\b(hate speech|defam\w*|slander|libel|propaganda|incit(e|ing) violence|radicalization|glorif\w* (?:violence|war|gang violence))\b",
        r"\b(manipulat(e|ion)|trick people|cyberbully|harass|stalk|threaten|blackmail|false accusations|bullying|inflammatory comments)\b",
        r"\b(dangerous (?:behaviors|stunts|challenge)|(?:drug|alcohol) use|reckless driving|eating disorder)\b",
        r"\b(exploit.*(vulnerable populations|insecurities|fears)|promotes.*(drugs|steroids|dangerous dieting)|encourage.*commit crimes)\b",
        r"(teach|guide|tutorial|instructions).*(cyberbullying|cyberstalking)",
    ],
    "hate": [
        r"\b(racist|nazi|supremacist|bigot|slur|xenophobic|homophobic|sexist)\b",
        r"\b(violence against.*(group|minority|women))|(oppression of.*group)|(dehumanization)|(inferior to others)|(threatening letter to.*minority)\b",
    ],
    "fraud": [
        r"\b(counterfeit|insider trading|steal.*(?:money|identity)|fraud|phishing|scam)\b",
        r"\b(credit card fraud|financial crimes|tax fraud|insurance fraud|mail fraud|falsify.*records)\b",
        r"(tutorial|guide|instructions|how to|educate).*(identity theft|credit card)",
        r"\b(?:steal|theft|obtain|use).*credit card(?: information)?\b",
        r"\bmanipulate.*financial market\b",
    ],
    "deception_rule_evasion": [
        r"\b(cheat on.*(exam|test))|(impersonat)|(fake (review|ID|identity|news|online store|product review))\b",
        r"\b(evad(e|ing) (?:law|taxes|detection|security))|(get away with)|(cover your tracks|concealing criminal activity)\b",
        r"\b(lie to|mislead|cheat(?:ing)|deceive|trick|review.*(not even used|without.*(using|testing))|manipulates customer ratings|automates.*fake reviews)\b",
        r"\b(circumvent.*government censorship)\b",
    ],
    "illegal_acts_goods": [
        r"\b(illegal drugs|smuggl(e|ing)|trafficking|poach|fake id|counterfeit money|illegal download|piracy|pirate|pirated content)\b",
        r"\b(steal.*(?:car|store|charity))|(pick a lock|vandalize|trespass|rob a bank|escape from.*(prison|correctional facility))|(infiltrate.*facility)\b",
        r"\b(synthesized at home|homemade firearms|date rape drug|acquire.*firearms illegally|teach.*firearms)\b",
        r"(tutorial|guide|instructions).*(illegal.*download|counterfeit.*money|fake id)",
    ],
    "self_harm": [
        r"\b(kill myself|suicide|self[- ]harm|end my life|how to die|anorexia|bulimia|eating disorder)\b",
    ],
    "sexual_exploit": [
        r"\b(sexual.*minors?|child(?:ren)? porn|child pornography|exploit(?:ing)? minor|groom|grooming|sex|exploitation|molest|child abuse)\b",
        r"\b(acquire.*child porn|solicit sexual conversations.*minor|promotes.*child pornography)\b",
    ],
    "misinformation": [
        r"\b(fake news|false information|misinformation|conspiracy theor(y|ies))\b",
        r"(deadly side effects.*vaccine)|(climate change.*hoax)",
        r"\b(mislead.*public about.*(climate change|healthcare))\b",
    ],
}

# Pre-compile patterns for efficiency
COMPILED_PATTERNS = {
    name: [re.compile(p, flags=re.I) for p in pats]
    for name, pats in STRATA_PATTERNS.items()
}


def assign_stratum(prompt: str) -> str:
    """Return first matching stratum or 'other'."""
    for name, pats in COMPILED_PATTERNS.items():
        if any(p.search(prompt) for p in pats):
            return name
    return "other"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AdvBench dataset for large-scale experiments."
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        help="Load from local CSV instead of HuggingFace (must have 'prompt' column)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation", "all"],
        help="Dataset split to use when loading from HuggingFace (default: train)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Output CSV path (e.g., data/advbench.csv)",
    )
    parser.add_argument(
        "--save-unmatched",
        type=Path,
        help="Save prompts classified as 'other' to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    args = parser.parse_args()

    # Load dataset from local CSV or HuggingFace
    if args.from_csv:
        print(f"Loading from local CSV: {args.from_csv}")
        df = pd.read_csv(args.from_csv)
        if "prompt" not in df.columns:
            print(
                f"❌ Error: CSV must have 'prompt' column. Found: {list(df.columns)}")
            return
        # Convert to list of dicts for consistent processing
        prompts = df["prompt"].tolist()
        total = len(prompts)
        print(f"Loaded {total} prompts from local CSV.")
    else:
        split_str = args.split if args.split != "all" else "train+test+validation"
        print(f"Loading walledai/AdvBench ({args.split} split)...")
        print("💡 Note: This is a gated dataset. Run 'huggingface-cli login' if you get access errors.")
        ds = load_dataset("walledai/AdvBench", split=split_str)
        prompts = [row["prompt"] for row in ds]
        total = len(prompts)
        print(f"Loaded {total} prompts from HuggingFace.")

    # Assign strata to each prompt
    counts = Counter()
    strata = []
    others = []

    for prompt in prompts:
        stratum = assign_stratum(prompt)
        strata.append(stratum)
        counts[stratum] += 1
        if stratum == "other":
            others.append(prompt)

    # Display frequency table
    if not args.quiet:
        table = [
            [k, v, f"{100 * v / total:5.2f}%"]
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        ]
        print("\n📊 Stratum Distribution:\n")
        print(tabulate(table, headers=[
              "Stratum", "Count", "Percent"], tablefmt="github"))

    # Save unmatched prompts (optional)
    if args.save_unmatched and others:
        args.save_unmatched.parent.mkdir(parents=True, exist_ok=True)
        args.save_unmatched.write_text("\n\n".join(others), encoding="utf-8")
        print(
            f"\n⚠️  Saved {len(others)} unmatched prompts → {args.save_unmatched}")

    # Save dataset to CSV
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Create DataFrame with prompt and stratum
        out_df = pd.DataFrame({
            "prompt": prompts,
            "stratum": strata
        })
        out_df.to_csv(args.out_csv, index=False)
        print(f"\n✅ Dataset saved to: {args.out_csv}")
        print(f"   Columns: prompt, stratum")
        print(f"   Rows: {len(out_df)}")
    elif not args.quiet:
        print("\n💡 Use --out-csv to save the dataset to a CSV file.")

    print("\nDone.")


if __name__ == "__main__":
    main()
