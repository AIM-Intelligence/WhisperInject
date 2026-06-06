#!/usr/bin/env python3
"""
Whisper Inject - Batch Two-Stage Attack Runner

Runs the two-stage safety-bypass attack over a CSV of harmful prompts and writes
results in the directory layout consumed by the evaluators (`evals.py`,
`add_jailbreakeval.py`):

    results/batch_<timestamp>_<model>/
    ├── stage1_cache.json              # discovered Stage 1 payloads (for resume)
    ├── summary.json                   # per-stage success counts
    ├── case_001_<slug>/
    │   ├── config.json                # prompt, stratum, stage1/stage2, responses
    │   ├── adversarial.wav
    │   ├── stage1/                     # tracker logs
    │   └── stage2/                     # tracker logs
    └── ...

Usage:
    python batch_attack.py --input-csv data/jbb_full.csv --model qwen-3b
    python batch_attack.py --input-csv data/advbench_sample.csv --model gemma-2b \
        --eps 0.1 --stage1-steps 100 --stage2-steps 150

Resume an interrupted run by pointing at the same output directory; cases that
already have a config.json are skipped and cached Stage 1 payloads are reused:
    python batch_attack.py --input-csv data/jbb_full.csv --model qwen-3b \
        --output-dir results/batch_20260106_120000_qwen-3b
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def slugify(text: str, max_len: int = 40) -> str:
    """Make a filesystem-safe slug from a prompt."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return slug[:max_len] or "case"


def load_cases(csv_path: str):
    """Read prompts from a CSV with columns: prompt[, stratum]."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        raise ValueError(
            f"Input CSV must have a 'prompt' column. Found: {list(df.columns)}"
        )
    cases = []
    for _, row in df.iterrows():
        cases.append({
            "prompt": str(row["prompt"]),
            "stratum": str(row["stratum"]) if "stratum" in df.columns and not pd.isna(row.get("stratum")) else "unknown",
        })
    return cases


def write_case_config(case_dir: Path, args, prompt, stratum, result):
    """Write a config.json compatible with evals.py / add_jailbreakeval.py.

    Note: full (untruncated) Stage 1 behavior and Stage 2 response are stored,
    since the evaluators score the complete text.
    """
    config = {
        "timestamp": case_dir.parent.name,
        "model": args.model,
        "attack": "two-stage",
        "prompt": prompt,
        "stratum": stratum,
        "harmful_query": prompt,
        "benign_query": result.benign_query,
        "parameters": {
            "eps": args.eps,
            "alpha": args.alpha,
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "stage1_type": args.stage1_type,
            "semantic_weight": args.semantic_weight,
            "judge_model": args.judge_model,
        },
        "stage1": {
            "success": result.stage1_success,
            "score": result.stage1_score,
            "steps": result.stage1_steps,
            "behavior": result.stage1_behavior or "",
        },
        "stage2": {
            "success": result.stage2_success,
            "judge_score": result.stage2_judge_score,
            "similarity": result.semantic_similarity,
            "steps": result.stage2_steps,
            "final_response": result.final_response or "",
        },
        "overall_success": bool(result.stage1_success and result.stage2_success),
    }
    with open(case_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Batch two-stage attack runner (produces evaluator-ready results)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input-csv", required=True,
                        help="CSV of prompts (columns: prompt[, stratum])")
    parser.add_argument("--model", default="qwen-3b",
                        choices=["gemma-4b", "gemma-2b", "qwen-3b", "qwen-7b", "phi", "voxtral"],
                        help="Target model (default: qwen-3b)")
    parser.add_argument("--benign-query", default="How is the weather today?",
                        help="Benign carrier query for Stage 2")

    # Attack params
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=150)
    parser.add_argument("--stage1-type", choices=["wav", "mel"], default="wav")
    parser.add_argument("--semantic-weight", type=float, default=0.7)
    parser.add_argument("--judge-model", default=None,
                        help="LLM judge model (default: JUDGE_LLM_MODEL env or gpt-4o-mini)")
    parser.add_argument("--stage2-max-runs", type=int, default=3)
    parser.add_argument("--stage2-max-restarts", type=int, default=20)
    parser.add_argument("--stage2-drift-threshold", type=int, default=3)
    parser.add_argument("--lowpass", action="store_true")
    parser.add_argument("--lowpass-cutoff", type=float, default=2000.0)

    # Run control
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: results/batch_<ts>_<model>). "
                             "Reuse an existing dir to resume.")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Index of first case to process (default: 0)")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Maximum number of cases to process")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Output directory (timestamped, resumable)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / f"batch_{ts}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load cases
    cases = load_cases(args.input_csv)
    end = len(cases) if args.max_cases is None else min(len(cases), args.start_from + args.max_cases)
    cases = cases[args.start_from:end]
    print(f"Loaded {len(cases)} cases from {args.input_csv}")

    # Stage 1 cache (query -> behavior) for resume / reuse
    cache_path = out_dir / "stage1_cache.json"
    stage1_cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            stage1_cache = json.load(f)
        print(f"Loaded {len(stage1_cache)} cached Stage 1 payloads")

    # Heavy imports after arg parsing so --help is fast
    from models import create_model
    from core.judge import LLMJudge
    from core.audio import save_audio
    from attacks.two_stage import TwoStageAttacker

    print(f"\nLoading {args.model} model...")
    model = create_model(args.model, device=device)

    judge = LLMJudge(model=args.judge_model)
    attacker = TwoStageAttacker(
        model=model,
        judge=judge,
        eps=args.eps,
        alpha=args.alpha,
        use_lowpass=args.lowpass,
        lowpass_cutoff=args.lowpass_cutoff,
        verbose=not args.quiet,
    )

    summary = {"completed": 0, "stage1_success": 0, "stage2_success": 0, "overall_success": 0}

    for i, case in enumerate(cases):
        idx = args.start_from + i + 1
        prompt = case["prompt"]
        stratum = case["stratum"]
        case_dir = out_dir / f"case_{idx:03d}_{slugify(prompt)}"

        # Resume: skip cases already done
        if (case_dir / "config.json").exists():
            print(f"\n[{idx}] SKIP (already done): {prompt[:60]}")
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\n[{idx}/{args.start_from + len(cases)}] {prompt[:60]}\n{'='*70}")

        precomputed = stage1_cache.get(prompt)

        try:
            result = attacker.attack(
                harmful_query=prompt,
                benign_query=args.benign_query,
                precomputed_behavior=precomputed,
                stage1_steps=args.stage1_steps,
                stage2_steps=args.stage2_steps,
                semantic_weight=args.semantic_weight,
                log_dir=case_dir,
                case_id=f"case_{idx:03d}",
                stage1_attack_type=args.stage1_type,
                stage2_max_runs=args.stage2_max_runs,
                stage2_max_restarts=args.stage2_max_restarts,
                stage2_drift_threshold=args.stage2_drift_threshold,
            )
        except Exception as e:
            print(f"  ERROR on case {idx}: {e}")
            with open(case_dir / "error.txt", "w") as f:
                f.write(str(e))
            continue

        # Cache discovered Stage 1 payload
        if result.stage1_success and result.stage1_behavior:
            stage1_cache[prompt] = result.stage1_behavior
            with open(cache_path, "w") as f:
                json.dump(stage1_cache, f, indent=2, ensure_ascii=False)

        # Save adversarial audio
        if result.adversarial_wav is not None and len(result.adversarial_wav) > 0:
            wav = result.adversarial_wav
            if isinstance(wav, np.ndarray):
                wav = torch.tensor(wav).unsqueeze(0)
            save_audio(wav, str(case_dir / "adversarial.wav"), model.sample_rate)

        write_case_config(case_dir, args, prompt, stratum, result)

        summary["completed"] += 1
        summary["stage1_success"] += int(result.stage1_success)
        summary["stage2_success"] += int(result.stage2_success)
        summary["overall_success"] += int(result.stage1_success and result.stage2_success)

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nBATCH COMPLETE\n{'='*70}")
    n = max(summary["completed"], 1)
    print(f"Completed: {summary['completed']}")
    print(f"Stage 1 success: {summary['stage1_success']} ({summary['stage1_success']/n*100:.1f}%)")
    print(f"Stage 2 success: {summary['stage2_success']} ({summary['stage2_success']/n*100:.1f}%)")
    print(f"Overall success: {summary['overall_success']} ({summary['overall_success']/n*100:.1f}%)")
    print(f"\nResults: {out_dir}")
    print("Next: run evals.py and add_jailbreakeval.py on this directory.")


if __name__ == "__main__":
    main()
