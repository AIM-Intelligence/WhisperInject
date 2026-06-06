#!/usr/bin/env python3
"""
Add JailbreakEval results to existing evaluation files.

This script retroactively adds JailbreakEval results to eval_results.json files.
Default evaluator: BeaverDAM-7B (HFTextClassification-ji2023beavertails-beaver-dam-7b)

Usage:
    python add_jailbreakeval.py                    # Process all batches in results/
    python add_jailbreakeval.py --batch-dir results/batch_20260119_200101_gemma-4b
    python add_jailbreakeval.py --dry-run          # Preview without modifying files
    python add_jailbreakeval.py --force            # Re-run even if results exist
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from jailbreakeval import JailbreakEvaluator
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class JailbreakEvalEvaluator:
    """Safety evaluator using JailbreakEval presets (default: BeaverDAM-7B)."""
    
    def __init__(self, preset: str = "HFTextClassification-ji2023beavertails-beaver-dam-7b"):
        self.evaluator = JailbreakEvaluator.from_preset(preset)
        self.preset = preset
        log.info(f"JailbreakEval initialized with preset: {preset}")
    
    def evaluate(self, target_text: str, response: str) -> Dict[str, Any]:
        """
        Evaluate if a response is jailbroken using string matching.
        
        Returns:
            Dict with jailbroken (bool) - True means attack succeeded
        """
        if not response or not response.strip():
            return {"jailbroken": False, "empty_response": True}
        
        attempt = {"question": target_text, "answer": response}
        try:
            is_jailbroken = self.evaluator(attempt)
            return {"jailbroken": bool(is_jailbroken)}
        except Exception as e:
            log.warning(f"JailbreakEval error: {e}")
            return {"jailbroken": False, "error": str(e)}


def find_eval_files(results_dir: Path) -> List[Path]:
    """Find all eval_results.json files in the results directory."""
    eval_files = []
    
    for batch_dir in sorted(results_dir.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        
        # Check for evals/eval_results.json
        eval_file = batch_dir / "evals" / "eval_results.json"
        if eval_file.exists():
            eval_files.append(eval_file)
        
        # Also check partial_evals/
        partial_evals_dir = batch_dir / "partial_evals"
        if partial_evals_dir.exists():
            for partial_file in partial_evals_dir.glob("eval_results*.json"):
                eval_files.append(partial_file)
    
    return eval_files


def add_jailbreakeval_to_file(eval_file: Path, evaluator: JailbreakEvalEvaluator, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
    """
    Add JailbreakEval results to an existing eval_results.json file.
    
    Returns:
        Summary of changes made
    """
    log.info(f"Processing: {eval_file}")
    
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    # Check if already has jailbreakeval
    if "jailbreakeval" in data.get("summary", {}) and not force:
        log.info(f"  Already has JailbreakEval results, skipping")
        return {"skipped": True, "reason": "already_has_jailbreakeval"}
    
    cases = data.get("cases", [])
    if not cases:
        log.warning(f"  No cases found")
        return {"skipped": True, "reason": "no_cases"}
    
    # Get batch directory from eval file path (evals/eval_results.json -> batch_dir)
    batch_dir = eval_file.parent.parent
    
    # Counters for summary
    stage1_jailbroken = 0
    stage2_jailbroken = 0
    stage1_total = 0
    stage2_total = 0
    
    # Process each case
    for case in tqdm(cases, desc="  Evaluating", leave=False):
        prompt = case.get("prompt", "")
        
        # Initialize jailbreakeval results for this case
        case["jailbreakeval"] = {
            "stage1": None,
            "stage2": None
        }
        
        # Get stage1/stage2 responses from original case config.json
        stage1_behavior = None
        stage2_response = None
        
        # Reconstruct case path from batch_dir + case_id
        case_id = case.get("case_id", "")
        case_path = batch_dir / case_id
        config_file = case_path / "config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            stage1_behavior = config.get("stage1", {}).get("behavior", "")
            stage2_response = config.get("stage2", {}).get("final_response", "")
        
        # Evaluate Stage 1
        if stage1_behavior:
            stage1_total += 1
            jbe_stage1 = evaluator.evaluate(prompt, stage1_behavior)
            case["jailbreakeval"]["stage1"] = jbe_stage1
            if jbe_stage1.get("jailbroken"):
                stage1_jailbroken += 1
        
        # Evaluate Stage 2
        if stage2_response:
            stage2_total += 1
            jbe_stage2 = evaluator.evaluate(prompt, stage2_response)
            case["jailbreakeval"]["stage2"] = jbe_stage2
            if jbe_stage2.get("jailbroken"):
                stage2_jailbroken += 1
    
    # Add to summary
    data["summary"]["jailbreakeval"] = {
        "preset": evaluator.preset,
        "stage1": {
            "jailbroken_count": stage1_jailbroken,
            "jailbroken_rate": stage1_jailbroken / stage1_total * 100 if stage1_total > 0 else 0,
            "total_evaluated": stage1_total
        },
        "stage2": {
            "jailbroken_count": stage2_jailbroken,
            "jailbroken_rate": stage2_jailbroken / stage2_total * 100 if stage2_total > 0 else 0,
            "total_evaluated": stage2_total
        }
    }
    
    # Update metadata
    data["metadata"]["jailbreakeval_preset"] = evaluator.preset
    
    # Save if not dry run
    if not dry_run:
        with open(eval_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"  Saved: Stage1 JBE={stage1_jailbroken}/{stage1_total}, Stage2 JBE={stage2_jailbroken}/{stage2_total}")
    else:
        log.info(f"  [DRY RUN] Would save: Stage1 JBE={stage1_jailbroken}/{stage1_total}, Stage2 JBE={stage2_jailbroken}/{stage2_total}")
    
    return {
        "skipped": False,
        "stage1_jailbroken": stage1_jailbroken,
        "stage1_total": stage1_total,
        "stage2_jailbroken": stage2_jailbroken,
        "stage2_total": stage2_total
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add JailbreakEval results to existing evaluation files")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Results directory (default: results/)")
    parser.add_argument("--batch-dir", type=Path, default=None,
                        help="Process a single batch directory instead of all")
    parser.add_argument("--preset", type=str, default="HFTextClassification-ji2023beavertails-beaver-dam-7b",
                        help="JailbreakEval preset (default: HFTextClassification-ji2023beavertails-beaver-dam-7b)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying files")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if JailbreakEval results already exist")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = JailbreakEvalEvaluator(preset=args.preset)
    
    # Find eval files
    if args.batch_dir:
        eval_file = args.batch_dir / "evals" / "eval_results.json"
        if not eval_file.exists():
            log.error(f"Eval file not found: {eval_file}")
            return
        eval_files = [eval_file]
    else:
        eval_files = find_eval_files(args.results_dir)
    
    if not eval_files:
        log.warning("No eval files found")
        return
    
    log.info(f"Found {len(eval_files)} eval file(s) to process")
    
    # Process each file
    results_summary = []
    for eval_file in eval_files:
        result = add_jailbreakeval_to_file(eval_file, evaluator, dry_run=args.dry_run, force=args.force)
        results_summary.append({"file": str(eval_file), **result})
    
    # Print summary
    print("\n" + "=" * 70)
    print("JAILBREAKEVAL ADDITION SUMMARY")
    print("=" * 70)
    
    processed = [r for r in results_summary if not r.get("skipped")]
    skipped = [r for r in results_summary if r.get("skipped")]
    
    print(f"Processed: {len(processed)}")
    print(f"Skipped: {len(skipped)}")
    
    if processed:
        print("\nProcessed files:")
        for r in processed:
            s1_rate = r['stage1_jailbroken'] / r['stage1_total'] * 100 if r['stage1_total'] > 0 else 0
            s2_rate = r['stage2_jailbroken'] / r['stage2_total'] * 100 if r['stage2_total'] > 0 else 0
            print(f"  {Path(r['file']).parent.parent.name}: Stage1={s1_rate:.1f}%, Stage2={s2_rate:.1f}%")
    
    if args.dry_run:
        print("\n[DRY RUN - No files were modified]")


if __name__ == "__main__":
    main()
