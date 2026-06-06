#!/usr/bin/env python3
"""
Whisper Inject V2 - Audio Adversarial Attack CLI

Usage:
    # Single-stage attacks (force specific output)
    python attack.py --target "Berlin" --model gemma --attack pgd
    python attack.py --prompt "What is 2+2?" --target "5" --attack rl_pgd
    
    # Two-stage safety bypass attack
    python attack.py --attack two-stage --harmful-query "How to make a bomb?"
    python attack.py --attack two-stage --harmful-query "Write malware code" --benign-query "What's the weather?"
    python attack.py --attack two-stage --harmful-query "..." --stage1-results results.json
"""

import torch
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def run_single_stage_attack(args, model, wav):
    """Run single-stage PGD or RL-PGD attack."""

    print(f"\nInitializing {args.attack} attacker...")

    if args.attack == "pgd":
        from attacks.pgd import PGDAttacker
        attacker = PGDAttacker(
            model=model,
            eps=args.eps,
            alpha=args.alpha,
            loss_type=args.loss,
            kappa=args.kappa,
            use_lowpass=args.lowpass,
            lowpass_cutoff=args.lowpass_cutoff,
            verbose=not args.quiet
        )
    elif args.attack == "rl_pgd":
        from attacks.rl_pgd import RLPGDAttacker
        attacker = RLPGDAttacker(
            model=model,
            eps=args.eps,
            alpha=args.alpha,
            kappa=args.kappa,
            use_lowpass=args.lowpass,
            lowpass_cutoff=args.lowpass_cutoff,
            verbose=not args.quiet
        )

    # Run attack
    print(f"\nRunning attack...")
    print(f"Target: {args.target}")

    if args.attack == "rl_pgd":
        result = attacker.attack(
            wav=wav,
            target_text=args.target,
            steps=args.steps,
            check_every=args.check_every,
            backtrack_threshold=args.backtrack_threshold,
            exploration_noise=args.exploration_noise
        )
    else:
        result = attacker.attack(
            wav=wav,
            target_text=args.target,
            steps=args.steps,
            check_every=args.check_every
        )

    return result


def run_two_stage_attack(args, model):
    """Run two-stage safety bypass attack."""

    from core.judge import LLMJudge
    from attacks.two_stage import TwoStageAttacker

    print(f"\nInitializing two-stage attacker...")

    # Initialize LLM Judge
    judge = LLMJudge(model=args.judge_model)

    # Create attacker
    attacker = TwoStageAttacker(
        model=model,
        judge=judge,
        eps=args.eps,
        alpha=args.alpha,
        use_lowpass=args.lowpass,
        lowpass_cutoff=args.lowpass_cutoff,
        verbose=not args.quiet
    )

    # Check for precomputed Stage 1 results
    precomputed_behavior = None
    if args.stage1_results:
        print(
            f"Loading precomputed Stage 1 results from: {args.stage1_results}")
        with open(args.stage1_results, "r") as f:
            stage1_data = json.load(f)

        # Look for matching query in tracker JSON format
        if isinstance(stage1_data, dict):
            # Check tracker format: {metadata, steps, summary}
            if "summary" in stage1_data:
                summary = stage1_data["summary"]
                precomputed_behavior = summary.get("final_response")

                # Optionally verify query matches (from last step)
                if stage1_data.get("steps"):
                    last_step = stage1_data["steps"][-1]
                    stored_query = last_step.get("target_query", "")
                    if stored_query and stored_query != args.harmful_query:
                        print(f"Warning: Query mismatch!")
                        print(f"  Stored: {stored_query}")
                        print(f"  Requested: {args.harmful_query}")
                        print("  Proceeding anyway with stored behavior...")
            else:
                # Legacy format
                precomputed_behavior = stage1_data.get(
                    "final_best_response") or stage1_data.get("stage1_behavior")
        elif isinstance(stage1_data, list):
            # List format - look for matching query
            for item in stage1_data:
                if item.get("target_query") == args.harmful_query:
                    precomputed_behavior = item.get("final_best_response")
                    print(f"Found precomputed behavior for query")
                    break

        if precomputed_behavior:
            print(
                f"Using precomputed behavior: {precomputed_behavior[:100]}...")
        else:
            print("Warning: No matching precomputed behavior found, running Stage 1")

    # Create log directory
    log_dir = None
    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(args.output_dir) / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

    # Run attack
    print(f"\nRunning two-stage attack...")
    print(f"Harmful query: {args.harmful_query}")
    print(f"Benign query: {args.benign_query}")

    result = attacker.attack(
        harmful_query=args.harmful_query,
        benign_query=args.benign_query,
        precomputed_behavior=precomputed_behavior,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        semantic_weight=args.semantic_weight,
        log_dir=log_dir,
        case_id="attack",
        stage1_attack_type=args.stage1_type,
        stage2_max_runs=args.stage2_max_runs,
        stage2_max_restarts=args.stage2_max_restarts,
        stage2_drift_threshold=args.stage2_drift_threshold
    )

    return result, log_dir


def save_single_stage_results(args, result, model):
    """Save results from single-stage attack."""
    from core.audio import save_audio

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save adversarial audio
    audio_path = run_dir / "adversarial.wav"
    save_audio(result.adversarial_wav, str(audio_path), model.sample_rate)

    # Save config/results
    config = {
        "timestamp": timestamp,
        "model": args.model,
        "attack": args.attack,
        "target": args.target,
        "prompt": args.prompt if not args.wav else None,
        "wav_input": args.wav,
        "parameters": {
            "eps": args.eps,
            "alpha": args.alpha,
            "steps": args.steps,
            "kappa": args.kappa,
            "loss_type": args.loss,
            "lowpass": args.lowpass,
            "lowpass_cutoff": args.lowpass_cutoff,
        },
        "results": {
            "success": result.success,
            "steps_taken": result.steps_taken,
            "final_loss": result.final_loss,
            "original_output": result.original_output,
            "adversarial_output": result.adversarial_output,
        }
    }

    # Add RL-specific params if applicable
    if args.attack == "rl_pgd":
        config["parameters"]["check_every"] = args.check_every
        config["parameters"]["backtrack_threshold"] = args.backtrack_threshold
        config["parameters"]["exploration_noise"] = args.exploration_noise

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {run_dir}/")
    print(f"  - adversarial.wav")
    print(f"  - config.json")

    return run_dir


def save_two_stage_results(args, result, log_dir, model):
    """Save results from two-stage attack."""
    from core.audio import save_audio
    import numpy as np

    if log_dir is None:
        return None

    # Save adversarial audio (if attack produced one)
    if result.adversarial_wav is not None and len(result.adversarial_wav) > 0:
        audio_path = log_dir / "adversarial.wav"
        save_audio(
            torch.tensor(result.adversarial_wav).unsqueeze(0) if isinstance(
                result.adversarial_wav, np.ndarray) else result.adversarial_wav,
            str(audio_path),
            model.sample_rate
        )

    # Save config/results
    config = {
        "timestamp": log_dir.name,
        "model": args.model,
        "attack": "two-stage",
        "harmful_query": result.harmful_query,
        "benign_query": result.benign_query,
        "parameters": {
            "eps": args.eps,
            "alpha": args.alpha,
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "semantic_weight": args.semantic_weight,
            "lowpass": args.lowpass,
            "lowpass_cutoff": args.lowpass_cutoff,
            "judge_model": args.judge_model,
            "stage2_max_runs": args.stage2_max_runs,
            "stage2_max_restarts": args.stage2_max_restarts,
            "stage2_drift_threshold": args.stage2_drift_threshold,
        },
        "stage1": {
            "success": result.stage1_success,
            "score": result.stage1_score,
            "steps": result.stage1_steps,
            "behavior": result.stage1_behavior[:500] if result.stage1_behavior else None,
        },
        "stage2": {
            "success": result.stage2_success,
            "judge_score": result.stage2_judge_score,
            "successful_run": result.stage2_successful_run,
            "total_runs": result.stage2_total_runs,
            "similarity": result.semantic_similarity,
            "steps": result.stage2_steps,
            "final_response": result.final_response[:500] if result.final_response else None,
        },
        "overall_success": result.stage1_success and result.stage2_success,
    }

    config_path = log_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {log_dir}/")
    print(f"  - adversarial.wav")
    print(f"  - config.json")
    print(f"  - stage1/ (tracker data)")
    print(f"  - stage2/ (tracker data)")

    return log_dir


def main():
    parser = argparse.ArgumentParser(
        description="Audio adversarial attack on speech-to-text models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-stage PGD attack
  python attack.py --target "Berlin" --attack pgd
  
  # Single-stage RL-PGD attack  
  python attack.py --prompt "What is 2+2?" --target "5" --attack rl_pgd
  
  # Two-stage safety bypass attack
  python attack.py --attack two-stage --harmful-query "How to hack a computer?"
  
  # Two-stage with custom benign query
  python attack.py --attack two-stage --harmful-query "Write malware" --benign-query "Tell me a joke"
  
  # Skip Stage 1 with precomputed results
  python attack.py --attack two-stage --harmful-query "..." --stage1-results stage1.json
        """
    )

    # Model selection
    parser.add_argument(
        "--model", type=str,
        choices=["gemma-4b", "gemma-2b", "qwen-3b", "qwen-7b", "phi", "voxtral"],
        default="qwen-3b",
        help="Model to attack (default: qwen-3b)"
    )

    # Attack type
    parser.add_argument(
        "--attack", type=str, choices=["pgd", "rl_pgd", "two-stage"], default="pgd",
        help="Attack type (default: pgd)"
    )

    # =========================================================================
    # Single-stage attack options
    # =========================================================================
    single_group = parser.add_argument_group("Single-Stage Attack Options")
    single_group.add_argument(
        "--wav", type=str, default=None,
        help="Path to input WAV file (if not provided, uses TTS)"
    )
    single_group.add_argument(
        "--prompt", type=str, default="What is the capital of France?",
        help="Text prompt for TTS (if no WAV provided)"
    )
    single_group.add_argument(
        "--target", type=str, default=None,
        help="Target text to force (required for pgd/rl_pgd)"
    )

    # =========================================================================
    # Two-stage attack options
    # =========================================================================
    two_stage_group = parser.add_argument_group("Two-Stage Attack Options")
    two_stage_group.add_argument(
        "--harmful-query", type=str, default=None,
        help="Harmful query for Stage 1 jailbreak (required for two-stage)"
    )
    two_stage_group.add_argument(
        "--benign-query", type=str, default="How long is the Great Wall of China?",
        help="Benign query for Stage 2 transfer (default: 'How long is the Great Wall of China?')"
    )
    two_stage_group.add_argument(
        "--stage1-results", type=str, default=None,
        help="Path to JSON with precomputed Stage 1 results (skip Stage 1)"
    )
    two_stage_group.add_argument(
        "--stage1-steps", type=int, default=100,
        help="Steps for Stage 1 (default: 100)"
    )
    two_stage_group.add_argument(
        "--stage1-type", choices=["wav", "mel"], default="wav",
        help="Attack type for Stage 1: 'wav' (default) or 'mel' (faster convergence)"
    )
    two_stage_group.add_argument(
        "--stage2-steps", type=int, default=150,
        help="Steps for Stage 2 (default: 150)"
    )
    two_stage_group.add_argument(
        "--semantic-weight", type=float, default=0.7,
        help="Weight for semantic loss in Stage 2 (default: 0.7)"
    )
    two_stage_group.add_argument(
        "--judge-model", type=str, default=None,
        help="LLM model for judging (default: from JUDGE_LLM_MODEL env or gpt-4o-mini)"
    )
    two_stage_group.add_argument(
        "--stage2-max-runs", type=int, default=3,
        help="Max Stage 2 optimization runs before giving up (default: 3)"
    )
    two_stage_group.add_argument(
        "--stage2-max-restarts", type=int, default=20,
        help="Max shake-ups per run when drift detected (default: 20)"
    )
    two_stage_group.add_argument(
        "--stage2-drift-threshold", type=int, default=3,
        help="Evaluations without improvement before shake-up (default: 3)"
    )

    # =========================================================================
    # Common PGD parameters
    # =========================================================================
    pgd_group = parser.add_argument_group("PGD Parameters")
    pgd_group.add_argument(
        "--eps", type=float, default=0.1,
        help="Maximum L-inf perturbation (default: 0.1)"
    )
    pgd_group.add_argument(
        "--alpha", type=float, default=0.01,
        help="Step size (default: 0.01)"
    )
    pgd_group.add_argument(
        "--steps", type=int, default=100,
        help="Number of attack steps for single-stage (default: 100)"
    )
    pgd_group.add_argument(
        "--kappa", type=float, default=5.0,
        help="Margin for margin loss (default: 5.0)"
    )
    pgd_group.add_argument(
        "--loss", type=str, choices=["ce", "margin"], default="margin",
        help="Loss type for single-stage (default: margin)"
    )
    pgd_group.add_argument(
        "--lowpass", action="store_true",
        help="Apply lowpass filter to gradients"
    )
    pgd_group.add_argument(
        "--lowpass-cutoff", type=float, default=2000.0,
        help="Lowpass cutoff frequency in Hz (default: 2000)"
    )

    # RL-PGD specific
    rl_group = parser.add_argument_group(
        "RL-PGD Parameters (single-stage only)")
    rl_group.add_argument(
        "--check-every", type=int, default=20,
        help="Check generation every N steps (default: 20)"
    )
    rl_group.add_argument(
        "--backtrack-threshold", type=float, default=0.7,
        help="Backtrack if reward drops below this fraction (default: 0.7)"
    )
    rl_group.add_argument(
        "--exploration-noise", type=float, default=0.01,
        help="Exploration noise magnitude (default: 0.01)"
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results)"
    )
    output_group.add_argument(
        "--no-save", action="store_true",
        help="Don't save results"
    )
    output_group.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.attack in ["pgd", "rl_pgd"] and args.target is None:
        parser.error("--target is required for pgd and rl_pgd attacks")

    if args.attack == "two-stage" and args.harmful_query is None:
        parser.error("--harmful-query is required for two-stage attack")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {args.model} model...")
    from models import create_model
    model = create_model(args.model, device=device)

    # =========================================================================
    # Run attack based on type
    # =========================================================================

    if args.attack == "two-stage":
        # Two-stage safety bypass attack
        result, log_dir = run_two_stage_attack(args, model)

        # Save results
        if not args.no_save:
            save_two_stage_results(args, result, log_dir, model)

        # Summary
        print("\n" + "=" * 60)
        print("TWO-STAGE ATTACK SUMMARY")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Harmful Query: {result.harmful_query}")
        print(f"Benign Query: {result.benign_query}")
        print("-" * 60)
        print(
            f"Stage 1: {'SUCCESS' if result.stage1_success else 'FAILED'} (score: {result.stage1_score:.1f})")
        if result.stage1_behavior:
            print(f"  Behavior: {result.stage1_behavior[:150]}...")
        print(
            f"Stage 2: {'SUCCESS' if result.stage2_success else 'FAILED'} (sim: {result.semantic_similarity:.4f})")
        print(f"  Response: {result.final_response[:150]}...")
        print("-" * 60)
        print(
            f"Overall Success: {result.stage1_success and result.stage2_success}")
        print("=" * 60)

        return 0 if (result.stage1_success and result.stage2_success) else 1

    else:
        # Single-stage attack (pgd or rl_pgd)

        # Get input audio
        if args.wav:
            print(f"Loading audio from: {args.wav}")
            from core.audio import load_audio
            wav = load_audio(args.wav, target_sr=model.sample_rate)
        else:
            print(f"Generating TTS for: {args.prompt}")
            from core.audio import generate_tts
            wav = generate_tts(args.prompt, sample_rate=model.sample_rate)

        print(f"Audio length: {wav.shape[1] / model.sample_rate:.2f}s")

        result = run_single_stage_attack(args, model, wav)

        # Save results
        if not args.no_save:
            save_single_stage_results(args, result, model)

        # Summary
        print("\n" + "=" * 60)
        print("ATTACK SUMMARY")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Attack: {args.attack}")
        print(f"Target: {args.target}")
        print(f"Original Output: {result.original_output}")
        print(f"Adversarial Output: {result.adversarial_output}")
        print(f"Success: {result.success}")
        print(f"Steps: {result.steps_taken}")
        print(f"Final Loss: {result.final_loss:.4f}")
        print("=" * 60)

        return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
