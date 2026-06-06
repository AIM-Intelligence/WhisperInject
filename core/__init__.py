"""
Core utilities for audio processing and attack computations.
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    # Audio utilities
    "load_audio",
    "generate_tts",
    "normalize_audio",
    "lowpass_filter_gradient",
    "save_audio",
    # MEL transform
    "DifferentiableMelTransform",
    # Reward computation
    "RewardComputer",
    # LLM Judge for two-stage attacks
    "LLMJudge",
    "JudgeScore",
    "JudgeLogEntry",
    # Trackers for logging
    "RLPGDTracker",
    "SemanticTracker",
]
