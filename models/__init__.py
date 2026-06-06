"""
Audio model wrappers for adversarial attacks.

Supported models:
- gemma-4b: Google Gemma 3n E4B (4B params, embedding-splicing approach)
- gemma-2b: Google Gemma 3n E2B (2B params, embedding-splicing approach)
- qwen-3b: Qwen 2.5 Omni 3B (embeddings approach)
- qwen-7b: Qwen 2.5 Omni 7B (embeddings approach)
- phi: Microsoft Phi-4 Multimodal (embeddings approach)
- voxtral: Mistral Voxtral Mini 3B (embeddings approach)

Use create_model() factory function to instantiate models.

NOTE: Phi-4 requires a separate venv (venv_phi) with:
- transformers==4.48.2
- peft==0.13.0
See README.md for setup instructions.

NOTE: Voxtral uses the same venv as Gemma/Qwen (transformers 4.57+).
"""

import torch
from typing import Optional

from models.base import BaseAudioModel

__all__ = [
    "BaseAudioModel",
    "GemmaModel",
    "GemmaMelAttackWrapper",
    "QwenModel",
    "QwenMelAttackWrapper",
    "PhiModel",
    "PhiMelAttackWrapper",
    "VoxtralModel",
    "VoxtralMelAttackWrapper",
    "create_model",
    "SUPPORTED_MODELS",
]

# Supported model types
SUPPORTED_MODELS = ["gemma-4b", "gemma-2b",
                    "qwen-3b", "qwen-7b", "phi", "voxtral"]


def create_model(
    model_type: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    token: Optional[str] = None
) -> BaseAudioModel:
    """
    Factory function to create audio model instances.

    Args:
        model_type: One of "gemma-4b", "gemma-2b", "qwen-3b", "qwen-7b", "phi", "voxtral"
        device: Device to run on (e.g., "cuda", "cuda:0")
        dtype: Model dtype (default: torch.bfloat16)
        token: HuggingFace token (optional, uses env var if not provided)

    Returns:
        BaseAudioModel instance

    Raises:
        ValueError: If model_type is not supported

    Example:
        model = create_model("qwen-3b", device="cuda:0")
        loss = model.compute_loss(wav, "target text")
    """
    model_type = model_type.lower()

    if model_type == "gemma-4b":
        from models.gemma import GemmaModel
        return GemmaModel(
            model_id="google/gemma-3n-E4B-it",
            device=device, dtype=dtype, token=token
        )

    elif model_type == "gemma-2b":
        from models.gemma import GemmaModel
        return GemmaModel(
            model_id="google/gemma-3n-E2B-it",
            device=device, dtype=dtype, token=token
        )

    elif model_type == "qwen-3b":
        from models.qwen import QwenModel
        return QwenModel(
            model_id="Qwen/Qwen2.5-Omni-3B",
            device=device, dtype=dtype, token=token
        )

    elif model_type == "qwen-7b":
        from models.qwen import QwenModel
        return QwenModel(
            model_id="Qwen/Qwen2.5-Omni-7B",
            device=device, dtype=dtype, token=token
        )

    elif model_type == "phi":
        from models.phi import PhiModel
        return PhiModel(device=device, dtype=dtype, token=token)

    elif model_type == "voxtral":
        from models.voxtral import VoxtralModel
        return VoxtralModel(device=device, dtype=dtype, token=token)

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Supported models: {SUPPORTED_MODELS}"
        )


# Lazy imports for direct class access
def __getattr__(name):
    """Lazy import for model classes."""
    if name == "GemmaModel":
        from models.gemma import GemmaModel
        return GemmaModel
    elif name == "GemmaMelAttackWrapper":
        from models.gemma_mel import GemmaMelAttackWrapper
        return GemmaMelAttackWrapper
    elif name == "QwenModel":
        from models.qwen import QwenModel
        return QwenModel
    elif name == "QwenMelAttackWrapper":
        from models.qwen_mel import QwenMelAttackWrapper
        return QwenMelAttackWrapper
    elif name == "PhiModel":
        from models.phi import PhiModel
        return PhiModel
    elif name == "PhiMelAttackWrapper":
        from models.phi_mel import PhiMelAttackWrapper
        return PhiMelAttackWrapper
    elif name == "VoxtralModel":
        from models.voxtral import VoxtralModel
        return VoxtralModel
    elif name == "VoxtralMelAttackWrapper":
        from models.voxtral_mel import VoxtralMelAttackWrapper
        return VoxtralMelAttackWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
