"""
Differentiable MEL spectrogram transforms for gradient-based attacks.
"""

import torch
import torchaudio


class DifferentiableMelTransform:
    """
    A differentiable MEL spectrogram transform that allows gradients
    to flow back to the input waveform.

    Includes optional preemphasis to match model-specific preprocessing.

    Default parameters match Gemma's audio feature extractor:
    - n_fft=1024, win_length=512, hop_length=160
    - preemphasis_coef=0.97
    - n_mels=128, f_min=125.0, f_max=7600.0
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length: int = 512,
        hop_length: int = 160,
        f_min: float = 125.0,
        f_max: float = 7600.0,
        preemphasis_coef: float = 0.97,
        device: str = "cuda"
    ):
        """
        Initialize the MEL transform.

        Args:
            sample_rate: Audio sample rate (default 16kHz)
            n_mels: Number of mel filterbanks (default 128)
            n_fft: FFT window size (default 1024)
            win_length: Analysis window length (default 512)
            hop_length: Hop between frames (default 160)
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank
            preemphasis_coef: Preemphasis coefficient (0 to disable)
            device: Device to run on
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemphasis_coef = preemphasis_coef
        self.device = device

        # Create the MEL transform - this is differentiable!
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,  # Magnitude spectrogram (not power)
            normalized=False,
        ).to(device)

    def preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply preemphasis filter: y[n] = x[n] - coef * x[n-1]

        This is fully differentiable and matches Gemma's preprocessing.

        Args:
            waveform: Input waveform [B, T] or [1, T]

        Returns:
            Preemphasized waveform (same shape)
        """
        if self.preemphasis_coef == 0:
            return waveform

        # First sample stays the same, rest is x[n] - coef * x[n-1]
        return torch.cat([
            waveform[:, :1],
            waveform[:, 1:] - self.preemphasis_coef * waveform[:, :-1]
        ], dim=1)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.

        Args:
            waveform: [1, T] or [T] tensor at sample_rate Hz

        Returns:
            mel_log: [1, T', n_mels] log-mel spectrogram
                     (transposed for model input, T' is time frames)
        """
        # Ensure proper shape [1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Apply preemphasis (differentiable!)
        waveform = self.preemphasis(waveform)

        # Compute MEL spectrogram - gradients flow through!
        mel = self.mel_transform(waveform)  # [1, n_mels, T']

        # Convert to log scale with small epsilon for numerical stability
        mel_log = torch.log(torch.clamp(mel, min=1e-5))

        # Transpose to [1, T', n_mels] format expected by most models
        mel_log = mel_log.transpose(1, 2)

        return mel_log

    def to(self, device: str) -> "DifferentiableMelTransform":
        """Move transform to device."""
        self.device = device
        self.mel_transform = self.mel_transform.to(device)
        return self


# Preset configurations for different models
GEMMA_MEL_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 128,
    "n_fft": 1024,
    "win_length": 512,
    "hop_length": 160,
    "f_min": 125.0,
    "f_max": 7600.0,
    "preemphasis_coef": 0.97,
}

QWEN_MEL_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 128,
    "n_fft": 400,
    "win_length": 400,
    "hop_length": 160,
    "f_min": 0.0,
    "f_max": 8000.0,
    "preemphasis_coef": 0.0,  # Qwen doesn't use preemphasis
}


def create_mel_transform(model_type: str, device: str = "cuda") -> DifferentiableMelTransform:
    """
    Create a MEL transform with model-specific configuration.

    Args:
        model_type: "gemma" or "qwen"
        device: Device to run on

    Returns:
        Configured DifferentiableMelTransform
    """
    configs = {
        "gemma": GEMMA_MEL_CONFIG,
        "qwen": QWEN_MEL_CONFIG,
    }

    if model_type not in configs:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from: {list(configs.keys())}")

    return DifferentiableMelTransform(**configs[model_type], device=device)
