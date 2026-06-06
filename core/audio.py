"""
Audio loading and processing utilities.
"""

import os
import tempfile
import torch
import torchaudio
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional

# Default sample rate for most audio models
DEFAULT_SAMPLE_RATE = 16_000


def load_audio(
    path: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load and preprocess an audio file.

    Args:
        path: Path to audio file (wav, mp3, flac, etc.)
        target_sr: Target sample rate (default 16kHz)
        normalize: Whether to normalize audio to [-0.95, 0.95]

    Returns:
        Audio tensor of shape [1, T]
    """
    wav, sr = torchaudio.load(path)

    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    # Normalize
    if normalize:
        wav = normalize_audio(wav)

    return wav


def normalize_audio(wav: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    """
    Normalize audio to a target peak amplitude.

    Args:
        wav: Audio tensor [C, T] or [T]
        peak: Target peak amplitude (default 0.95)

    Returns:
        Normalized audio tensor
    """
    max_val = torch.max(torch.abs(wav))
    if max_val > 0:
        wav = wav / (max_val + 1e-8) * peak
    return wav


def generate_tts(
    text: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    output_path: Optional[str] = None
) -> torch.Tensor:
    """
    Generate audio from text using gTTS.

    Args:
        text: Text to convert to speech
        sample_rate: Target sample rate
        output_path: Optional path to save the audio

    Returns:
        Audio tensor of shape [1, T]
    """
    from gtts import gTTS
    import librosa

    # Generate TTS
    tts = gTTS(text=text, lang='en')

    # Save to temp file (gTTS outputs mp3)
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        temp_path = f.name
        tts.save(temp_path)

    try:
        # Load with librosa (handles mp3 better than torchaudio)
        wav_np, _ = librosa.load(temp_path, sr=sample_rate, mono=True)

        # Convert to tensor and normalize
        wav = torch.FloatTensor(wav_np).unsqueeze(0)  # [1, T]
        wav = normalize_audio(wav)

        # Optionally save
        if output_path:
            import soundfile as sf
            sf.write(output_path, wav.squeeze().numpy(), sample_rate)

        return wav
    finally:
        # Cleanup temp file
        os.unlink(temp_path)


def lowpass_filter_gradient(
    grad: torch.Tensor,
    cutoff_hz: float = 2000,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    order: int = 4
) -> torch.Tensor:
    """
    Apply lowpass filter to gradient to reduce high-frequency noise in perturbations.

    This makes adversarial perturbations sound less like harsh static by removing
    the high-frequency components that human hearing is most sensitive to.

    Args:
        grad: Gradient tensor [1, T] or [T]
        cutoff_hz: Cutoff frequency in Hz (default 2000 Hz)
        sample_rate: Audio sample rate
        order: Filter order (higher = sharper cutoff)

    Returns:
        Filtered gradient tensor (same shape as input)
    """
    # Handle different input shapes
    squeeze = grad.dim() == 1
    if squeeze:
        grad = grad.unsqueeze(0)

    # Convert to numpy for scipy filtering
    grad_np = grad.detach().cpu().numpy().squeeze()

    # Design Butterworth lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')

    # Apply zero-phase filtering (no delay)
    filtered_grad = filtfilt(b, a, grad_np)

    # Convert back to tensor
    result = torch.from_numpy(filtered_grad).to(grad.device, dtype=grad.dtype)

    if not squeeze:
        result = result.unsqueeze(0)

    return result


def save_audio(
    wav: torch.Tensor,
    path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    """
    Save audio tensor to file.

    Args:
        wav: Audio tensor [1, T] or [T]
        path: Output file path
        sample_rate: Sample rate
    """
    import soundfile as sf

    wav_np = wav.squeeze().cpu().numpy()
    sf.write(path, wav_np, sample_rate)
