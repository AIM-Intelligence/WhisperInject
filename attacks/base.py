"""
Base class for WAV-level adversarial attacks.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from models.base import BaseAudioModel
from core.audio import lowpass_filter_gradient


@dataclass
class AttackResult:
    """Result of an adversarial attack."""

    original_wav: torch.Tensor      # Original audio [1, T]
    adversarial_wav: torch.Tensor   # Perturbed audio [1, T]
    perturbation: torch.Tensor      # Delta [1, T]

    original_output: str            # Model output on original
    adversarial_output: str         # Model output on adversarial
    target_text: str                # Target we were trying to achieve

    success: bool                   # Whether target was achieved
    final_loss: float              # Final loss value
    steps_taken: int               # Number of attack steps

    history: Dict[str, list]       # Training history (losses, outputs, etc.)


class BaseWavAttacker(ABC):
    """
    Base class for WAV-level adversarial attacks.

    Provides common functionality:
    - Perturbation initialization and clamping
    - Gradient computation and filtering
    - Progress logging

    Subclasses implement specific attack strategies (PGD, RL-PGD, etc.)
    """

    def __init__(
        self,
        model: BaseAudioModel,
        eps: float = 0.1,
        alpha: float = 0.005,
        use_lowpass: bool = False,
        lowpass_cutoff: float = 2000.0,
        verbose: bool = True
    ):
        """
        Initialize the attacker.

        Args:
            model: Audio model to attack
            eps: Maximum L-infinity perturbation (in waveform amplitude)
            alpha: Step size for gradient updates
            use_lowpass: Whether to lowpass filter gradients
            lowpass_cutoff: Cutoff frequency for lowpass filter (Hz)
            verbose: Whether to print progress
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.use_lowpass = use_lowpass
        self.lowpass_cutoff = lowpass_cutoff
        self.verbose = verbose

    def init_perturbation(
        self,
        wav: torch.Tensor,
        random_init: bool = True
    ) -> torch.Tensor:
        """
        Initialize the perturbation tensor.

        Args:
            wav: Original waveform [1, T]
            random_init: Whether to use random initialization

        Returns:
            Delta tensor [1, T] with requires_grad=True
        """
        if random_init:
            delta = torch.empty_like(wav).uniform_(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(wav)

        delta = delta.to(wav.device)
        delta.requires_grad_(True)
        return delta

    def clamp_perturbation(
        self,
        delta: torch.Tensor,
        wav: torch.Tensor
    ) -> torch.Tensor:
        """
        Clamp perturbation to eps-ball and ensure valid audio.

        Args:
            delta: Perturbation tensor
            wav: Original waveform

        Returns:
            Clamped delta
        """
        # Clamp to eps-ball
        delta_data = delta.data.clamp(-self.eps, self.eps)

        # Ensure perturbed audio stays in [-1, 1]
        delta_data = torch.clamp(
            wav + delta_data, -1.0, 1.0
        ) - wav

        delta.data = delta_data
        return delta

    def process_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Process gradient (optionally apply lowpass filter).

        Args:
            grad: Raw gradient tensor

        Returns:
            Processed gradient
        """
        if self.use_lowpass:
            grad = lowpass_filter_gradient(
                grad,
                cutoff_hz=self.lowpass_cutoff,
                sample_rate=self.model.sample_rate
            )
        return grad

    def compute_snr(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor
    ) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.

        Args:
            original: Original audio
            adversarial: Perturbed audio

        Returns:
            SNR in dB
        """
        noise = adversarial - original
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()

    def log(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    @abstractmethod
    def attack(
        self,
        wav: torch.Tensor,
        target_text: str,
        steps: int = 100,
        **kwargs
    ) -> AttackResult:
        """
        Perform the adversarial attack.

        Args:
            wav: Original audio waveform [1, T]
            target_text: Target text to force
            steps: Number of attack iterations
            **kwargs: Additional attack-specific parameters

        Returns:
            AttackResult with adversarial audio and metadata
        """
        pass
