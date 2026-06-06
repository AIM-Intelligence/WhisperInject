"""
Abstract base class for audio models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class BaseAudioModel(ABC):
    """
    Abstract interface for audio-to-text models.

    Required methods (must implement):
    - compute_loss(): Compute loss for target text given audio (differentiable)
    - generate(): Generate text from audio
    - sample_rate, device, dtype properties

    Optional methods (for models that use embedding-splicing approach):
    - wav_to_embeddings(): Convert waveform to embeddings
    - create_attack_inputs(): Create inputs from embeddings
    - _forward(): Forward pass through model

    Models like Gemma use the embedding-splicing approach and implement all methods.
    Models like Qwen take MEL directly and only need compute_loss/generate.

    This abstraction allows attack code to work with any audio model.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Required sample rate for this model (e.g., 16000)."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Device the model is on (e.g., 'cuda')."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Model dtype (e.g., torch.bfloat16)."""
        pass

    def wav_to_embeddings(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to audio embeddings (differentiable).

        This is OPTIONAL - only needed for models that splice audio embeddings
        into text embeddings (e.g., Gemma).

        Models that take audio features directly (e.g., Qwen) should override
        compute_loss() and compute_margin_loss() instead.

        Args:
            wav: Audio waveform tensor [1, T]

        Returns:
            Audio embeddings tensor [1, T', D]
            where T' is the number of audio tokens and D is embedding dim
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement wav_to_embeddings(). "
            "This is optional - only needed for embedding-splicing models like Gemma."
        )

    def create_attack_inputs(
        self,
        audio_embeddings: torch.Tensor,
        target_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create inputs for computing attack loss.

        This is OPTIONAL - only needed for models that splice audio embeddings
        into text embeddings (e.g., Gemma).

        Args:
            audio_embeddings: Output from wav_to_embeddings [1, T', D]
            target_text: The text we want the model to output

        Returns:
            full_embeddings: Combined prompt embeddings [1, seq_len, D]
            attention_mask: Attention mask [1, seq_len]
            labels: Token labels for loss (-100 for masked positions)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_attack_inputs(). "
            "This is optional - only needed for embedding-splicing models like Gemma."
        )

    @abstractmethod
    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate text from audio.

        Args:
            wav: Audio waveform tensor [1, T]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy)

        Returns:
            Generated text string
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for target text.

        This is the PRIMARY method for gradient-based attacks.
        Must be differentiable with respect to wav.

        Args:
            wav: Audio waveform tensor [1, T]
            target_text: Target text

        Returns:
            Loss tensor (scalar, with grad_fn for backprop)
        """
        pass

    def compute_margin_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        kappa: float = 5.0,
        early_weight: float = 5.0
    ) -> torch.Tensor:
        """
        Compute margin loss (Carlini-Wagner style).

        Margin loss: max(0, max_other_logit - target_logit + kappa)

        This loss pushes the target token to be the most probable by margin kappa.

        Default implementation uses wav_to_embeddings/create_attack_inputs/_forward.
        Models that don't use embedding-splicing should override this method.

        Args:
            wav: Audio waveform tensor [1, T]
            target_text: Target text
            kappa: Margin (higher = more confident target)
            early_weight: Extra weight for first few tokens

        Returns:
            Loss tensor (scalar)
        """
        # Default implementation for embedding-splicing models (Gemma)
        # Models like Qwen should override this entirely
        audio_emb = self.wav_to_embeddings(wav)
        full_emb, attn_mask, labels = self.create_attack_inputs(
            audio_emb, target_text)

        # Forward pass
        outputs = self._forward(full_emb, attn_mask, labels)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Find valid (non-masked) positions
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return outputs.loss

        valid_positions = valid_mask[0].nonzero(as_tuple=True)[0]
        valid_logits = shift_logits[0, valid_positions]
        valid_labels = shift_labels[0, valid_positions]

        # Get target token logits
        target_logits = valid_logits.gather(
            1, valid_labels.unsqueeze(1)).squeeze(1)

        # Get max non-target logits
        label_mask = torch.ones_like(valid_logits, dtype=torch.bool)
        label_mask.scatter_(1, valid_labels.unsqueeze(1), False)
        masked_logits = valid_logits.masked_fill(~label_mask, float('-inf'))
        top_other_logits = masked_logits.max(dim=-1).values

        # Margin loss
        margin_losses = torch.clamp(
            top_other_logits - target_logits + kappa, min=0)

        # Weight early tokens more heavily
        num_tokens = len(margin_losses)
        weights = torch.ones(num_tokens, device=margin_losses.device)
        num_early = min(3, num_tokens)
        weights[:num_early] = early_weight

        loss = (margin_losses * weights).sum() / weights.sum()
        return loss

    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Forward pass through the model.

        This is OPTIONAL - only needed for models that use the default
        compute_margin_loss() implementation.

        Args:
            inputs_embeds: Input embeddings
            attention_mask: Attention mask
            labels: Labels for loss computation

        Returns:
            Model outputs with logits and loss
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _forward(). "
            "This is optional - only needed if using default compute_margin_loss()."
        )
