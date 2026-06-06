"""
Voxtral Mini 3B audio model wrapper for adversarial attacks.

Uses the EMBEDDINGS-BASED approach:
1. WAV -> MEL spectrogram via WhisperFeatureExtractor
2. MEL -> audio embeddings via get_audio_features() (VoxtralEncoder + projector)
3. Audio embeddings + text embeddings -> inputs_embeds
4. Forward pass through language_model with inputs_embeds

Key differences from other models:
- Uses `model.language_model` for LM operations (not model.thinker like Qwen)
- Uses `model.get_audio_features()` for audio embedding extraction
- Simple prompt format with [INST] prefix
- Compatible with transformers 4.57+ (same as Gemma/Qwen)
"""

import os
import torch
import numpy as np
from typing import Optional

from models.base import BaseAudioModel


class VoxtralModel(BaseAudioModel):
    """
    Wrapper for Voxtral Mini 3B model with audio capabilities.

    Voxtral uses a VoxtralEncoder (based on Whisper) to process audio,
    followed by a multi-modal projector to align with the LM embedding space.

    Implements the BaseAudioModel interface for use with attack algorithms.
    """

    MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None
    ):
        """
        Initialize the Voxtral model.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on
            dtype: Model dtype
            token: HuggingFace token (or uses env var)
        """
        self._device = device
        self._dtype = dtype

        # Get HF token
        if token is None:
            token = os.getenv(
                "HUGGINGFACE_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

        print(f"Loading Voxtral model: {model_id}")
        print(f"Device: {device}, Dtype: {dtype}")

        # Load model and processor
        from transformers import VoxtralForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            token=token
        ).eval()

        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        print("Voxtral model loaded successfully!")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to MEL spectrogram via feature extractor.

        Args:
            wav: Audio waveform [T] or [1, T]

        Returns:
            MEL spectrogram [1, 128, time]
        """
        if isinstance(wav, torch.Tensor):
            if wav.dim() > 1:
                wav = wav.squeeze(0)
            wav_np = wav.detach().cpu().numpy()
        else:
            wav_np = wav

        mel = self.feature_extractor(
            wav_np,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(self._device)

        return mel

    def _get_audio_embeddings(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Get audio embeddings via VoxtralEncoder and projector.

        Args:
            mel: MEL spectrogram [1, 128, time]

        Returns:
            Audio embeddings [1, compressed_time, hidden_dim]
        """
        audio_embeds = self.model.get_audio_features(mel.to(self._dtype))
        return audio_embeds.unsqueeze(0)

    def _create_embeddings_from_mel(
        self,
        mel: torch.Tensor,
        target_text: str = ""
    ) -> torch.Tensor:
        """
        Create input embeddings from MEL spectrogram.

        Args:
            mel: MEL spectrogram [1, 128, time]
            target_text: Optional target text to append

        Returns:
            Combined embeddings [1, seq_len, hidden_dim]
        """
        audio_embeddings = self._get_audio_embeddings(mel)

        # Voxtral prompt format
        prompt_text = "[INST]"
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt"
        ).input_ids.to(self._device)
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)

        inputs_embeds = torch.cat([prompt_embeds, audio_embeddings], dim=1)

        return inputs_embeds

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for target text.

        Differentiable with respect to wav for gradient-based attacks.

        Args:
            wav: Audio waveform tensor [T] or [1, T]
            target_text: Target text

        Returns:
            Loss tensor (scalar, with grad_fn)
        """
        # WAV → MEL (via feature extractor - non-differentiable)
        # For differentiable MEL, use VoxtralMelAttackWrapper
        mel = self.wav_to_mel(wav)
        audio_embeds = self._get_audio_embeddings(mel)

        # Create prompt embeddings
        prompt_text = "[INST]"
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt"
        ).input_ids.to(self._device)
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)

        # Target embeddings
        target_ids = self.tokenizer(
            target_text, return_tensors="pt"
        ).input_ids.to(self._device)
        target_embeds = self.model.get_input_embeddings()(target_ids)

        # Combine: [prompt] [audio] [target]
        inputs_embeds = torch.cat(
            [prompt_embeds, audio_embeds, target_embeds], dim=1
        )

        # Labels: -100 for prompt+audio, actual ids for target
        labels = torch.full(
            (1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=self._device
        )
        labels[:, -target_ids.shape[1]:] = target_ids

        # Forward pass through language model
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )

        return outputs.loss

    def compute_margin_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        kappa: float = 5.0,
        early_weight: float = 5.0
    ) -> torch.Tensor:
        """
        Compute margin loss (Carlini-Wagner style) for Voxtral.

        Args:
            wav: Audio waveform tensor [T] or [1, T]
            target_text: Target text
            kappa: Margin (higher = more confident target)
            early_weight: Extra weight for first few tokens

        Returns:
            Loss tensor (scalar)
        """
        mel = self.wav_to_mel(wav)
        audio_embeds = self._get_audio_embeddings(mel)

        # Create prompt embeddings
        prompt_text = "[INST]"
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt"
        ).input_ids.to(self._device)
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)

        # Target embeddings
        target_ids = self.tokenizer(
            target_text, return_tensors="pt"
        ).input_ids.to(self._device)
        target_embeds = self.model.get_input_embeddings()(target_ids)

        # Combine
        inputs_embeds = torch.cat(
            [prompt_embeds, audio_embeds, target_embeds], dim=1
        )

        # Labels
        labels = torch.full(
            (1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=self._device
        )
        labels[:, -target_ids.shape[1]:] = target_ids

        # Forward pass
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )

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
        valid_labels = shift_labels[0, valid_positions].to(valid_logits.device)

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
            wav: Audio waveform tensor [T] or [1, T]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy)

        Returns:
            Generated text string
        """
        with torch.no_grad():
            mel = self.wav_to_mel(wav)
            audio_embeds = self._get_audio_embeddings(mel)

            # Create prompt embeddings
            prompt_text = "[INST]"
            prompt_ids = self.tokenizer(
                prompt_text, return_tensors="pt"
            ).input_ids.to(self._device)
            prompt_embeds = self.model.get_input_embeddings()(prompt_ids)

            inputs_embeds = torch.cat([prompt_embeds, audio_embeds], dim=1)
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], device=self._device)

            gen_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
