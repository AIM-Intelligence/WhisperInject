"""
MEL-level attack wrapper for Voxtral Mini 3B model.

Uses the EMBEDDINGS-BASED approach:
1. MEL -> audio embeddings via get_audio_features() (VoxtralEncoder + projector)
2. Audio embeddings + text embeddings -> inputs_embeds
3. Forward pass through language_model with inputs_embeds

This enables gradient-based optimization in MEL space.
"""

import torch
import torchaudio
from typing import List, Tuple, Optional

from models.voxtral import VoxtralModel


class VoxtralMelAttackWrapper:
    """
    MEL-level attack wrapper for Voxtral Mini 3B model.

    Uses get_audio_features() for differentiable MEL -> embeddings conversion.

    Usage:
        model = VoxtralModel()
        mel_wrapper = VoxtralMelAttackWrapper(model)

        # Convert wav to mel once
        base_mel = mel_wrapper.wav_to_mel(wav)

        # Optimize perturbation in mel space
        perturbation = torch.zeros_like(base_mel, requires_grad=True)
        adv_mel = base_mel + perturbation

        # Compute loss directly from mel
        loss = mel_wrapper.compute_loss_from_mel(adv_mel, target_text)
    """

    # MEL-level attack parameters
    DEFAULT_EPS = 1.0        # Epsilon for mel space
    DEFAULT_ALPHA = 0.05     # Learning rate for MEL perturbation
    MEL_CLAMP_MIN = -2.0     # Min mel value
    MEL_CLAMP_MAX = 8.0      # Max mel value
    MAX_MEL_LENGTH = 3000    # Voxtral's expected MEL length

    def __init__(self, voxtral_model: VoxtralModel):
        """
        Initialize MEL attack wrapper.

        Args:
            voxtral_model: Initialized VoxtralModel instance
        """
        self.voxtral = voxtral_model
        self.model = voxtral_model.model
        self.processor = voxtral_model.processor
        self.tokenizer = voxtral_model.tokenizer
        self.feature_extractor = voxtral_model.feature_extractor
        self.device = voxtral_model.device
        self.dtype = voxtral_model.dtype

        # Create torchaudio MEL transform for differentiable conversion
        # Voxtral uses WhisperFeatureExtractor: 128 mels, 16kHz
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=voxtral_model.sample_rate,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=128,
            f_min=0,
            f_max=8000,
            power=1.0,
        ).to(self.device)

    @property
    def sample_rate(self) -> int:
        return self.voxtral.sample_rate

    def wav_to_mel(self, wav: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram (differentiable).

        Uses torchaudio MelSpectrogram with WHISPER-STYLE normalization.
        This matches what the WhisperFeatureExtractor produces.

        Args:
            wav: Audio waveform [1, T] or [T]
            detach: Whether to detach from computation graph (default True).
                    Set to True when using as base for perturbation optimization.

        Returns:
            Log-mel spectrogram [1, n_mels, T']
        """
        wav = wav.to(self.device, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Get raw MEL spectrogram
        mel_raw = self.mel_transform(wav)

        # WHISPER-STYLE log-mel normalization (NOT log1p!)
        # This matches the WhisperFeatureExtractor's output range [-0.65, 1.35]
        mel = torch.log10(torch.clamp(mel_raw, min=1e-10))
        mel = torch.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0

        # Detach by default so only perturbation has gradients
        if detach:
            mel = mel.detach()

        return mel

    def get_audio_length(self, wav: torch.Tensor) -> int:
        """
        Get the number of MEL frames for a waveform.

        Args:
            wav: Audio waveform [1, T] or [T]

        Returns:
            Number of MEL frames
        """
        if wav.dim() == 1:
            length = wav.shape[0]
        else:
            length = wav.shape[1]
        # Using hop_length=160, n_fft=400
        return (length + 400) // 160

    def _pad_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Pad MEL spectrogram to expected length (3000 frames).

        Voxtral's audio tower expects fixed-length MEL input.

        Args:
            mel: Log-mel spectrogram [1, n_mels, time_frames]

        Returns:
            Padded mel [1, n_mels, MAX_MEL_LENGTH]
        """
        current_length = mel.shape[2]
        if current_length < self.MAX_MEL_LENGTH:
            # Pad with minimum value (silence)
            pad_value = mel.min().item()
            padding = torch.full(
                (mel.shape[0], mel.shape[1],
                 self.MAX_MEL_LENGTH - current_length),
                pad_value,
                dtype=mel.dtype,
                device=mel.device
            )
            mel = torch.cat([mel, padding], dim=2)
        elif current_length > self.MAX_MEL_LENGTH:
            mel = mel[:, :, :self.MAX_MEL_LENGTH]
        return mel

    def _create_embeddings_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Create input embeddings from MEL spectrogram using get_audio_features().

        Args:
            mel: Log-mel spectrogram [1, n_mels, time_frames]

        Returns:
            Combined embeddings [1, seq_len, hidden_dim]
        """
        # Pad MEL to expected length
        mel_padded = self._pad_mel(mel)

        # Get audio embeddings via VoxtralEncoder + projector
        # get_audio_features expects [batch, mels, time] and returns [batch, seq, hidden]
        audio_embeds = self.model.get_audio_features(mel_padded.to(self.dtype))
        audio_embeds = audio_embeds.unsqueeze(0)  # [1, seq, hidden]

        # Create prompt embeddings
        prompt_text = "[INST]"
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt"
        ).input_ids.to(self.device)
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)

        # Concatenate: [prompt] [audio]
        inputs_embeds = torch.cat([prompt_embeds, audio_embeds], dim=1)

        return inputs_embeds

    def compute_loss_from_mel(
        self,
        mel: torch.Tensor,
        target_text: str,
        # Unused, kept for API compatibility
        audio_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss directly from mel spectrogram.

        Uses the EMBEDDINGS approach:
        1. MEL -> audio embeddings via get_audio_features()
        2. Concatenate with text embeddings
        3. Forward with inputs_embeds

        Args:
            mel: Log-mel spectrogram [1, n_mels, T']
            target_text: Target text to generate
            audio_length: Unused (kept for API compatibility)

        Returns:
            Loss tensor (scalar, with grad_fn for backprop)
        """
        # Get prompt embeddings via audio features
        prompt_embeds = self._create_embeddings_from_mel(mel)

        # Create target embeddings
        target_ids = self.tokenizer(
            target_text, return_tensors="pt"
        ).input_ids.to(self.device)
        target_embeds = self.model.get_input_embeddings()(target_ids)

        # Combine prompt and target embeddings
        inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

        # Create labels (only target text is used for loss)
        labels = torch.full(
            (1, inputs_embeds.size(1)), -100, dtype=torch.long, device=self.device
        )
        labels[0, -target_ids.size(1):] = target_ids

        # Forward pass through language model
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )

        return outputs.loss

    def generate_from_mel(
        self,
        mel: torch.Tensor,
        max_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        top_p: float = 0.9,
        # Unused, kept for API compatibility
        audio_length: Optional[int] = None
    ) -> str:
        """
        Generate text from mel spectrogram using EMBEDDINGS approach.

        Args:
            mel: Log-mel spectrogram [1, n_mels, T']
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            top_p: Top-p for nucleus sampling
            audio_length: Unused (kept for API compatibility)

        Returns:
            Generated text string
        """
        with torch.no_grad():
            # Get embeddings via audio features
            embeds = self._create_embeddings_from_mel(mel)
            attn_mask = torch.ones(
                embeds.shape[:2], device=self.device, dtype=torch.long)

            # Generate using language_model
            gen_ids = self.model.language_model.generate(
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Decode generated tokens
            response = self.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            ).strip()

        return response

    def generate_candidates_from_mel(
        self,
        mel: torch.Tensor,
        num_candidates: int = 3,
        audio_length: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        Generate multiple candidate responses from mel spectrogram.

        Uses various decoding strategies to produce diverse candidates.
        Default is 3 candidates to reduce memory usage.

        Args:
            mel: Log-mel spectrogram [1, n_mels, T']
            num_candidates: Number of candidates to generate (default 3)
            audio_length: Actual audio length in frames (optional)

        Returns:
            List of (strategy_name, response) tuples
        """
        candidates = []

        # 1. Greedy decoding (deterministic baseline)
        response = self.generate_from_mel(
            mel, do_sample=False, max_tokens=128, audio_length=audio_length
        )
        candidates.append(("greedy", response))
        if len(candidates) >= num_candidates:
            return candidates

        # 2. Beam search (better quality than greedy)
        response = self.generate_from_mel(
            mel, do_sample=False, num_beams=3, max_tokens=128,
            audio_length=audio_length
        )
        candidates.append(("beam_3", response))
        if len(candidates) >= num_candidates:
            return candidates

        # 3. Sampling (diversity)
        response = self.generate_from_mel(
            mel, do_sample=True, temperature=0.7, top_p=0.9, max_tokens=128,
            audio_length=audio_length
        )
        candidates.append(("sample_t0.7", response))
        if len(candidates) >= num_candidates:
            return candidates

        # 4-6. Additional candidates if requested
        for beam_size in [5]:
            response = self.generate_from_mel(
                mel, do_sample=False, num_beams=beam_size, max_tokens=128,
                audio_length=audio_length
            )
            candidates.append((f"beam_{beam_size}", response))
            if len(candidates) >= num_candidates:
                return candidates

        for temp in [1.0, 1.3]:
            response = self.generate_from_mel(
                mel, do_sample=True, temperature=temp, top_p=0.9, max_tokens=128,
                audio_length=audio_length
            )
            candidates.append((f"sample_t{temp}", response))
            if len(candidates) >= num_candidates:
                return candidates

        return candidates

    def clamp_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Clamp mel spectrogram to valid range.

        Args:
            mel: Log-mel spectrogram

        Returns:
            Clamped mel spectrogram
        """
        return torch.clamp(mel, self.MEL_CLAMP_MIN, self.MEL_CLAMP_MAX)
