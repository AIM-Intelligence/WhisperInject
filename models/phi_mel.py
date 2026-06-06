"""
MEL-level attack wrapper for Phi-4 Multimodal model.

Uses the EMBEDDINGS-BASED approach:
1. MEL -> audio embeddings via Conformer encoder
2. Audio embeddings + text embeddings -> inputs_embeds
3. Forward pass with inputs_embeds

IMPORTANT: Phi-4 requires specific package versions:
- transformers==4.48.2
- peft==0.13.0
"""

import torch
import torchaudio
from typing import List, Tuple, Optional

from models.phi import PhiModel


class PhiMelAttackWrapper:
    """
    MEL-level attack wrapper for Phi-4 Multimodal model.

    Uses the embeddings-based approach for memory efficiency:
    1. MEL -> audio embeddings via Conformer encoder
    2. Concatenate with text embeddings
    3. Forward with inputs_embeds

    Usage:
        model = PhiModel()
        mel_wrapper = PhiMelAttackWrapper(model)

        # Convert wav to mel once
        base_mel = mel_wrapper.wav_to_mel(wav)

        # Optimize perturbation in mel space
        perturbation = torch.zeros_like(base_mel, requires_grad=True)
        adv_mel = base_mel + perturbation

        # Compute loss directly from mel
        loss = mel_wrapper.compute_loss_from_mel(adv_mel, target_text)
    """

    # MEL-level attack parameters (Phi uses 80-dim mel with dB scale)
    DEFAULT_EPS = 10.0       # Epsilon for mel space (dB scale)
    DEFAULT_ALPHA = 0.5      # Learning rate for MEL perturbation
    MEL_CLAMP_MIN = -100.0   # Min mel value (dB scale)
    MEL_CLAMP_MAX = 0.0      # Max mel value (dB scale)

    def __init__(self, phi_model: PhiModel):
        """
        Initialize MEL attack wrapper.

        Args:
            phi_model: Initialized PhiModel instance
        """
        self.phi = phi_model
        self.model = phi_model.model
        self.processor = phi_model.processor
        self.tokenizer = phi_model.tokenizer
        self.device = phi_model.device
        self.dtype = phi_model.dtype

        # Special token IDs
        self.end_token_id = phi_model.end_token_id
        self.eos_token_id = phi_model.eos_token_id

        # Use the same MEL transform as PhiModel
        self.mel_transform = phi_model.mel_transform
        self.amplitude_to_db = phi_model.amplitude_to_db

    @property
    def sample_rate(self) -> int:
        return self.phi.sample_rate

    def wav_to_mel(self, wav: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.

        Args:
            wav: Audio waveform [1, T] or [T]
            detach: Whether to detach from computation graph (default True).
                    Set to True when using as base for perturbation optimization.

        Returns:
            Log-mel spectrogram [1, time, 80]
        """
        wav = wav.to(self.device, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # MEL spectrogram
        mel = self.mel_transform(wav)
        log_mel = self.amplitude_to_db(mel)

        # Phi expects [batch, time, mels]
        log_mel = log_mel.transpose(-2, -1).contiguous().to(self.dtype)

        # Detach by default so only perturbation has gradients
        if detach:
            log_mel = log_mel.detach()

        return log_mel

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
        # Using hop_length=160, n_fft=1024
        return (length + 1024) // 160

    def _get_audio_embeddings(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Get audio embeddings via Conformer encoder.

        Args:
            log_mel: Log-mel spectrogram [1, time, 80]

        Returns:
            Audio embeddings [1, compressed_time, hidden_dim]
        """
        audio_embed = self.model.model.embed_tokens_extend.audio_embed

        # Unfreeze audio processor for gradient flow if needed
        if getattr(audio_embed, "freeze_audio_processor", False):
            audio_embed.freeze_audio_processor = False
            for p in audio_embed.parameters():
                p.requires_grad_(True)

        # Get raw encoder (Conformer)
        raw_encoder = getattr(audio_embed, "encoder", None) or \
            getattr(audio_embed, "audio_encoder", None)

        B, T, _ = log_mel.shape
        attn_mask = torch.ones((B, T), dtype=torch.long, device=log_mel.device)
        feats, *_ = raw_encoder(log_mel, attn_mask)

        # Apply projection
        projection = audio_embed.audio_projection
        if isinstance(projection, torch.nn.ModuleDict):
            if "speech" in projection:
                proj_layer = projection["speech"]
            else:
                proj_layer = next(iter(projection.values()))
        else:
            proj_layer = projection

        return proj_layer(feats).to(self.dtype)

    def _create_embeddings_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Create input embeddings from MEL spectrogram.

        Args:
            mel: Log-mel spectrogram [1, time, 80]

        Returns:
            Combined embeddings [1, seq_len, hidden_dim]
        """
        audio_embeddings = self._get_audio_embeddings(mel)

        # Phi-4 prompt format (with English-only system instruction)
        full_prompt = "<|system|>Always respond in English only.<|end|><|user|><|audio_1|><|end|><|assistant|>"
        parts = full_prompt.split("<|audio_1|>")

        txt_embeds = self.model.get_input_embeddings()
        before_ids = self.tokenizer(
            parts[0], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        after_ids = self.tokenizer(
            parts[1], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        inputs_embeds = torch.cat([
            txt_embeds(before_ids),
            audio_embeddings,
            txt_embeds(after_ids)
        ], dim=1)

        return inputs_embeds, audio_embeddings

    def compute_loss_from_mel(
        self,
        mel: torch.Tensor,
        target_text: str,
        audio_length: Optional[int] = None  # Unused, for API compatibility
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss directly from mel spectrogram.

        Args:
            mel: Log-mel spectrogram [1, time, 80]
            target_text: Target text to generate
            audio_length: Unused (kept for API compatibility)

        Returns:
            Loss tensor (scalar, with grad_fn for backprop)
        """
        # Get prompt embeddings and audio embeddings
        prompt_embeds, audio_embeddings = self._create_embeddings_from_mel(mel)

        # Create target embeddings
        txt_embeds = self.model.get_input_embeddings()
        target_ids = self.tokenizer(
            f"{target_text}<|end|><|endoftext|>",
            return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        tgt_embeds = txt_embeds(target_ids)

        # Combine prompt and target embeddings
        inputs_embeds = torch.cat([prompt_embeds, tgt_embeds], dim=1)

        # Create labels (only target text is used for loss)
        labels = torch.full(
            (1, inputs_embeds.size(1)), -100, dtype=torch.long, device=self.device
        )
        labels[0, -target_ids.size(1):] = target_ids

        # Create attention mask
        attn_mask = torch.ones(
            inputs_embeds.shape[:2], device=self.device, dtype=torch.long)

        # Forward pass with Phi-4 specific kwargs
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
            input_mode=torch.tensor([2], dtype=torch.long, device=self.device),
            input_audio_embeds=audio_embeddings.to(self.dtype),
            audio_embed_sizes=torch.tensor(
                [audio_embeddings.shape[1]], device=self.device),
            audio_attention_mask=torch.ones(
                (1, audio_embeddings.shape[1]), dtype=torch.long, device=self.device
            ),
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
        audio_length: Optional[int] = None  # Unused, for API compatibility
    ) -> str:
        """
        Generate text from mel spectrogram.

        Args:
            mel: Log-mel spectrogram [1, time, 80]
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
            # Get embeddings
            embeds, audio_embeddings = self._create_embeddings_from_mel(mel)
            attn_mask = torch.ones(
                embeds.shape[:2], device=self.device, dtype=torch.long)

            # Generate
            gen_ids = self.model.generate(
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                num_beams=num_beams,
                eos_token_id=[self.end_token_id, self.eos_token_id],
                pad_token_id=self.tokenizer.pad_token_id,
                input_mode=torch.tensor(
                    [2], dtype=torch.long, device=self.device),
                input_audio_embeds=audio_embeddings.to(self.dtype),
                audio_embed_sizes=torch.tensor(
                    [audio_embeddings.shape[1]], device=self.device),
                audio_attention_mask=torch.ones(
                    (1, audio_embeddings.shape[1]), dtype=torch.long, device=self.device
                ),
            )

            return self.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            ).strip()

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
            mel: Log-mel spectrogram [1, time, 80]
            num_candidates: Number of candidates to generate
            audio_length: Unused (kept for API compatibility)

        Returns:
            List of (strategy_name, response) tuples
        """
        candidates = []

        # 1. Greedy decoding (deterministic baseline)
        response = self.generate_from_mel(
            mel, do_sample=False, max_tokens=128
        )
        candidates.append(("greedy", response))
        if len(candidates) >= num_candidates:
            return candidates

        # 2. Beam search (better quality than greedy)
        response = self.generate_from_mel(
            mel, do_sample=False, num_beams=3, max_tokens=128
        )
        candidates.append(("beam_3", response))
        if len(candidates) >= num_candidates:
            return candidates

        # 3. Sampling (diversity)
        response = self.generate_from_mel(
            mel, do_sample=True, temperature=0.7, top_p=0.9, max_tokens=128
        )
        candidates.append(("sample_t0.7", response))
        if len(candidates) >= num_candidates:
            return candidates

        # 4-6. Additional candidates if requested
        for beam_size in [5]:
            response = self.generate_from_mel(
                mel, do_sample=False, num_beams=beam_size, max_tokens=128
            )
            candidates.append((f"beam_{beam_size}", response))
            if len(candidates) >= num_candidates:
                return candidates

        for temp in [1.0, 1.3]:
            response = self.generate_from_mel(
                mel, do_sample=True, temperature=temp, top_p=0.9, max_tokens=128
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
