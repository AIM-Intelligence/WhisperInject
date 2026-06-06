"""
MEL-level attack wrapper for Qwen 2.5 Omni model.

Uses the EMBEDDINGS-BASED approach (same as safety-bypass/utils/mel_attacks.py):
1. MEL -> compressed audio features via get_audio_features() (~4x smaller)
2. Audio features + text embeddings -> inputs_embeds
3. Forward pass with inputs_embeds (NOT input_features)

This approach is much more memory-efficient:
- No fixed 30000 frame padding needed during optimization
- Smaller computation graph
- Works with gradient checkpointing for ~50% memory reduction
"""

import torch
import torchaudio
from typing import List, Tuple, Optional

from models.qwen import QwenModel


class QwenMelAttackWrapper:
    """
    MEL-level attack wrapper for Qwen 2.5 Omni model.

    Uses the embeddings-based approach for memory efficiency:
    1. MEL -> audio features via get_audio_features()
    2. Concatenate with text embeddings
    3. Forward with inputs_embeds (not input_features)

    Usage:
        model = QwenModel()
        mel_wrapper = QwenMelAttackWrapper(model)

        # Convert wav to mel once
        base_mel = mel_wrapper.wav_to_mel(wav)

        # Optimize perturbation in mel space
        perturbation = torch.zeros_like(base_mel, requires_grad=True)
        adv_mel = base_mel + perturbation

        # Compute loss directly from mel (memory efficient!)
        loss = mel_wrapper.compute_loss_from_mel(adv_mel, target_text)
    """

    # MEL-level attack parameters
    # Using torchaudio log1p normalization scale
    DEFAULT_EPS = 1.0        # Epsilon for mel space
    DEFAULT_ALPHA = 0.05     # Learning rate for MEL perturbation
    MEL_CLAMP_MIN = -2.0     # Min mel value
    MEL_CLAMP_MAX = 8.0      # Max mel value (log1p scale can go higher)
    MAX_MEL_LENGTH = 30000   # Qwen's expected MEL length (for generation only)

    def __init__(self, qwen_model: QwenModel):
        """
        Initialize MEL attack wrapper.

        Args:
            qwen_model: Initialized QwenModel instance
        """
        self.qwen = qwen_model
        self.model = qwen_model.model
        self.thinker = qwen_model.thinker
        self.processor = qwen_model.processor
        self.tokenizer = qwen_model.tokenizer
        self.device = qwen_model.device
        self.dtype = qwen_model.dtype
        self.eos_token_id = qwen_model.eos_token_id

        # Special token IDs for embeddings approach
        self.audio_bos_id = qwen_model.audio_bos_id
        self.audio_eos_id = qwen_model.audio_eos_id

        # Create torchaudio MEL transform for embeddings approach
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=qwen_model.sample_rate,
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
        return self.qwen.sample_rate

    def wav_to_mel(self, wav: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.

        Uses torchaudio MelSpectrogram with log1p normalization.
        No padding is applied - embeddings approach doesn't need fixed-length MEL.

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

        # Convert to MEL with log1p normalization
        mel = torch.log1p(self.mel_transform(wav))

        # Detach by default so only perturbation has gradients during optimization
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

    def _create_embeddings_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Create input embeddings from MEL spectrogram using get_audio_features().

        This is the KEY memory optimization:
        - Compresses MEL to audio features (~4x smaller)
        - Concatenates with text embeddings

        Args:
            mel: Log-mel spectrogram [1, n_mels, time_frames]

        Returns:
            Combined embeddings [1, seq_len, hidden_dim]
        """
        # Create feature attention mask for the actual mel length
        feature_attention_mask = torch.ones(
            mel.shape[0], mel.shape[2], dtype=torch.long, device=self.device
        )

        # Get compressed audio features (KEY STEP!)
        audio_features = self.thinker.get_audio_features(
            input_features=mel.to(self.dtype),
            feature_attention_mask=feature_attention_mask
        ).unsqueeze(0)  # Shape: (1, 1, seq_len, hidden_dim)

        # Squeeze the extra dimension if needed
        if audio_features.dim() == 4:
            audio_features = audio_features.squeeze(
                1)  # (1, seq_len, hidden_dim)

        # Create text parts using Qwen chat format
        system_part = f"<|im_start|>system\n{self.qwen.SYSTEM_PROMPT}<|im_end|>\n"
        user_part_prefix = f"<|im_start|>user\n"
        user_part_suffix = f"<|im_end|>\n"
        assistant_cue = "<|im_start|>assistant\n"

        txt_embeds = self.thinker.get_input_embeddings()

        # Tokenize text parts
        sys_ids = self.tokenizer(
            [system_part], return_tensors="pt").input_ids.to(self.device)
        user_prefix_ids = self.tokenizer(
            [user_part_prefix], return_tensors="pt").input_ids.to(self.device)
        user_suffix_ids = self.tokenizer(
            [user_part_suffix], return_tensors="pt").input_ids.to(self.device)
        assist_cue_ids = self.tokenizer(
            [assistant_cue], return_tensors="pt").input_ids.to(self.device)

        # Create audio token embeddings
        audio_bos_embed = txt_embeds(torch.tensor(
            [[self.audio_bos_id]], device=self.device))
        audio_eos_embed = txt_embeds(torch.tensor(
            [[self.audio_eos_id]], device=self.device))

        # Concatenate all embeddings
        inputs_embeds = torch.cat([
            txt_embeds(sys_ids),
            txt_embeds(user_prefix_ids),
            audio_bos_embed,
            audio_features,
            audio_eos_embed,
            txt_embeds(user_suffix_ids),
            txt_embeds(assist_cue_ids),
        ], dim=1)

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

        Uses the EMBEDDINGS approach for memory efficiency:
        1. MEL -> audio features via get_audio_features()
        2. Concatenate with text embeddings
        3. Forward with inputs_embeds

        Args:
            mel: Log-mel spectrogram [1, n_mels, T']
            target_text: Target text to generate
            audio_length: Unused (kept for API compatibility)

        Returns:
            Loss tensor (scalar, with grad_fn for backprop)
        """
        # Get prompt embeddings via compressed audio features
        prompt_embeds = self._create_embeddings_from_mel(mel)

        # Create target embeddings
        txt_embeds = self.thinker.get_input_embeddings()
        tgt_ids = self.tokenizer(
            target_text, return_tensors="pt").input_ids.to(self.device)
        tgt_embeds = txt_embeds(tgt_ids)

        # Combine prompt and target embeddings
        inputs_embeds = torch.cat([prompt_embeds, tgt_embeds], dim=1)

        # Create labels (only target text is used for loss)
        labels = torch.full(
            (1, inputs_embeds.size(1)), -100, dtype=torch.long, device=self.device
        )
        labels[0, -tgt_ids.size(1):] = tgt_ids

        # Create attention mask
        attn_mask = torch.ones(
            inputs_embeds.shape[:2], device=self.device, dtype=torch.long)

        # Forward pass with inputs_embeds (NOT input_features)
        outputs = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels
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
            # Get embeddings via compressed audio features
            embeds = self._create_embeddings_from_mel(mel)
            attn_mask = torch.ones(
                embeds.shape[:2], device=self.device, dtype=torch.long)

            # Generate using inputs_embeds
            gen_ids = self.thinker.generate(
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                num_beams=num_beams,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode all generated tokens
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
        Default is 3 candidates to reduce memory usage (vs 6 for Gemma).

        Args:
            mel: Log-mel spectrogram [1, n_mels, T']
            num_candidates: Number of candidates to generate (default 3 for memory)
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

        # 4-6. Additional candidates if requested (more memory)
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
