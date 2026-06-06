"""
Qwen 2.5 Omni audio model wrapper for adversarial attacks.

Uses the EMBEDDINGS-BASED approach (same as safety-bypass/utils/mel_attacks.py):
1. WAV -> MEL spectrogram (differentiable)
2. MEL -> compressed audio features via get_audio_features() (~4x smaller)
3. Audio features + text embeddings -> inputs_embeds
4. Forward pass with inputs_embeds (NOT input_features)

This approach is much more memory-efficient than passing raw MEL directly:
- No fixed 30000 frame padding needed
- Smaller computation graph
- Works with gradient checkpointing for ~50% memory reduction
"""

import os
import torch
import torchaudio
from typing import Optional

from models.base import BaseAudioModel


class QwenMelTransform:
    """
    Differentiable MEL spectrogram for Qwen 2.5 Omni.

    Uses the EXACT mel filterbank from the processor's WhisperFeatureExtractor
    to ensure consistency between attack and inference.

    Parameters (from WhisperFeatureExtractor):
    - n_fft: 400
    - hop_length: 160
    - n_mels: 128
    - mel_filters: from librosa (201, 128)

    Qwen expects: [batch, 128 mels, time frames]
    """

    def __init__(
        self,
        processor,  # Qwen processor to extract mel_filters
        n_fft: int = 400,
        hop_length: int = 160,
        device: str = "cuda"
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device

        # Extract mel filterbank from processor (EXACT match)
        mel_filters_np = processor.feature_extractor.mel_filters  # (201, 128)
        self.mel_filters = torch.from_numpy(
            mel_filters_np).float().to(device)  # (201, 128)
        self.n_mels = self.mel_filters.shape[1]

        # Hann window for STFT
        self.window = torch.hann_window(n_fft).to(device)

    def __call__(self, waveform: torch.Tensor, max_length: int = 30000, pad: bool = True) -> torch.Tensor:
        """
        Convert waveform to MEL spectrogram using processor's exact mel filterbank.

        Args:
            waveform: [1, T] audio tensor
            max_length: Pad to this many frames (default: 30000 for Qwen)
            pad: Whether to pad to max_length (default: True). Set to False for 
                 memory-efficient attack optimization.

        Returns:
            mel: [1, n_mels, T'] normalized log-mel spectrogram
                 If pad=True, T' = max_length. If pad=False, T' = actual frames.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Ensure waveform is on correct device
        waveform = waveform.to(self.device)

        # Compute STFT
        stft = torch.stft(
            waveform.squeeze(0),  # (T,)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode='reflect',
            return_complex=True
        )  # (n_fft//2+1, time)

        # Compute magnitude squared (power spectrogram)
        magnitudes = stft.abs() ** 2  # (201, time)

        # Apply mel filterbank: (201, 128).T @ (201, time) = (128, time)
        mel = torch.matmul(self.mel_filters.T, magnitudes)  # (128, time)

        # Add batch dimension
        mel = mel.unsqueeze(0)  # (1, 128, time)

        # Log-mel spectrogram (Whisper's exact formula)
        mel = torch.log10(torch.clamp(mel, min=1e-10))
        mel = torch.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0  # Normalize to roughly [-1, 1]

        # Optionally pad to max_length (Qwen expects fixed-length for generation)
        if pad:
            current_length = mel.shape[2]
            if current_length < max_length:
                pad_value = mel.min().item()
                padding = torch.full(
                    (mel.shape[0], mel.shape[1], max_length - current_length),
                    pad_value,
                    dtype=mel.dtype,
                    device=mel.device
                )
                mel = torch.cat([mel, padding], dim=2)
            elif current_length > max_length:
                mel = mel[:, :, :max_length]

        return mel

    def get_num_frames(self, waveform: torch.Tensor) -> int:
        """Get the number of MEL frames for a waveform (before padding)."""
        if waveform.dim() == 1:
            length = waveform.shape[0]
        else:
            length = waveform.shape[1]
        return (length + self.n_fft) // self.hop_length


class QwenModel(BaseAudioModel):
    """
    Wrapper for Qwen 2.5 Omni model with audio capabilities.

    Unlike Gemma, Qwen takes MEL spectrograms directly as input_features
    rather than splicing audio embeddings into text embeddings.

    Implements the BaseAudioModel interface for use with attack algorithms.
    """

    MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
    SAMPLE_RATE = 16000

    # Default system prompt for Qwen
    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual inputs, "
        "as well as generating text and speech."
    )

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None
    ):
        """
        Initialize the Qwen model.

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

        print(f"Loading Qwen model: {model_id}")
        print(f"Device: {device}, Dtype: {dtype}")

        # Load model and processor
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_id, token=token)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            token=token
        ).eval()

        # Disable talker to save memory (we only need text output)
        if hasattr(self.model, 'disable_talker'):
            self.model.disable_talker()

        # Enable gradient checkpointing to reduce memory usage
        # Trades compute for memory by recomputing activations during backward
        self.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (memory optimization)")

        # Get the thinker for inference (text generation)
        self.thinker = self.model.thinker
        self.tokenizer = self.processor.tokenizer

        # Initialize differentiable MEL transform using processor's exact filterbank
        self.mel_transform = QwenMelTransform(
            processor=self.processor,
            device=device
        )

        # Also create a torchaudio-based MEL transform for embeddings approach
        # (slightly different but simpler and also works well)
        self.mel_transform_torchaudio = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=128,
            f_min=0,
            f_max=8000,
            power=1.0,
        ).to(device)

        # Special token IDs for embeddings approach
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids(
            "<|audio_bos|>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids(
            "<|audio_eos|>")

        print("Qwen model loaded successfully!")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def wav_to_mel_simple(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram using torchaudio.

        This is a simpler MEL transform used for the embeddings approach.

        Args:
            wav: Audio waveform [T] or [1, T]

        Returns:
            Log-mel spectrogram [1, n_mels, time_frames]
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self._device)
        mel = torch.log1p(self.mel_transform_torchaudio(wav))
        return mel

    def _create_embeddings_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Create input embeddings from MEL spectrogram using get_audio_features().

        This is the KEY memory optimization:
        - Instead of passing raw MEL (30000 frames) directly as input_features
        - We compress to audio features (~4x smaller) using get_audio_features()
        - Then concatenate with text embeddings

        Args:
            mel: Log-mel spectrogram [1, n_mels, time_frames]

        Returns:
            Combined embeddings [1, seq_len, hidden_dim]
        """
        # Create feature attention mask for the actual mel length
        feature_attention_mask = torch.ones(
            mel.shape[0], mel.shape[2], dtype=torch.long, device=self._device
        )

        # Get compressed audio features (KEY STEP!)
        # This compresses the MEL to ~4x smaller representation
        audio_features = self.thinker.get_audio_features(
            input_features=mel.to(self._dtype),
            feature_attention_mask=feature_attention_mask
        ).unsqueeze(0)  # Shape: (1, 1, seq_len, hidden_dim)

        # Squeeze the extra dimension if needed
        if audio_features.dim() == 4:
            audio_features = audio_features.squeeze(
                1)  # (1, seq_len, hidden_dim)

        # Create text parts using Qwen chat format
        system_part = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
        user_part_prefix = f"<|im_start|>user\n"
        user_part_suffix = f"<|im_end|>\n"
        assistant_cue = "<|im_start|>assistant\n"

        txt_embeds = self.thinker.get_input_embeddings()

        # Tokenize text parts
        sys_ids = self.tokenizer(
            [system_part], return_tensors="pt").input_ids.to(self._device)
        user_prefix_ids = self.tokenizer(
            [user_part_prefix], return_tensors="pt").input_ids.to(self._device)
        user_suffix_ids = self.tokenizer(
            [user_part_suffix], return_tensors="pt").input_ids.to(self._device)
        assist_cue_ids = self.tokenizer(
            [assistant_cue], return_tensors="pt").input_ids.to(self._device)

        # Create audio token embeddings
        audio_bos_embed = txt_embeds(torch.tensor(
            [[self.audio_bos_id]], device=self._device))
        audio_eos_embed = txt_embeds(torch.tensor(
            [[self.audio_eos_id]], device=self._device))

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

    def _create_inputs(
        self,
        mel: torch.Tensor,
        target_text: str = "",
        audio_length: Optional[int] = None
    ) -> dict:
        """
        Create model inputs from MEL spectrogram.

        NOTE: This method is DEPRECATED for loss computation.
        Use _create_embeddings_from_mel() + forward with inputs_embeds instead.
        Kept for backward compatibility with generation code.

        Args:
            mel: [1, n_mels, time] MEL spectrogram (padded)
            target_text: Target text to append for loss computation
            audio_length: Actual audio length in frames (for attention mask)

        Returns:
            dict with input_ids, attention_mask, input_features, etc.
        """
        mel_length = mel.shape[2]
        if audio_length is None:
            audio_length = mel_length

        # Calculate number of audio tokens (4 frames per token)
        num_audio_tokens = audio_length // 4

        # Create conversation with system prompt
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio_url": "placeholder"}]
            },
        ]

        # Apply chat template
        chat_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # Expand <|AUDIO|> token based on audio length
        audio_token = "<|AUDIO|>"
        expanded_audio = audio_token * num_audio_tokens
        chat_text = chat_text.replace(audio_token, expanded_audio)

        # Add target text if provided (for loss computation)
        full_text = chat_text + target_text

        # Tokenize
        tokens = self.processor.tokenizer(full_text, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)

        # Create feature attention mask (1 for real audio, 0 for padding)
        feature_attention_mask = torch.zeros(
            1, mel_length, dtype=torch.int32, device=self.model.device
        )
        feature_attention_mask[:, :audio_length] = 1

        # Create labels if target text provided
        labels = None
        if target_text:
            labels = input_ids.clone()
            target_tokens = self.processor.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            target_len = target_tokens.shape[1]
            # Mask everything except target tokens
            labels[:, :-target_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": mel.to(self._dtype),
            "feature_attention_mask": feature_attention_mask,
            "labels": labels
        }

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for target text using EMBEDDINGS approach.

        Uses get_audio_features() to compress MEL, then forwards with inputs_embeds.
        This is much more memory-efficient than passing raw MEL directly.

        Differentiable with respect to wav for gradient-based attacks.

        Args:
            wav: Audio waveform tensor [T] or [1, T]
            target_text: Target text

        Returns:
            Loss tensor (scalar, with grad_fn)
        """
        wav = wav.to(self._device, dtype=torch.float32)

        # WAV → MEL using simple torchaudio transform (differentiable)
        mel = self.wav_to_mel_simple(wav)

        # Get prompt embeddings via compressed audio features
        prompt_embeds = self._create_embeddings_from_mel(mel)

        # Create target embeddings
        txt_embeds = self.thinker.get_input_embeddings()
        tgt_ids = self.tokenizer(
            target_text, return_tensors="pt").input_ids.to(self._device)
        tgt_embeds = txt_embeds(tgt_ids)

        # Combine prompt and target embeddings
        inputs_embeds = torch.cat([prompt_embeds, tgt_embeds], dim=1)

        # Create labels (only target text is used for loss)
        labels = torch.full(
            (1, inputs_embeds.size(1)), -100, dtype=torch.long, device=self._device
        )
        labels[0, -tgt_ids.size(1):] = tgt_ids

        # Create attention mask
        attn_mask = torch.ones(
            inputs_embeds.shape[:2], device=self._device, dtype=torch.long)

        # Forward pass with inputs_embeds (NOT input_features)
        outputs = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels
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
        Compute margin loss (Carlini-Wagner style) for Qwen using EMBEDDINGS approach.

        Args:
            wav: Audio waveform tensor [T] or [1, T]
            target_text: Target text
            kappa: Margin (higher = more confident target)
            early_weight: Extra weight for first few tokens

        Returns:
            Loss tensor (scalar)
        """
        wav = wav.to(self._device, dtype=torch.float32)

        # WAV → MEL using simple torchaudio transform (differentiable)
        mel = self.wav_to_mel_simple(wav)

        # Get prompt embeddings via compressed audio features
        prompt_embeds = self._create_embeddings_from_mel(mel)

        # Create target embeddings
        txt_embeds = self.thinker.get_input_embeddings()
        tgt_ids = self.tokenizer(
            target_text, return_tensors="pt").input_ids.to(self._device)
        tgt_embeds = txt_embeds(tgt_ids)

        # Combine prompt and target embeddings
        inputs_embeds = torch.cat([prompt_embeds, tgt_embeds], dim=1)

        # Create labels (only target text is used for loss)
        labels = torch.full(
            (1, inputs_embeds.size(1)), -100, dtype=torch.long, device=self._device
        )
        labels[0, -tgt_ids.size(1):] = tgt_ids

        # Create attention mask
        attn_mask = torch.ones(
            inputs_embeds.shape[:2], device=self._device, dtype=torch.long)

        # Forward pass with inputs_embeds
        outputs = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels
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

    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate text from audio using Qwen with EMBEDDINGS approach.

        Args:
            wav: Audio waveform tensor [T] or [1, T]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy)

        Returns:
            Generated text string
        """
        with torch.no_grad():
            wav = wav.to(self._device, dtype=torch.float32)

            # WAV → MEL using simple torchaudio transform
            mel = self.wav_to_mel_simple(wav)

            # Get embeddings via compressed audio features
            embeds = self._create_embeddings_from_mel(mel)
            attn_mask = torch.ones(
                embeds.shape[:2], device=self._device, dtype=torch.long)

            # Generate using inputs_embeds
            gen_ids = self.thinker.generate(
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode all generated tokens
            response = self.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            ).strip()

        return response
