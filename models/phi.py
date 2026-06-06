"""
Phi-4 Multimodal audio model wrapper for adversarial attacks.

Uses the EMBEDDINGS-BASED approach:
1. WAV -> MEL spectrogram (differentiable, 80-dim)
2. MEL -> audio embeddings via Conformer encoder
3. Audio embeddings + text embeddings -> inputs_embeds
4. Forward pass with inputs_embeds

IMPORTANT: Phi-4 requires specific package versions:
- transformers==4.48.2
- peft==0.13.0

See whisper-inject-v2/README.md for venv_phi setup instructions.
"""

import os
import torch
import torchaudio
from typing import Optional

from models.base import BaseAudioModel


class PhiModel(BaseAudioModel):
    """
    Wrapper for Phi-4 Multimodal model with audio capabilities.

    Phi-4 uses a Conformer encoder to process audio, which compresses
    log-mel spectrograms into embeddings that can be concatenated
    with text embeddings.

    Implements the BaseAudioModel interface for use with attack algorithms.
    """

    MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None
    ):
        """
        Initialize the Phi-4 model.

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

        print(f"Loading Phi-4 model: {model_id}")
        print(f"Device: {device}, Dtype: {dtype}")

        # Load model and processor
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            _attn_implementation="eager",
        ).to(device).eval()

        self.tokenizer = self.processor.tokenizer

        # MEL transform for Phi-4 (80-dim mel, different from Qwen's 128)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=1024,
            hop_length=160,
            win_length=1024,
            n_mels=80,
            power=1.0,
            f_min=0,
            f_max=8000,
        ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="magnitude", top_db=None
        ).to(device)

        # Special tokens for Phi-4
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.eos_token_id = self.tokenizer.eos_token_id

        print("Phi-4 model loaded successfully!")

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
        Convert waveform to log-mel spectrogram (differentiable).

        Args:
            wav: Audio waveform [T] or [1, T]

        Returns:
            Log-mel spectrogram [1, time_frames, 80]
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self._device, dtype=torch.float32)
        mel = self.mel_transform(wav)
        log_mel = self.amplitude_to_db(mel)
        # Phi-4 expects [batch, time, mels]
        return log_mel.transpose(-2, -1).contiguous().to(self._dtype)

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

        return proj_layer(feats).to(self._dtype)

    def _create_embeddings_from_mel(
        self,
        mel: torch.Tensor,
        target_text: str = ""
    ) -> torch.Tensor:
        """
        Create input embeddings from MEL spectrogram.

        Args:
            mel: Log-mel spectrogram [1, time, 80]
            target_text: Optional target text to append

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
        ).input_ids.to(self._device)
        after_ids = self.tokenizer(
            parts[1], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)

        inputs_embeds = torch.cat([
            txt_embeds(before_ids),
            audio_embeddings,
            txt_embeds(after_ids)
        ], dim=1)

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
        wav = wav.to(self._device, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # WAV → MEL (differentiable)
        log_mel = self.wav_to_mel(wav)
        audio_embeddings = self._get_audio_embeddings(log_mel)

        # Create prompt embeddings (with English-only system instruction)
        full_prompt = "<|system|>Always respond in English only.<|end|><|user|><|audio_1|><|end|><|assistant|>"
        parts = full_prompt.split("<|audio_1|>")

        txt_embeds = self.model.get_input_embeddings()
        before_ids = self.tokenizer(
            parts[0], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)
        after_ids = self.tokenizer(
            parts[1], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)
        target_ids = self.tokenizer(
            f"{target_text}<|end|><|endoftext|>",
            return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)

        inputs_embeds = torch.cat([
            txt_embeds(before_ids),
            audio_embeddings,
            txt_embeds(after_ids),
            txt_embeds(target_ids)
        ], dim=1)

        # Create labels (only target text is used for loss)
        labels = torch.full(
            (1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=self._device
        )
        labels[:, -target_ids.shape[1]:] = target_ids

        attn_mask = torch.ones(
            inputs_embeds.shape[:2], device=self._device, dtype=torch.long)

        # Forward pass with Phi-4 specific kwargs
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
            input_mode=torch.tensor(
                [2], dtype=torch.long, device=self._device),
            input_audio_embeds=audio_embeddings.to(self._dtype),
            audio_embed_sizes=torch.tensor(
                [audio_embeddings.shape[1]], device=self._device),
            audio_attention_mask=torch.ones(
                (1, audio_embeddings.shape[1]), dtype=torch.long, device=self._device
            ),
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
        Compute margin loss (Carlini-Wagner style) for Phi-4.

        Args:
            wav: Audio waveform tensor [T] or [1, T]
            target_text: Target text
            kappa: Margin (higher = more confident target)
            early_weight: Extra weight for first few tokens

        Returns:
            Loss tensor (scalar)
        """
        wav = wav.to(self._device, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # WAV → MEL (differentiable)
        log_mel = self.wav_to_mel(wav)
        audio_embeddings = self._get_audio_embeddings(log_mel)

        # Create prompt embeddings (with English-only system instruction)
        full_prompt = "<|system|>Always respond in English only.<|end|><|user|><|audio_1|><|end|><|assistant|>"
        parts = full_prompt.split("<|audio_1|>")

        txt_embeds = self.model.get_input_embeddings()
        before_ids = self.tokenizer(
            parts[0], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)
        after_ids = self.tokenizer(
            parts[1], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)
        target_ids = self.tokenizer(
            f"{target_text}<|end|><|endoftext|>",
            return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)

        inputs_embeds = torch.cat([
            txt_embeds(before_ids),
            audio_embeddings,
            txt_embeds(after_ids),
            txt_embeds(target_ids)
        ], dim=1)

        # Create labels
        labels = torch.full(
            (1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=self._device
        )
        labels[:, -target_ids.shape[1]:] = target_ids

        attn_mask = torch.ones(
            inputs_embeds.shape[:2], device=self._device, dtype=torch.long)

        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
            input_mode=torch.tensor(
                [2], dtype=torch.long, device=self._device),
            input_audio_embeds=audio_embeddings.to(self._dtype),
            audio_embed_sizes=torch.tensor(
                [audio_embeddings.shape[1]], device=self._device),
            audio_attention_mask=torch.ones(
                (1, audio_embeddings.shape[1]), dtype=torch.long, device=self._device
            ),
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
            wav = wav.to(self._device, dtype=torch.float32)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            log_mel = self.wav_to_mel(wav)
            audio_embeddings = self._get_audio_embeddings(log_mel)

            # Phi-4 prompt format (with English-only system instruction)
            full_prompt = "<|system|>Always respond in English only.<|end|><|user|><|audio_1|><|end|><|assistant|>"
            parts = full_prompt.split("<|audio_1|>")

            txt_embeds = self.model.get_input_embeddings()
            before_ids = self.tokenizer(
                parts[0], return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self._device)
            after_ids = self.tokenizer(
                parts[1], return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self._device)

            inputs_embeds = torch.cat([
                txt_embeds(before_ids),
                audio_embeddings,
                txt_embeds(after_ids)
            ], dim=1)

            attn_mask = torch.ones(
                inputs_embeds.shape[:2], device=self._device, dtype=torch.long)

            gen_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                eos_token_id=[self.end_token_id, self.eos_token_id],
                pad_token_id=self.tokenizer.pad_token_id,
                input_mode=torch.tensor(
                    [2], dtype=torch.long, device=self._device),
                input_audio_embeds=audio_embeddings.to(self._dtype),
                audio_embed_sizes=torch.tensor(
                    [audio_embeddings.shape[1]], device=self._device),
                audio_attention_mask=torch.ones(
                    (1, audio_embeddings.shape[1]), dtype=torch.long, device=self._device
                ),
            )

            return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
