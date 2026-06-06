"""
Gemma 3n audio model wrapper for adversarial attacks.
"""

import os
import torch
from typing import Tuple, Optional

from models.base import BaseAudioModel
from core.mel import DifferentiableMelTransform, GEMMA_MEL_CONFIG


class GemmaModel(BaseAudioModel):
    """
    Wrapper for Google's Gemma 3n model with audio capabilities.

    Implements the BaseAudioModel interface for use with attack algorithms.
    """

    MODEL_ID = "google/gemma-3n-E4B-it"
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None
    ):
        """
        Initialize the Gemma model.

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

        print(f"Loading Gemma model: {model_id}")
        print(f"Device: {device}, Dtype: {dtype}")

        # Load model and processor
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=token
        ).to(device).eval()

        # Initialize differentiable MEL transform with Gemma's config
        self.mel_transform = DifferentiableMelTransform(
            **GEMMA_MEL_CONFIG,
            device=device
        )

        print("Gemma model loaded successfully!")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def wav_to_embeddings(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to Gemma audio embeddings (differentiable).

        Pipeline:
        1. WAV → MEL spectrogram (differentiable)
        2. MEL → audio_tower features
        3. Features → embed_audio embeddings
        """
        # Ensure wav is on correct device
        wav = wav.to(self._device)

        # WAV → log-MEL spectrogram [1, T', 128]
        mel_log = self.mel_transform(wav)
        mel_log = torch.clamp(mel_log, min=-50, max=50)
        mel_log = mel_log.to(self._dtype)

        # Create padding mask (all False = no padding)
        padding_mask = torch.zeros(
            mel_log.shape[:2],
            dtype=torch.bool,
            device=self._device
        )

        # MEL → audio tower features
        audio_features, _ = self.model.audio_tower(mel_log, padding_mask)

        # Features → final embeddings
        audio_embeddings = self.model.model.embed_audio(
            inputs_embeds=audio_features)

        return audio_embeddings

    def create_attack_inputs(
        self,
        audio_embeddings: torch.Tensor,
        target_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create Gemma-format inputs for attack loss computation.

        Uses Gemma's chat format:
        <start_of_turn>user
        <audio_soft_token><end_of_turn>
        <start_of_turn>model
        {target_text}

        IMPORTANT: Loss is computed ONLY on target_text tokens.
        All prompt/template tokens are masked with -100.
        """
        # Gemma chat template
        prompt = f"<start_of_turn>user\n<audio_soft_token><end_of_turn>\n<start_of_turn>model\n{target_text}"

        # Tokenize full prompt
        tokenized = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self._device)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Tokenize ONLY target_text to find its token length
        target_tokens = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self._device)
        target_len = target_tokens.size(1)

        # Find audio token position
        audio_token_id = self.processor.tokenizer.audio_token_id
        audio_pos = (input_ids[0] == audio_token_id).nonzero(
            as_tuple=True)[0][0]

        # Get text embeddings
        text_embeddings = self.model.get_input_embeddings()(input_ids)

        # Splice: [text_before] + [audio_emb] + [text_after]
        audio_len = audio_embeddings.size(1)
        full_embeddings = torch.cat([
            text_embeddings[:, :audio_pos],
            audio_embeddings,
            text_embeddings[:, audio_pos + 1:]
        ], dim=1)

        # Calculate total sequence length after splicing
        total_len = full_embeddings.size(1)

        # Create labels: -100 everywhere EXCEPT last target_len positions
        # This ensures loss is computed ONLY on target_text tokens
        labels = torch.full(
            (1, total_len),
            -100,
            dtype=input_ids.dtype,
            device=self._device
        )
        # Copy target token IDs to the end of labels
        if target_len > 0:
            labels[0, -target_len:] = target_tokens[0]

        # Create attention mask for full sequence
        full_attention_mask = torch.cat([
            attention_mask[:, :audio_pos],
            torch.ones((1, audio_len), dtype=attention_mask.dtype,
                       device=self._device),
            attention_mask[:, audio_pos + 1:]
        ], dim=1)

        return full_embeddings, full_attention_mask, labels

    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """Generate text from audio using Gemma."""
        with torch.no_grad():
            # Get audio embeddings
            audio_emb = self.wav_to_embeddings(wav)

            # Create generation inputs (empty target)
            full_emb, attn_mask, _ = self.create_attack_inputs(audio_emb, "")

            # Generate
            gen_ids = self.model.generate(
                inputs_embeds=full_emb,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
            )

            response = self.processor.batch_decode(
                gen_ids, skip_special_tokens=True
            )[0]

        return response

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str
    ) -> torch.Tensor:
        """Compute cross-entropy loss for target text."""
        audio_emb = self.wav_to_embeddings(wav)
        full_emb, attn_mask, labels = self.create_attack_inputs(
            audio_emb, target_text)

        with torch.amp.autocast(self._device, dtype=self._dtype):
            outputs = self.model(
                inputs_embeds=full_emb,
                attention_mask=attn_mask,
                labels=labels
            )

        return outputs.loss

    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ):
        """Forward pass through Gemma."""
        with torch.amp.autocast(self._device, dtype=self._dtype):
            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )

    def compute_margin_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        kappa: float = 5.0,
        early_weight: float = 5.0
    ) -> torch.Tensor:
        """Compute margin loss (Carlini-Wagner style) for Gemma."""
        audio_emb = self.wav_to_embeddings(wav)
        full_emb, attn_mask, labels = self.create_attack_inputs(
            audio_emb, target_text)

        with torch.amp.autocast(self._device, dtype=self._dtype):
            outputs = self.model(
                inputs_embeds=full_emb,
                attention_mask=attn_mask,
                labels=labels
            )

        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Find valid positions
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return outputs.loss

        valid_positions = valid_mask[0].nonzero(as_tuple=True)[0]
        valid_logits = shift_logits[0, valid_positions]
        valid_labels = shift_labels[0, valid_positions]

        # Target token logits
        target_logits = valid_logits.gather(
            1, valid_labels.unsqueeze(1)).squeeze(1)

        # Max non-target logits
        label_mask = torch.ones_like(valid_logits, dtype=torch.bool)
        label_mask.scatter_(1, valid_labels.unsqueeze(1), False)
        masked_logits = valid_logits.masked_fill(~label_mask, float('-inf'))
        top_other_logits = masked_logits.max(dim=-1).values

        # Margin loss
        margin_losses = torch.clamp(
            top_other_logits - target_logits + kappa, min=0)

        # Weight early tokens
        num_tokens = len(margin_losses)
        weights = torch.ones(num_tokens, device=margin_losses.device)
        num_early = min(3, num_tokens)
        weights[:num_early] = early_weight

        loss = (margin_losses * weights).sum() / weights.sum()
        return loss
