"""
MEL-level attack wrapper for Gemma model.

This wrapper enables attacks operating directly on mel spectrograms,
which can converge faster than WAV-level attacks due to the lower
dimensionality of the feature space.
"""

import torch
from typing import List, Optional, Tuple

from models.gemma import GemmaModel


class GemmaMelAttackWrapper:
    """
    MEL-level attack wrapper for Gemma model.
    
    Provides methods for computing loss and generating responses directly
    from mel spectrograms, bypassing the wav-to-mel conversion step during
    optimization. This enables faster convergence in Stage 1 attacks.
    
    Usage:
        model = GemmaModel()
        mel_wrapper = GemmaMelAttackWrapper(model)
        
        # Convert wav to mel once
        base_mel = mel_wrapper.wav_to_mel(wav)
        
        # Optimize perturbation in mel space
        perturbation = torch.zeros_like(base_mel, requires_grad=True)
        adv_mel = base_mel + perturbation
        
        # Compute loss directly from mel
        loss = mel_wrapper.compute_loss_from_mel(adv_mel, target_text)
    """
    
    # MEL-level attack parameters (different from WAV-level)
    DEFAULT_EPS = 5.0        # Larger epsilon for mel space (log scale)
    DEFAULT_ALPHA = 0.1      # Larger learning rate
    MEL_CLAMP_MIN = -12.0    # Min mel value (log scale)
    MEL_CLAMP_MAX = 4.0      # Max mel value (log scale)
    
    def __init__(self, gemma_model: GemmaModel):
        """
        Initialize MEL attack wrapper.
        
        Args:
            gemma_model: Initialized GemmaModel instance
        """
        self.gemma = gemma_model
        self.model = gemma_model.model
        self.processor = gemma_model.processor
        self.mel_transform = gemma_model.mel_transform
        self.device = gemma_model.device
        self.dtype = gemma_model.dtype
        
    @property
    def sample_rate(self) -> int:
        return self.gemma.sample_rate
    
    def wav_to_mel(self, wav: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.
        
        Args:
            wav: Audio waveform [1, T] or [T]
            detach: Whether to detach from computation graph (default True).
                    Set to True when using as base for perturbation optimization.
            
        Returns:
            Log-mel spectrogram [1, T', n_mels] where T' is time frames
        """
        wav = wav.to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        # Use the existing differentiable mel transform
        mel_log = self.mel_transform(wav)  # [1, T', 128]
        
        # Detach by default so only perturbation has gradients during optimization
        if detach:
            mel_log = mel_log.detach()
        
        return mel_log
    
    def mel_to_embeddings(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram directly to audio embeddings.
        
        This bypasses the wav-to-mel conversion, allowing gradients
        to flow directly from the mel perturbation to the loss.
        
        Args:
            mel: Log-mel spectrogram [1, T', n_mels]
            
        Returns:
            Audio embeddings [1, T'', D] where T'' is audio token count
        """
        # Clamp and convert to model dtype
        mel = torch.clamp(mel, min=-50, max=50)
        mel = mel.to(self.dtype)
        
        # Create padding mask (all False = no padding)
        padding_mask = torch.zeros(
            mel.shape[:2],
            dtype=torch.bool,
            device=self.device
        )
        
        # MEL → audio tower features
        audio_features, _ = self.model.audio_tower(mel, padding_mask)
        
        # Features → final embeddings
        audio_embeddings = self.model.model.embed_audio(
            inputs_embeds=audio_features
        )
        
        return audio_embeddings
    
    def create_attack_inputs(
        self,
        audio_embeddings: torch.Tensor,
        target_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create Gemma-format inputs for attack loss computation.
        
        Delegates to the underlying GemmaModel's implementation.
        """
        return self.gemma.create_attack_inputs(audio_embeddings, target_text)
    
    def compute_loss_from_mel(
        self,
        mel: torch.Tensor,
        target_text: str
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss directly from mel spectrogram.
        
        Args:
            mel: Log-mel spectrogram [1, T', n_mels]
            target_text: Target text to generate
            
        Returns:
            Loss tensor (scalar)
        """
        # MEL → embeddings (differentiable)
        audio_emb = self.mel_to_embeddings(mel)
        
        # Create attack inputs
        full_emb, attn_mask, labels = self.create_attack_inputs(
            audio_emb, target_text
        )
        
        # Forward pass
        with torch.amp.autocast(self.device, dtype=self.dtype):
            outputs = self.model(
                inputs_embeds=full_emb,
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
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from mel spectrogram.
        
        Args:
            mel: Log-mel spectrogram [1, T', n_mels]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated text string
        """
        with torch.no_grad():
            # MEL → embeddings
            audio_emb = self.mel_to_embeddings(mel)
            
            # Create generation inputs (empty target)
            full_emb, attn_mask, _ = self.create_attack_inputs(audio_emb, "")
            
            # Generate
            gen_ids = self.model.generate(
                inputs_embeds=full_emb,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                num_beams=num_beams,
            )
            
            response = self.processor.batch_decode(
                gen_ids, skip_special_tokens=True
            )[0]
        
        return response
    
    def generate_candidates_from_mel(
        self,
        mel: torch.Tensor,
        num_candidates: int = 6
    ) -> List[Tuple[str, str]]:
        """
        Generate multiple candidate responses from mel spectrogram.
        
        Uses various decoding strategies (greedy, beam search, sampling)
        to produce diverse candidates for RL-style scoring.
        
        Args:
            mel: Log-mel spectrogram [1, T', n_mels]
            num_candidates: Number of candidates to generate (default 6)
            
        Returns:
            List of (strategy_name, response) tuples
        """
        candidates = []
        
        # 1. Greedy decoding (deterministic baseline)
        response = self.generate_from_mel(
            mel, do_sample=False, max_tokens=128
        )
        candidates.append(("greedy", response))
        
        # 2. Beam search variants
        for beam_size in [3, 5]:
            response = self.generate_from_mel(
                mel, do_sample=False, num_beams=beam_size, max_tokens=128
            )
            candidates.append((f"beam_{beam_size}", response))
        
        # 3. Sampling variants with different temperatures
        for temp in [0.7, 1.0, 1.3]:
            response = self.generate_from_mel(
                mel, do_sample=True, temperature=temp, top_p=0.9, max_tokens=128
            )
            candidates.append((f"sample_t{temp}", response))
        
        # Return up to num_candidates
        return candidates[:num_candidates]
    
    def clamp_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Clamp mel spectrogram to valid range.
        
        Args:
            mel: Log-mel spectrogram
            
        Returns:
            Clamped mel spectrogram
        """
        return torch.clamp(mel, self.MEL_CLAMP_MIN, self.MEL_CLAMP_MAX)

