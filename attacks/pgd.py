"""
Basic Projected Gradient Descent (PGD) attack on audio waveforms.
"""

import torch
from typing import Optional, Literal

from attacks.base import BaseWavAttacker, AttackResult
from models.base import BaseAudioModel


class PGDAttacker(BaseWavAttacker):
    """
    Standard PGD attack for audio models.

    Iteratively perturbs the audio waveform using gradient ascent on the
    loss function, projecting back to the epsilon-ball after each step.

    Supports both cross-entropy and margin loss.
    """

    def __init__(
        self,
        model: BaseAudioModel,
        eps: float = 0.1,
        alpha: float = 0.005,
        loss_type: Literal["ce", "margin"] = "margin",
        kappa: float = 5.0,
        use_lowpass: bool = False,
        lowpass_cutoff: float = 2000.0,
        verbose: bool = True
    ):
        """
        Initialize PGD attacker.

        Args:
            model: Audio model to attack
            eps: Maximum L-infinity perturbation
            alpha: Step size
            loss_type: "ce" for cross-entropy, "margin" for margin loss
            kappa: Margin parameter (only for margin loss)
            use_lowpass: Whether to lowpass filter gradients
            lowpass_cutoff: Cutoff frequency for lowpass (Hz)
            verbose: Print progress
        """
        super().__init__(
            model=model,
            eps=eps,
            alpha=alpha,
            use_lowpass=use_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            verbose=verbose
        )
        self.loss_type = loss_type
        self.kappa = kappa

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str
    ) -> torch.Tensor:
        """Compute loss based on loss_type."""
        if self.loss_type == "margin":
            return self.model.compute_margin_loss(wav, target_text, kappa=self.kappa)
        else:
            return self.model.compute_loss(wav, target_text)

    def attack(
        self,
        wav: torch.Tensor,
        target_text: str,
        steps: int = 100,
        random_init: bool = True,
        check_every: int = 20,
        early_stop: bool = True
    ) -> AttackResult:
        """
        Perform PGD attack.

        Args:
            wav: Original audio [1, T]
            target_text: Target text to force
            steps: Number of PGD iterations
            random_init: Random initialization of perturbation
            check_every: Check generation every N steps
            early_stop: Stop early if target achieved

        Returns:
            AttackResult with adversarial audio
        """
        wav = wav.to(self.model.device)

        # Get original output
        self.log("Getting original model output...")
        original_output = self.model.generate(wav)
        self.log(f"Original: {original_output}")
        self.log(f"Target: {target_text}")

        # Initialize perturbation
        delta = self.init_perturbation(wav, random_init)

        # History tracking
        history = {
            "loss": [],
            "outputs": [],
            "steps": []
        }

        best_delta = delta.data.clone()
        best_loss = float('inf')
        success = False

        self.log(f"\nStarting PGD attack ({self.loss_type} loss)")
        self.log(f"eps={self.eps}, alpha={self.alpha}, steps={steps}")
        self.log("-" * 50)

        for step in range(steps):
            # Forward pass
            adv_wav = wav + delta
            loss = self.compute_loss(adv_wav, target_text)

            # Backward pass
            loss.backward()

            # Get and process gradient
            grad = delta.grad.data
            grad = self.process_gradient(grad)

            # PGD step (gradient descent to minimize loss)
            delta.data = delta.data - self.alpha * grad.sign()

            # Project to epsilon ball
            delta = self.clamp_perturbation(delta, wav)

            # Zero gradients
            delta.grad.zero_()

            # Track best
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            history["loss"].append(loss_val)

            # Progress logging
            if (step + 1) % 10 == 0:
                self.log(f"Step {step+1}/{steps} | Loss: {loss_val:.4f}")

            # Periodic generation check
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    adv_wav = wav + delta
                    output = self.model.generate(adv_wav)

                history["outputs"].append(output)
                history["steps"].append(step + 1)

                self.log(f"  → Output: {output[:80]}...")

                # Check for success
                if target_text.lower() in output.lower():
                    self.log(f"\n✓ Target achieved at step {step+1}!")
                    success = True
                    if early_stop:
                        break

        # Final adversarial audio
        with torch.no_grad():
            final_wav = wav + best_delta
            final_output = self.model.generate(final_wav)

        # Check final success
        if target_text.lower() in final_output.lower():
            success = True

        # Compute SNR
        snr = self.compute_snr(wav, final_wav)

        self.log("\n" + "=" * 50)
        self.log("Attack Complete")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"SNR: {snr:.2f} dB")
        self.log(f"Success: {success}")
        self.log(f"Final Output: {final_output}")
        self.log("=" * 50)

        return AttackResult(
            original_wav=wav.detach().cpu(),
            adversarial_wav=final_wav.detach().cpu(),
            perturbation=best_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=step + 1,
            history=history
        )
