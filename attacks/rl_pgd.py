"""
RL-PGD: Reinforcement Learning enhanced Projected Gradient Descent attack.

Combines gradient-based optimization with generation-based reward signals
using semantic similarity to guide the attack.
"""

import torch
import numpy as np
from typing import Optional, Literal

from attacks.base import BaseWavAttacker, AttackResult
from models.base import BaseAudioModel
from core.reward import RewardComputer


class RLPGDAttacker(BaseWavAttacker):
    """
    RL-enhanced PGD attack.

    Key features:
    1. Uses semantic similarity (sentence transformers) for reward
    2. Checkpoints good perturbations and backtracks from bad directions
    3. Combines margin loss (gradient signal) with generation checking (reward)
    4. Exploration noise for escaping local minima
    """

    def __init__(
        self,
        model: BaseAudioModel,
        eps: float = 0.1,
        alpha: float = 0.005,
        kappa: float = 5.0,
        use_lowpass: bool = False,
        lowpass_cutoff: float = 2000.0,
        reward_model: str = "all-MiniLM-L6-v2",
        verbose: bool = True
    ):
        """
        Initialize RL-PGD attacker.

        Args:
            model: Audio model to attack
            eps: Maximum L-infinity perturbation
            alpha: Step size
            kappa: Margin parameter for loss
            use_lowpass: Whether to lowpass filter gradients
            lowpass_cutoff: Cutoff frequency for lowpass (Hz)
            reward_model: Sentence transformer model for reward
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
        self.kappa = kappa

        # Initialize reward computer
        self.log("Loading reward model...")
        self.reward_computer = RewardComputer(
            model_name=reward_model,
            device="cpu"  # Keep on CPU to save GPU memory
        )

    def attack(
        self,
        wav: torch.Tensor,
        target_text: str,
        steps: int = 200,
        random_init: bool = True,
        check_every: int = 20,
        backtrack_threshold: float = 0.7,
        exploration_noise: float = 0.01,
        early_stop: bool = True
    ) -> AttackResult:
        """
        Perform RL-PGD attack.

        Args:
            wav: Original audio [1, T]
            target_text: Target text to force
            steps: Number of PGD iterations
            random_init: Random initialization of perturbation
            check_every: Check generation and compute reward every N steps
            backtrack_threshold: Backtrack if reward drops below this fraction of best
            exploration_noise: Noise magnitude for exploration
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

        # Pre-compute target embedding for reward
        target_embedding = self.reward_computer.get_target_embedding(
            target_text)

        # Initialize perturbation
        delta = self.init_perturbation(wav, random_init)

        # RL tracking
        best_delta = delta.data.clone()
        best_reward = -float('inf')
        best_output = ""
        checkpoint_delta = delta.data.clone()
        checkpoint_reward = -float('inf')

        # History
        history = {
            "loss": [],
            "reward": [],
            "outputs": [],
            "steps": [],
            "backtracks": []
        }

        success = False

        self.log(f"\nStarting RL-PGD attack")
        self.log(f"eps={self.eps}, alpha={self.alpha}, steps={steps}")
        self.log(
            f"check_every={check_every}, backtrack_threshold={backtrack_threshold}")
        self.log("-" * 60)

        for step in range(steps):
            # Forward pass
            adv_wav = wav + delta
            loss = self.model.compute_margin_loss(
                adv_wav, target_text, kappa=self.kappa)

            # Backward pass
            loss.backward()

            # Get and process gradient
            grad = delta.grad.data
            grad = self.process_gradient(grad)

            # PGD step
            delta.data = delta.data - self.alpha * grad.sign()

            # Add exploration noise occasionally
            if step % 50 == 0 and step > 0:
                noise = torch.randn_like(delta.data) * exploration_noise
                delta.data = delta.data + noise

            # Project to epsilon ball
            delta = self.clamp_perturbation(delta, wav)
            delta.grad.zero_()

            loss_val = loss.item()
            history["loss"].append(loss_val)

            # Progress logging
            if (step + 1) % 10 == 0:
                self.log(f"Step {step+1}/{steps} | Loss: {loss_val:.4f}")

            # RL check: compute reward from actual generation
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    adv_wav = wav + delta
                    output = self.model.generate(adv_wav)

                # Compute reward
                reward, info = self.reward_computer.compute_reward(
                    output, target_text, target_embedding
                )

                history["reward"].append(reward)
                history["outputs"].append(output)
                history["steps"].append(step + 1)

                self.log(f"  → Output: {output[:60]}...")
                self.log(
                    f"  → Reward: {reward:.2f} (sem={info.get('semantic_similarity', 0):.3f})")

                # Check for exact match (success!)
                if info.get('exact_match', False):
                    self.log(f"\n✓ Target achieved at step {step+1}!")
                    best_delta = delta.data.clone()
                    best_reward = reward
                    best_output = output
                    success = True
                    if early_stop:
                        break

                # Update best
                if reward > best_reward:
                    best_reward = reward
                    best_delta = delta.data.clone()
                    best_output = output
                    self.log(f"  ★ New best reward: {reward:.2f}")

                    # Update checkpoint
                    checkpoint_delta = delta.data.clone()
                    checkpoint_reward = reward

                # Backtracking check
                elif checkpoint_reward > 0 and reward < checkpoint_reward * backtrack_threshold:
                    self.log(
                        f"  ↩ Backtracking (reward dropped from {checkpoint_reward:.2f} to {reward:.2f})")
                    delta.data = checkpoint_delta.clone()
                    # Add noise to escape
                    delta.data = delta.data + \
                        torch.randn_like(delta.data) * exploration_noise * 2
                    delta = self.clamp_perturbation(delta, wav)
                    history["backtracks"].append(step + 1)

        # Final evaluation
        with torch.no_grad():
            final_wav = wav + best_delta
            final_output = self.model.generate(final_wav)

        # Final reward
        final_reward, final_info = self.reward_computer.compute_reward(
            final_output, target_text, target_embedding
        )

        if final_info.get('exact_match', False):
            success = True

        # Use best output if it was better
        if best_reward > final_reward:
            final_output = best_output
            final_reward = best_reward

        # Compute SNR
        snr = self.compute_snr(wav, final_wav)

        self.log("\n" + "=" * 60)
        self.log("RL-PGD Attack Complete")
        self.log(f"Final Reward: {final_reward:.2f}")
        self.log(f"Best Reward: {best_reward:.2f}")
        self.log(f"SNR: {snr:.2f} dB")
        self.log(f"Backtracks: {len(history['backtracks'])}")
        self.log(f"Success: {success}")
        self.log(f"Final Output: {final_output}")
        self.log("=" * 60)

        return AttackResult(
            original_wav=wav.detach().cpu(),
            adversarial_wav=final_wav.detach().cpu(),
            perturbation=best_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=history["loss"][-1] if history["loss"] else 0.0,
            steps_taken=step + 1,
            history=history
        )
