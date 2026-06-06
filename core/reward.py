"""
Reward computation for RL-based adversarial attacks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from difflib import SequenceMatcher
from typing import Tuple, Dict, Optional


class RewardComputer:
    """
    Computes reward signals for RL-PGD attacks using:
    - Semantic similarity (sentence transformers)
    - Exact/partial text matches
    - Character-level similarity

    The reward guides the attack toward generating the target text.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize the reward computer.

        Args:
            model_name: Sentence transformer model name
            device: Device for the sentence transformer (CPU recommended to save GPU)
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(device)
        self.device = device

        # Cache for target embeddings
        self._target_cache: Dict[str, np.ndarray] = {}

    def get_target_embedding(self, target: str) -> np.ndarray:
        """Get cached embedding for target text."""
        if target not in self._target_cache:
            self._target_cache[target] = self.model.encode(
                target, convert_to_numpy=True
            )
        return self._target_cache[target]

    def compute_semantic_similarity(
        self,
        output: str,
        target_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between output and target embeddings.

        Returns:
            Similarity score in [-1, 1], higher is better
        """
        output_embedding = self.model.encode(output, convert_to_numpy=True)

        similarity = float(F.cosine_similarity(
            torch.from_numpy(target_embedding).unsqueeze(0),
            torch.from_numpy(output_embedding).unsqueeze(0)
        ).item())

        return similarity

    def compute_reward(
        self,
        output: str,
        target: str,
        target_embedding: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict]:
        """
        Compute composite reward for model output vs target.

        Args:
            output: Model's generated text
            target: Target text we want
            target_embedding: Pre-computed target embedding (optional)
            weights: Custom weights for reward components

        Returns:
            reward: Float reward value
            info: Dict with breakdown of reward components
        """
        if target_embedding is None:
            target_embedding = self.get_target_embedding(target)

        # Default weights
        if weights is None:
            weights = {
                "semantic": 50.0,
                "first_token": 15.0,
                "token_overlap": 10.0,
                "char_similarity": 10.0,
                "length_penalty": 5.0,
            }

        info = {}

        # 1. EXACT MATCH (jackpot!)
        if target.lower().strip() in output.lower():
            info['exact_match'] = True
            info['semantic_similarity'] = 1.0
            info['first_token_match'] = True
            info['token_overlap'] = 1.0
            info['char_similarity'] = 1.0
            info['length_ratio'] = 1.0
            return 100.0, info

        info['exact_match'] = False
        reward = 0.0

        # 2. SEMANTIC SIMILARITY (main signal)
        similarity = self.compute_semantic_similarity(output, target_embedding)
        semantic_reward = similarity * weights["semantic"]
        reward += semantic_reward
        info['semantic_similarity'] = similarity
        info['semantic_reward'] = semantic_reward

        # 3. FIRST TOKEN MATCH
        target_words = target.lower().split()
        output_words = output.lower().split() if output.strip() else []

        first_match = False
        if output_words and target_words:
            first_match = output_words[0] == target_words[0]

        if first_match:
            reward += weights["first_token"]
        info['first_token_match'] = first_match

        # 4. TOKEN OVERLAP (bag of words)
        if target_words and output_words:
            target_set = set(target_words)
            output_set = set(output_words)
            overlap = len(target_set & output_set) / len(target_set)
            overlap_reward = overlap * weights["token_overlap"]
            reward += overlap_reward
            info['token_overlap'] = overlap
            info['overlap_reward'] = overlap_reward
        else:
            info['token_overlap'] = 0.0
            info['overlap_reward'] = 0.0

        # 5. CHARACTER-LEVEL SIMILARITY
        char_sim = SequenceMatcher(
            None,
            output.lower()[:100],  # Limit to first 100 chars
            target.lower()[:100]
        ).ratio()
        char_reward = char_sim * weights["char_similarity"]
        reward += char_reward
        info['char_similarity'] = char_sim
        info['char_reward'] = char_reward

        # 6. LENGTH PENALTY (prefer similar length)
        if output.strip():
            length_ratio = min(len(output), len(target)) / \
                max(len(output), len(target), 1)
            length_bonus = length_ratio * weights["length_penalty"]
            reward += length_bonus
            info['length_ratio'] = length_ratio
            info['length_bonus'] = length_bonus
        else:
            info['length_ratio'] = 0.0
            info['length_bonus'] = 0.0

        return reward, info

    def clear_cache(self) -> None:
        """Clear the target embedding cache."""
        self._target_cache.clear()
