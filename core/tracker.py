"""
Tracking and logging utilities for two-stage attacks.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class RLPGDStep:
    """Single step data for RL-PGD tracking."""
    step: int
    rewards: List[float]
    advantages: List[float]
    loss: float
    best_response: str
    best_score: float
    target_query: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class SemanticStep:
    """Single step data for Semantic PGD tracking."""
    step: int
    similarities: List[float]
    target_loss: float
    behavior_loss: float
    best_response: str
    target_text: str
    target_behavior: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RLPGDTracker:
    """Tracks Stage 1 (RL-PGD jailbreak) attack progress."""

    def __init__(self, save_dir: Path, case_id: str):
        """
        Initialize tracker.
        
        Args:
            save_dir: Directory to save tracking data.
            case_id: Identifier for this attack case.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.case_id = case_id
        self.steps: List[RLPGDStep] = []
        self.metadata: Dict[str, Any] = {
            "case_id": case_id,
            "start_time": datetime.now().isoformat(),
            "stage": "stage1_rl_pgd",
        }
        
    def update(
        self,
        step: int,
        rewards: List[float],
        advantages: List[float],
        loss: float,
        best_response: str,
        best_score: float,
        target_query: str,
    ):
        """Record a step's data."""
        step_data = RLPGDStep(
            step=step,
            rewards=rewards,
            advantages=advantages,
            loss=loss,
            best_response=best_response[:500],  # Truncate for storage
            best_score=best_score,
            target_query=target_query,
        )
        self.steps.append(step_data)
        
        # Auto-save every 10 steps
        if step % 10 == 0:
            self.save()
            
    def save(self):
        """Save tracking data to JSON file."""
        filepath = self.save_dir / f"{self.case_id}_stage1_tracker.json"
        data = {
            "metadata": self.metadata,
            "steps": [asdict(s) for s in self.steps],
            "summary": self._get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        log.debug(f"Saved Stage 1 tracker to {filepath}")
        
    def _get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.steps:
            return {}
        
        scores = [s.best_score for s in self.steps]
        return {
            "total_steps": len(self.steps),
            "final_score": scores[-1] if scores else 0,
            "max_score": max(scores) if scores else 0,
            "final_response": self.steps[-1].best_response if self.steps else "",
        }
        
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get the best result from all steps."""
        if not self.steps:
            return None
        best_step = max(self.steps, key=lambda s: s.best_score)
        return {
            "step": best_step.step,
            "score": best_step.best_score,
            "response": best_step.best_response,
        }


class SemanticTracker:
    """Tracks Stage 2 (Semantic PGD transfer) attack progress."""

    def __init__(self, save_dir: Path, case_id: str):
        """
        Initialize tracker.
        
        Args:
            save_dir: Directory to save tracking data.
            case_id: Identifier for this attack case.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.case_id = case_id
        self.steps: List[SemanticStep] = []
        self.metadata: Dict[str, Any] = {
            "case_id": case_id,
            "start_time": datetime.now().isoformat(),
            "stage": "stage2_semantic",
        }
        
    def update(
        self,
        step: int,
        similarities: List[float],
        target_loss: float,
        behavior_loss: float,
        best_response: str,
        target_text: str,
        target_behavior: str,
    ):
        """Record a step's data."""
        step_data = SemanticStep(
            step=step,
            similarities=similarities,
            target_loss=target_loss,
            behavior_loss=behavior_loss,
            best_response=best_response[:500],  # Truncate for storage
            target_text=target_text,
            target_behavior=target_behavior[:500],
        )
        self.steps.append(step_data)
        
        # Auto-save every 10 steps
        if step % 10 == 0:
            self.save()
            
    def save(self):
        """Save tracking data to JSON file."""
        filepath = self.save_dir / f"{self.case_id}_stage2_tracker.json"
        data = {
            "metadata": self.metadata,
            "steps": [asdict(s) for s in self.steps],
            "summary": self._get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        log.debug(f"Saved Stage 2 tracker to {filepath}")
        
    def _get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.steps:
            return {}
        
        best_sims = [max(s.similarities) if s.similarities else 0 for s in self.steps]
        return {
            "total_steps": len(self.steps),
            "final_similarity": best_sims[-1] if best_sims else 0,
            "max_similarity": max(best_sims) if best_sims else 0,
            "final_response": self.steps[-1].best_response if self.steps else "",
        }
        
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get the best result from all steps."""
        if not self.steps:
            return None
        
        # Find step with highest similarity
        best_step = max(
            self.steps, 
            key=lambda s: max(s.similarities) if s.similarities else 0
        )
        return {
            "step": best_step.step,
            "similarity": max(best_step.similarities) if best_step.similarities else 0,
            "response": best_step.best_response,
        }

