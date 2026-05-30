"""
Low-VRAM Trainer Orchestrator
==============================

Main entry point for low-VRAM training.
Orchestrates all components following Clean Architecture.

This class depends on abstractions (interfaces), not concrete implementations.
Concrete implementations are injected via the factory.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from go1.tools.low_vram.core.interfaces import (
    FeatureCache,
    MemoryConfig,
    MemoryManager,
    MemorySnapshot,
    ModelFreezer,
    ProgressReporter,
    TrainingConfig,
    TrainingStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Mutable state container for trainer (separates state from logic)."""
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    total_tokens_trained: int = 0


class ConsoleProgressReporter(ProgressReporter):
    """Simple console-based progress reporter."""
    
    def __init__(self, log_interval: int = 10):
        self._log_interval = log_interval
        self._epoch_start_time: float = 0
    
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self._epoch_start_time = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{total_epochs}")
        logger.info(f"{'='*60}")
    
    def on_step(
        self,
        step: int,
        total_steps: int,
        loss: float,
        metrics: Dict[str, float],
    ) -> None:
        if step % self._log_interval == 0:
            gpu_mem = metrics.get("gpu_memory_mb", 0)
            lr = metrics.get("learning_rate", 0)
            logger.info(
                f"Step {step}/{total_steps} | "
                f"Loss: {loss:.4f} | "
                f"GPU: {gpu_mem:.0f}MB | "
                f"LR: {lr:.2e}"
            )
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        elapsed = time.time() - self._epoch_start_time
        avg_loss = metrics.get("avg_loss", 0)
        logger.info(
            f"Epoch {epoch + 1} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
    
    def on_checkpoint(self, path: str) -> None:
        logger.info(f"Checkpoint saved: {path}")


class LowVRAMTrainer:
    """
    Main orchestrator for low-VRAM training.
    
    Follows:
        - Dependency Inversion: Depends on interfaces, not implementations
        - Single Responsibility: Only orchestrates, doesn't implement components
        - Open/Closed: Extend via new strategies, don't modify this class
    
    Security (STRIDE):
        - Tampering: Checksums for cached features
        - DoS: Memory limits prevent OOM
        - Repudiation: Training logs for audit trail
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        memory_manager: MemoryManager,
        training_strategy: TrainingStrategy,
        feature_cache: Optional[FeatureCache] = None,
        model_freezer: Optional[ModelFreezer] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_dir: Optional[str] = None,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize trainer with injected dependencies.
        
        Args:
            model: The model to train
            optimizer: Optimizer instance
            memory_manager: Memory manager for GPU/CPU orchestration
            training_strategy: Strategy for training steps
            feature_cache: Optional cache for pre-computed features
            model_freezer: Optional freezer for parameter management
            progress_reporter: Optional progress reporter
            scheduler: Optional learning rate scheduler
            checkpoint_dir: Directory for saving checkpoints
            max_grad_norm: Maximum gradient norm for clipping
        """
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._memory_manager = memory_manager
        self._strategy = training_strategy
        self._cache = feature_cache
        self._freezer = model_freezer
        self._reporter = progress_reporter or ConsoleProgressReporter()
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._max_grad_norm = max_grad_norm
        
        # State
        self._state = TrainerState()
        
        # Device
        self._device = next(model.parameters()).device
        
        # Create checkpoint directory
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def state(self) -> TrainerState:
        """Get current trainer state (read-only access advised)."""
        return self._state
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        eval_interval: int = 1,
        checkpoint_interval: int = 1,
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            num_epochs: Number of epochs to train
            eval_dataloader: Optional evaluation data loader
            eval_interval: Epochs between evaluations
            checkpoint_interval: Epochs between checkpoints
            
        Returns:
            Training history dictionary
        """
        history: Dict[str, list] = {
            "train_loss": [],
            "eval_loss": [],
            "memory_usage": [],
        }
        
        total_steps = len(train_dataloader) * num_epochs
        config = self._strategy.get_config()
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        
        # Initial memory snapshot
        initial_snapshot = self._memory_manager.get_snapshot()
        logger.info(
            f"Initial GPU memory: {initial_snapshot.gpu_allocated_mb:.1f}MB / "
            f"{initial_snapshot.gpu_max_mb:.1f}MB"
        )
        
        for epoch in range(num_epochs):
            self._state.epoch = epoch
            self._reporter.on_epoch_start(epoch, num_epochs)
            
            # Training epoch
            epoch_metrics = self._train_epoch(train_dataloader)
            history["train_loss"].append(epoch_metrics["avg_loss"])
            
            self._reporter.on_epoch_end(epoch, epoch_metrics)
            
            # Evaluation
            if eval_dataloader and (epoch + 1) % eval_interval == 0:
                eval_metrics = self._evaluate(eval_dataloader)
                history["eval_loss"].append(eval_metrics["avg_loss"])
            
            # Checkpointing
            if self._checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch, epoch_metrics)
            
            # Memory tracking
            snapshot = self._memory_manager.get_snapshot()
            history["memory_usage"].append(snapshot.gpu_allocated_mb)
            
            # Clear cache between epochs
            self._memory_manager.clear_cache()
        
        logger.info("Training complete!")
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one training epoch."""
        self._model.train()
        
        epoch_loss = 0.0
        num_steps = len(dataloader)
        
        for step, batch in enumerate(dataloader):
            self._state.global_step += 1
            
            # Training step
            loss, metrics = self._strategy.training_step(
                self._model,
                batch,
                self._state.global_step,
            )
            
            epoch_loss += metrics["loss"]
            
            # Optimizer step (respects gradient accumulation)
            if not self._strategy.should_accumulate(self._state.global_step):
                opt_metrics = self._strategy.optimizer_step(
                    self._optimizer,
                    self._scheduler,
                    self._max_grad_norm,
                )
                metrics.update(opt_metrics)
            
            # Report progress
            self._reporter.on_step(
                step,
                num_steps,
                metrics["loss"],
                metrics,
            )
        
        return {
            "avg_loss": epoch_loss / num_steps,
            "total_steps": num_steps,
        }
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self._model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            batch = {
                k: v.to(self._device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            
            outputs = self._model(**batch)
            
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs
            
            total_loss += loss.item()
            num_batches += 1
        
        self._model.train()
        
        return {"avg_loss": total_loss / max(1, num_batches)}
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        if self._checkpoint_dir is None:
            return
        
        checkpoint_path = self._checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self._state.global_step,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "metrics": metrics,
            "best_loss": self._state.best_loss,
        }
        
        if self._scheduler:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self._reporter.on_checkpoint(str(checkpoint_path))
        
        # Update best loss
        if metrics.get("avg_loss", float("inf")) < self._state.best_loss:
            self._state.best_loss = metrics["avg_loss"]
            best_path = self._checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: loss={self._state.best_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training state from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self._scheduler and "scheduler_state_dict" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self._state.epoch = checkpoint["epoch"]
        self._state.global_step = checkpoint["global_step"]
        self._state.best_loss = checkpoint.get("best_loss", float("inf"))
        
        logger.info(f"Loaded checkpoint from epoch {self._state.epoch + 1}")
    
    def get_memory_report(self) -> str:
        """Generate memory usage report."""
        snapshot = self._memory_manager.get_snapshot()
        
        report = [
            "=" * 50,
            "MEMORY REPORT",
            "=" * 50,
            f"GPU Allocated: {snapshot.gpu_allocated_mb:.1f} MB",
            f"GPU Reserved:  {snapshot.gpu_reserved_mb:.1f} MB",
            f"GPU Maximum:   {snapshot.gpu_max_mb:.1f} MB",
            f"GPU Usage:     {snapshot.gpu_allocated_mb / snapshot.gpu_max_mb * 100:.1f}%",
            "",
            f"CPU (offloaded): {snapshot.cpu_used_mb:.1f} MB",
        ]
        
        if self._cache:
            stats = self._cache.get_stats()
            report.extend([
                "",
                "Feature Cache:",
                f"  Items:    {stats['cached_items']}",
                f"  Size:     {stats['total_size_mb']:.1f} MB",
                f"  Hit Rate: {stats['hit_rate']:.1%}",
            ])
        
        if self._freezer:
            frozen = self._freezer.get_frozen_params(self._model)
            trainable = self._freezer.get_trainable_params(self._model)
            report.extend([
                "",
                "Model Parameters:",
                f"  Frozen:    {frozen:,} ({frozen * 4 / 1e6:.1f} MB)",
                f"  Trainable: {trainable:,} ({trainable * 4 / 1e6:.1f} MB)",
            ])
        
        report.append("=" * 50)
        
        return "\n".join(report)
    
    def __repr__(self) -> str:
        return (
            f"LowVRAMTrainer("
            f"epoch={self._state.epoch}, "
            f"step={self._state.global_step}, "
            f"strategy={self._strategy})"
        )


__all__ = ["LowVRAMTrainer", "TrainerState", "ConsoleProgressReporter"]
