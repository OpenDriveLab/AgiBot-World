"""
Training Strategy Implementation
=================================

Implements low-VRAM training with gradient accumulation and mixed precision.
Single Responsibility: Training step logic only.

Design Patterns:
    - Strategy Pattern: Swappable training strategies
    - Template Method: Common training flow, customizable steps
"""

import logging
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast

from go1.tools.low_vram.core.interfaces import (
    TrainingConfig,
    TrainingStrategy,
    MemoryManager,
)

logger = logging.getLogger(__name__)


class GradientAccumulationMixin:
    """
    Mixin for gradient accumulation logic.
    
    Follows DRY: Shared logic extracted into reusable mixin.
    """
    
    def _should_accumulate_impl(
        self,
        step: int,
        accumulation_steps: int
    ) -> bool:
        """
        Determine if we should accumulate gradients (not step optimizer).
        
        Pure function - no side effects.
        """
        return (step + 1) % accumulation_steps != 0
    
    def _scale_loss(self, loss: Tensor, accumulation_steps: int) -> Tensor:
        """Scale loss for gradient accumulation."""
        return loss / accumulation_steps


class LowVRAMTrainingStrategy(TrainingStrategy, GradientAccumulationMixin):
    """
    Memory-efficient training strategy for 4-8GB GPUs.
    
    Features:
        - Gradient accumulation (effective batch size with micro batches)
        - Mixed precision (FP16 forward, FP32 gradients)
        - Gradient checkpointing support
        - Optional CPU offloading of optimizer states
    
    Single Responsibility: Only handles training step execution.
    Does NOT manage memory directly (uses MemoryManager interface).
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        memory_manager: Optional[MemoryManager] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize training strategy.
        
        Args:
            config: Frozen training configuration
            memory_manager: Optional memory manager for logging
            device: Target device (defaults to cuda if available)
        """
        self._config = config
        self._memory_manager = memory_manager
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        # Mixed precision scaler
        self._scaler: Optional[GradScaler] = None
        self._use_amp = torch.cuda.is_available()
        
        # Metrics tracking
        self._step_count = 0
        self._accumulated_loss = 0.0
        
    def _initialize_scaler(self) -> None:
        """Lazy initialization of GradScaler (allows serialization)."""
        if self._scaler is None and self._use_amp:
            self._scaler = GradScaler()
    
    def get_config(self) -> TrainingConfig:
        """Return immutable training configuration."""
        return self._config
    
    def should_accumulate(self, step: int) -> bool:
        """Check if gradients should be accumulated (no optimizer step)."""
        return self._should_accumulate_impl(
            step,
            self._config.gradient_accumulation_steps
        )
    
    def training_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        step: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Execute one training step with low-VRAM optimizations.
        
        Args:
            model: The model to train (must be in train mode)
            batch: Input batch with tensors on correct device
            step: Current global step number
            
        Returns:
            Tuple of (scaled loss tensor for backward, metrics dict)
            
        Note:
            Caller is responsible for:
            - Calling optimizer.step() when should_accumulate returns False
            - Calling optimizer.zero_grad() after optimizer.step()
        """
        self._initialize_scaler()
        
        model.train()
        metrics: Dict[str, float] = {}
        
        # Move batch to device (no-op if already there)
        batch = self._move_batch_to_device(batch)
        
        # Mixed precision forward pass
        if self._use_amp:
            with autocast():
                loss, model_metrics = self._forward_pass(model, batch)
        else:
            loss, model_metrics = self._forward_pass(model, batch)
        
        metrics.update(model_metrics)
        
        # Scale loss for gradient accumulation
        scaled_loss = self._scale_loss(
            loss,
            self._config.gradient_accumulation_steps
        )
        
        # Backward pass with mixed precision
        if self._use_amp and self._scaler is not None:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track metrics
        self._accumulated_loss += loss.detach().item()
        self._step_count += 1
        
        metrics["loss"] = loss.detach().item()
        metrics["scaled_loss"] = scaled_loss.detach().item()
        
        # Add memory metrics if manager available
        if self._memory_manager is not None:
            snapshot = self._memory_manager.get_snapshot()
            metrics["gpu_memory_mb"] = snapshot.gpu_allocated_mb
        
        return scaled_loss, metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Move batch tensors to target device."""
        return {
            key: value.to(self._device, non_blocking=True)
            if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
    
    def _forward_pass(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Execute forward pass and compute loss.
        
        Override this method for custom forward logic.
        Template Method Pattern: Default implementation, customizable.
        """
        # Extract inputs from batch (customize based on model interface)
        outputs = model(**batch)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            loss = outputs[0]
            metrics = {"action_loss": loss.item()} if hasattr(loss, 'item') else {}
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
            metrics = {}
            if hasattr(outputs, "action_loss") and outputs.action_loss is not None:
                metrics["action_loss"] = outputs.action_loss.item()
        else:
            loss = outputs
            metrics = {}
        
        return loss, metrics
    
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """
        Execute optimizer step with mixed precision support.
        
        Args:
            optimizer: The optimizer
            scheduler: Optional learning rate scheduler
            max_grad_norm: Max gradient norm for clipping (None to disable)
            
        Returns:
            Metrics dict with gradient info
        """
        metrics: Dict[str, float] = {}
        
        # Gradient clipping
        if max_grad_norm is not None:
            if self._use_amp and self._scaler is not None:
                self._scaler.unscale_(optimizer)
            
            total_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.grad is not None, 
                       filter(lambda p: p.requires_grad, 
                              optimizer.param_groups[0]['params'])),
                max_grad_norm
            )
            metrics["grad_norm"] = total_norm.item() if hasattr(total_norm, 'item') else total_norm
        
        # Optimizer step
        if self._use_amp and self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            metrics["learning_rate"] = scheduler.get_last_lr()[0]
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        # Calculate average loss over accumulation steps
        if self._step_count > 0:
            metrics["avg_loss"] = self._accumulated_loss / self._step_count
            self._accumulated_loss = 0.0
            self._step_count = 0
        
        return metrics
    
    def __repr__(self) -> str:
        return (
            f"LowVRAMTrainingStrategy("
            f"batch_size={self._config.batch_size}, "
            f"accum_steps={self._config.gradient_accumulation_steps}, "
            f"amp={self._use_amp})"
        )


__all__ = ["LowVRAMTrainingStrategy", "GradientAccumulationMixin"]
