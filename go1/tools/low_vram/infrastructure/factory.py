"""
Trainer Factory Implementation
===============================

Factory for creating training components.
Follows Abstract Factory Pattern for dependency injection.
"""

import logging
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW

from go1.tools.low_vram.core.interfaces import (
    FeatureCache,
    MemoryConfig,
    MemoryManager,
    ModelFreezer,
    TrainerFactory,
    TrainingConfig,
    TrainingStrategy,
)
from go1.tools.low_vram.adapters.memory_manager import TorchMemoryManager
from go1.tools.low_vram.adapters.training_strategy import LowVRAMTrainingStrategy
from go1.tools.low_vram.adapters.feature_cache import DiskFeatureCache
from go1.tools.low_vram.adapters.model_freezer import ComponentFreezer

logger = logging.getLogger(__name__)


class DefaultTrainerFactory(TrainerFactory):
    """
    Default factory for creating low-VRAM training components.
    
    Implements Abstract Factory Pattern.
    Single Responsibility: Only creates components, doesn't configure them.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize factory.
        
        Args:
            device: Target device (defaults to cuda if available)
        """
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        # Cached instances for reuse
        self._memory_manager: Optional[MemoryManager] = None
    
    def create_memory_manager(self, config: MemoryConfig) -> MemoryManager:
        """Create and cache a memory manager instance."""
        if self._memory_manager is None:
            self._memory_manager = TorchMemoryManager(config)
        return self._memory_manager
    
    def create_training_strategy(
        self,
        config: TrainingConfig,
        memory_manager: Optional[MemoryManager] = None,
    ) -> TrainingStrategy:
        """Create a training strategy instance."""
        return LowVRAMTrainingStrategy(
            config=config,
            memory_manager=memory_manager,
            device=self._device,
        )
    
    def create_feature_cache(self, cache_dir: str) -> FeatureCache:
        """Create a feature cache instance."""
        return DiskFeatureCache(cache_dir)
    
    def create_model_freezer(self) -> ModelFreezer:
        """Create a model freezer instance."""
        return ComponentFreezer()
    
    def create_optimizer(
        self,
        model: nn.Module,
        config: TrainingConfig,
        use_8bit: bool = False,
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with optional 8-bit quantization.
        
        Args:
            model: Model to optimize
            config: Training configuration
            use_8bit: Whether to use 8-bit Adam (requires bitsandbytes)
            
        Returns:
            Optimizer instance
        """
        # Filter trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        
        if not params:
            raise ValueError("No trainable parameters found!")
        
        if use_8bit:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.Adam8bit(
                    params,
                    lr=config.learning_rate,
                    weight_decay=0.01,
                )
                logger.info("Using 8-bit Adam optimizer (saves ~50% optimizer memory)")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed, falling back to standard AdamW. "
                    "Install with: pip install bitsandbytes"
                )
                optimizer = AdamW(params, lr=config.learning_rate, weight_decay=0.01)
        else:
            optimizer = AdamW(params, lr=config.learning_rate, weight_decay=0.01)
        
        return optimizer
    
    def __repr__(self) -> str:
        return f"DefaultTrainerFactory(device={self._device})"


def create_4gb_config() -> tuple[MemoryConfig, TrainingConfig]:
    """
    Factory function for 4GB GPU configuration.
    
    Returns minimal memory footprint settings.
    """
    memory_config = MemoryConfig(
        max_gpu_memory_mb=3500,  # Leave 500MB for system
        max_cpu_memory_mb=8000,
        enable_cpu_offload=True,
        enable_disk_offload=False,
        gradient_checkpointing=True,
        mixed_precision=True,
        pin_cpu_memory=True,
    )
    
    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        max_epochs=10,
        warmup_steps=100,
        freeze_vision=True,
        freeze_llm=True,
        train_action_expert=True,
    )
    
    return memory_config, training_config


def create_8gb_config() -> tuple[MemoryConfig, TrainingConfig]:
    """Factory function for 8GB GPU configuration."""
    memory_config = MemoryConfig(
        max_gpu_memory_mb=7000,
        max_cpu_memory_mb=16000,
        enable_cpu_offload=True,
        enable_disk_offload=False,
        gradient_checkpointing=True,
        mixed_precision=True,
        pin_cpu_memory=True,
    )
    
    training_config = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_epochs=10,
        warmup_steps=100,
        freeze_vision=True,
        freeze_llm=False,  # Can train last few LLM layers
        train_action_expert=True,
    )
    
    return memory_config, training_config


__all__ = [
    "DefaultTrainerFactory",
    "create_4gb_config",
    "create_8gb_config",
]
