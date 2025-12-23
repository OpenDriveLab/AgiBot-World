"""
Core Interfaces for Low-VRAM Training
======================================

Following Interface Segregation Principle (ISP):
Each interface is small and focused on a single responsibility.

Following Dependency Inversion Principle (DIP):
High-level modules depend on these abstractions, not concrete implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterator, Optional, Tuple, Any
import torch
from torch import Tensor


# =============================================================================
# ENUMS - Type-safe configuration options
# =============================================================================

class MemoryTier(Enum):
    """Memory tier for offloading decisions."""
    GPU_HIGH_PRIORITY = auto()  # Stay on GPU always
    GPU_LOW_PRIORITY = auto()   # On GPU, but can be offloaded
    CPU_PINNED = auto()          # On CPU with pinned memory
    CPU_STANDARD = auto()        # On CPU standard memory
    DISK = auto()                # Offloaded to disk (extreme low memory)


class TrainingPhase(Enum):
    """Current phase of training for strategy selection."""
    WARMUP = auto()
    TRAINING = auto()
    VALIDATION = auto()
    CHECKPOINTING = auto()


# =============================================================================
# DATA CLASSES - Immutable configuration and state (no side effects)
# =============================================================================

@dataclass(frozen=True)
class MemoryConfig:
    """
    Immutable configuration for memory management.
    
    Frozen dataclass ensures no tampering after creation (Integrity).
    """
    max_gpu_memory_mb: int
    max_cpu_memory_mb: int
    enable_cpu_offload: bool = True
    enable_disk_offload: bool = False
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    pin_cpu_memory: bool = True


@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable snapshot of current memory state."""
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_max_mb: float
    cpu_used_mb: float
    timestamp: float


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    max_epochs: int = 10
    warmup_steps: int = 100
    freeze_vision: bool = True
    freeze_llm: bool = True
    train_action_expert: bool = True


# =============================================================================
# ABSTRACT INTERFACES - Dependency Inversion Principle
# =============================================================================

class MemoryManager(ABC):
    """
    Interface for GPU/CPU memory management.
    
    Single Responsibility: Only manages memory allocation and offloading.
    Does NOT handle training logic or model operations.
    """
    
    @abstractmethod
    def get_snapshot(self) -> MemorySnapshot:
        """Get current memory state without side effects."""
        pass
    
    @abstractmethod
    def can_allocate(self, size_bytes: int) -> bool:
        """Check if allocation is possible without allocating."""
        pass
    
    @abstractmethod
    def offload_to_cpu(self, tensor: Tensor, name: str) -> Tensor:
        """
        Offload tensor to CPU, return CPU tensor.
        
        Args:
            tensor: GPU tensor to offload
            name: Identifier for later retrieval
            
        Returns:
            CPU tensor (pinned if configured)
        """
        pass
    
    @abstractmethod
    def restore_to_gpu(self, name: str, device: torch.device) -> Tensor:
        """
        Restore tensor from CPU to GPU.
        
        Args:
            name: Identifier used during offload
            device: Target GPU device
            
        Returns:
            GPU tensor
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """Clear GPU cache and garbage collect."""
        pass


class TrainingStrategy(ABC):
    """
    Interface for training strategies (Strategy Pattern).
    
    Single Responsibility: Only handles training step logic.
    Does NOT handle memory management directly.
    """
    
    @abstractmethod
    def training_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        step: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Execute one training step.
        
        Args:
            model: The model to train
            batch: Input batch dictionary
            step: Current global step number
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        pass
    
    @abstractmethod
    def should_accumulate(self, step: int) -> bool:
        """Check if gradients should be accumulated (no optimizer step)."""
        pass
    
    @abstractmethod
    def get_config(self) -> TrainingConfig:
        """Return current training configuration."""
        pass


class FeatureCache(ABC):
    """
    Interface for caching pre-computed features.
    
    Single Responsibility: Only handles feature storage and retrieval.
    """
    
    @abstractmethod
    def store(self, key: str, features: Tensor, checksum: Optional[str] = None) -> None:
        """
        Store features with optional integrity checksum.
        
        Args:
            key: Unique identifier for the features
            features: Tensor to store
            checksum: Optional hash for integrity verification
        """
        pass
    
    @abstractmethod
    def retrieve(self, key: str, verify_checksum: bool = True) -> Optional[Tensor]:
        """
        Retrieve features, optionally verifying integrity.
        
        Args:
            key: Identifier used during storage
            verify_checksum: Whether to verify data integrity
            
        Returns:
            Cached tensor or None if not found/corrupted
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if features exist in cache without loading."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached features."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (hits, misses, size)."""
        pass


class ModelFreezer(ABC):
    """
    Interface for model parameter freezing.
    
    Single Responsibility: Only handles freeze/unfreeze operations.
    """
    
    @abstractmethod
    def freeze_component(self, model: torch.nn.Module, component_name: str) -> int:
        """
        Freeze a model component by name.
        
        Args:
            model: The model containing the component
            component_name: Name of the component to freeze
            
        Returns:
            Number of parameters frozen
        """
        pass
    
    @abstractmethod
    def unfreeze_component(self, model: torch.nn.Module, component_name: str) -> int:
        """Unfreeze a model component by name."""
        pass
    
    @abstractmethod
    def get_frozen_params(self, model: torch.nn.Module) -> int:
        """Get total number of frozen parameters."""
        pass
    
    @abstractmethod
    def get_trainable_params(self, model: torch.nn.Module) -> int:
        """Get total number of trainable parameters."""
        pass


class ProgressReporter(ABC):
    """
    Interface for progress reporting (Observer Pattern).
    
    Single Responsibility: Only handles progress updates.
    Follows Open/Closed - extend by adding new reporters.
    """
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called when an epoch starts."""
        pass
    
    @abstractmethod
    def on_step(
        self,
        step: int,
        total_steps: int,
        loss: float,
        metrics: Dict[str, float],
    ) -> None:
        """Called after each training step."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called when an epoch ends."""
        pass
    
    @abstractmethod
    def on_checkpoint(self, path: str) -> None:
        """Called when a checkpoint is saved."""
        pass


# =============================================================================
# FACTORY INTERFACE - Abstract Factory Pattern
# =============================================================================

class TrainerFactory(ABC):
    """
    Abstract Factory for creating training components.
    
    Allows dependency injection of different implementations.
    """
    
    @abstractmethod
    def create_memory_manager(self, config: MemoryConfig) -> MemoryManager:
        """Create a memory manager instance."""
        pass
    
    @abstractmethod
    def create_training_strategy(self, config: TrainingConfig) -> TrainingStrategy:
        """Create a training strategy instance."""
        pass
    
    @abstractmethod
    def create_feature_cache(self, cache_dir: str) -> FeatureCache:
        """Create a feature cache instance."""
        pass
    
    @abstractmethod
    def create_model_freezer(self) -> ModelFreezer:
        """Create a model freezer instance."""
        pass


# Export all public interfaces
__all__ = [
    # Enums
    "MemoryTier",
    "TrainingPhase",
    # Data classes
    "MemoryConfig",
    "MemorySnapshot", 
    "TrainingConfig",
    # Interfaces
    "MemoryManager",
    "TrainingStrategy",
    "FeatureCache",
    "ModelFreezer",
    "ProgressReporter",
    "TrainerFactory",
]
