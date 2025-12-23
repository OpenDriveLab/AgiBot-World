"""
Memory Manager Implementation
==============================

Concrete implementation of MemoryManager interface.
Single Responsibility: GPU/CPU memory management only.

Security (CIA):
    - Availability: Prevents OOM by proactive monitoring
    - Integrity: Validates tensor state after transfers
"""

import gc
import logging
import time
from typing import Dict, Optional

import torch
from torch import Tensor

from go1.tools.low_vram.core.interfaces import (
    MemoryConfig,
    MemoryManager,
    MemorySnapshot,
    MemoryTier,
)

logger = logging.getLogger(__name__)


class TorchMemoryManager(MemoryManager):
    """
    PyTorch-based memory manager with CPU offloading support.
    
    Follows Single Responsibility: Only manages memory, no training logic.
    Follows Open/Closed: Extend by subclassing, don't modify.
    
    Attributes:
        config: Immutable memory configuration
        _offloaded: Dict of tensors offloaded to CPU
        _pinned: Whether to use pinned memory for CPU tensors
    """
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize memory manager.
        
        Args:
            config: Frozen MemoryConfig instance
        """
        self._config = config
        self._offloaded: Dict[str, Tensor] = {}
        self._offload_metadata: Dict[str, Dict] = {}
        
        # Set memory limits if GPU available
        if torch.cuda.is_available() and config.max_gpu_memory_mb > 0:
            # Reserve some memory for system operations
            fraction = config.max_gpu_memory_mb / (
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            )
            fraction = min(0.95, fraction)  # Never use more than 95%
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"GPU memory fraction set to {fraction:.2%}")
    
    @property
    def config(self) -> MemoryConfig:
        """Return immutable config (no setter - prevents tampering)."""
        return self._config
    
    def get_snapshot(self) -> MemorySnapshot:
        """
        Get current memory state without side effects.
        
        This is a pure query - no state modification.
        """
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_max = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        else:
            gpu_allocated = gpu_reserved = gpu_max = 0.0
        
        # Estimate CPU memory used by offloaded tensors
        cpu_used = sum(
            t.numel() * t.element_size() / (1024 * 1024)
            for t in self._offloaded.values()
        )
        
        return MemorySnapshot(
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_max_mb=gpu_max,
            cpu_used_mb=cpu_used,
            timestamp=time.time(),
        )
    
    def can_allocate(self, size_bytes: int) -> bool:
        """
        Check if allocation is possible without actually allocating.
        
        Pure function - no side effects.
        """
        if not torch.cuda.is_available():
            return True  # CPU allocation always possible (simplified)
        
        snapshot = self.get_snapshot()
        size_mb = size_bytes / (1024 * 1024)
        available = self._config.max_gpu_memory_mb - snapshot.gpu_allocated_mb
        
        # Add safety margin
        safety_margin = 50  # MB
        return size_mb < (available - safety_margin)
    
    def offload_to_cpu(self, tensor: Tensor, name: str) -> Tensor:
        """
        Offload tensor to CPU memory.
        
        Args:
            tensor: GPU tensor to offload
            name: Unique identifier
            
        Returns:
            CPU tensor (pinned if configured)
            
        Raises:
            ValueError: If name already exists (prevents silent overwrite)
        """
        if name in self._offloaded:
            raise ValueError(
                f"Tensor '{name}' already offloaded. "
                "Use unique names or restore first. (Integrity protection)"
            )
        
        # Store metadata for validation
        self._offload_metadata[name] = {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": str(tensor.device),
            "numel": tensor.numel(),
        }
        
        # Transfer to CPU
        if self._config.pin_cpu_memory and torch.cuda.is_available():
            cpu_tensor = tensor.detach().cpu().pin_memory()
        else:
            cpu_tensor = tensor.detach().cpu()
        
        self._offloaded[name] = cpu_tensor
        
        logger.debug(
            f"Offloaded '{name}': {tensor.shape} "
            f"({tensor.numel() * tensor.element_size() / 1024:.1f} KB)"
        )
        
        return cpu_tensor
    
    def restore_to_gpu(self, name: str, device: torch.device) -> Tensor:
        """
        Restore tensor from CPU to GPU.
        
        Args:
            name: Identifier used during offload
            device: Target GPU device
            
        Returns:
            GPU tensor
            
        Raises:
            KeyError: If name not found
            RuntimeError: If tensor integrity check fails
        """
        if name not in self._offloaded:
            raise KeyError(f"No offloaded tensor found with name '{name}'")
        
        cpu_tensor = self._offloaded.pop(name)
        metadata = self._offload_metadata.pop(name)
        
        # Integrity check
        if cpu_tensor.numel() != metadata["numel"]:
            raise RuntimeError(
                f"Tensor integrity check failed for '{name}'. "
                f"Expected {metadata['numel']} elements, got {cpu_tensor.numel()}"
            )
        
        # Transfer to GPU
        gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        
        logger.debug(f"Restored '{name}' to {device}")
        
        return gpu_tensor
    
    def clear_cache(self) -> None:
        """Clear GPU cache and trigger garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("Cache cleared")
    
    def get_offloaded_count(self) -> int:
        """Get number of currently offloaded tensors."""
        return len(self._offloaded)
    
    def get_offloaded_memory_mb(self) -> float:
        """Get total memory used by offloaded tensors in MB."""
        return sum(
            t.numel() * t.element_size() / (1024 * 1024)
            for t in self._offloaded.values()
        )
    
    def __repr__(self) -> str:
        snapshot = self.get_snapshot()
        return (
            f"TorchMemoryManager("
            f"gpu={snapshot.gpu_allocated_mb:.1f}/{snapshot.gpu_max_mb:.1f}MB, "
            f"offloaded={self.get_offloaded_count()} tensors)"
        )


class MemoryGuard:
    """
    Context manager for memory-safe operations.
    
    Usage:
        with MemoryGuard(manager, min_free_mb=500):
            # Operations that need memory
            pass
    """
    
    def __init__(
        self,
        manager: MemoryManager,
        min_free_mb: float = 100,
        auto_clear: bool = True
    ):
        self._manager = manager
        self._min_free_mb = min_free_mb
        self._auto_clear = auto_clear
        self._initial_snapshot: Optional[MemorySnapshot] = None
    
    def __enter__(self) -> "MemoryGuard":
        self._initial_snapshot = self._manager.get_snapshot()
        
        # Pre-emptively clear cache if low on memory
        available = (
            self._initial_snapshot.gpu_max_mb - 
            self._initial_snapshot.gpu_allocated_mb
        )
        if available < self._min_free_mb:
            self._manager.clear_cache()
            logger.warning(f"Pre-emptive cache clear: {available:.1f}MB available")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._auto_clear:
            self._manager.clear_cache()
        
        # Log memory change
        final_snapshot = self._manager.get_snapshot()
        delta = (
            final_snapshot.gpu_allocated_mb - 
            self._initial_snapshot.gpu_allocated_mb
        )
        if abs(delta) > 10:  # Only log significant changes
            logger.debug(f"Memory delta: {delta:+.1f}MB")
        
        return False  # Don't suppress exceptions


__all__ = ["TorchMemoryManager", "MemoryGuard"]
