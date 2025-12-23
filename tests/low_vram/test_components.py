"""
Unit Tests for Low-VRAM Training Framework
==========================================

Testing following AAA pattern: Arrange, Act, Assert
Each test has a single responsibility.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn, Tensor

from go1.tools.low_vram.core.interfaces import (
    MemoryConfig,
    MemorySnapshot,
    TrainingConfig,
)
from go1.tools.low_vram.adapters.memory_manager import TorchMemoryManager
from go1.tools.low_vram.adapters.training_strategy import (
    LowVRAMTrainingStrategy,
    GradientAccumulationMixin,
)
from go1.tools.low_vram.adapters.feature_cache import DiskFeatureCache, compute_checksum
from go1.tools.low_vram.adapters.model_freezer import ComponentFreezer


class TestMemoryConfig(unittest.TestCase):
    """Tests for MemoryConfig dataclass."""
    
    def test_config_is_frozen(self):
        """Verify config cannot be modified after creation (Integrity)."""
        config = MemoryConfig(max_gpu_memory_mb=4000, max_cpu_memory_mb=8000)
        
        with self.assertRaises(Exception):  # FrozenInstanceError
            config.max_gpu_memory_mb = 5000
    
    def test_config_defaults(self):
        """Verify default values are sensible."""
        config = MemoryConfig(max_gpu_memory_mb=4000, max_cpu_memory_mb=8000)
        
        self.assertTrue(config.enable_cpu_offload)
        self.assertTrue(config.gradient_checkpointing)
        self.assertTrue(config.mixed_precision)


class TestTorchMemoryManager(unittest.TestCase):
    """Tests for TorchMemoryManager."""
    
    def setUp(self):
        self.config = MemoryConfig(
            max_gpu_memory_mb=4000,
            max_cpu_memory_mb=8000,
            pin_cpu_memory=False,  # Disable for tests without CUDA
        )
        self.manager = TorchMemoryManager(self.config)
    
    def test_get_snapshot_returns_valid_data(self):
        """Snapshot should return non-negative values."""
        snapshot = self.manager.get_snapshot()
        
        self.assertIsInstance(snapshot, MemorySnapshot)
        self.assertGreaterEqual(snapshot.gpu_allocated_mb, 0)
        self.assertGreaterEqual(snapshot.timestamp, 0)
    
    def test_offload_and_restore_cpu(self):
        """Tensor should roundtrip through CPU offload."""
        # Arrange
        tensor = torch.randn(100, 100)
        original_sum = tensor.sum().item()
        
        # Act
        cpu_tensor = self.manager.offload_to_cpu(tensor, "test_tensor")
        restored = self.manager.restore_to_gpu("test_tensor", torch.device("cpu"))
        
        # Assert
        self.assertAlmostEqual(restored.sum().item(), original_sum, places=5)
        self.assertEqual(self.manager.get_offloaded_count(), 0)
    
    def test_offload_duplicate_name_raises(self):
        """Offloading with same name should raise (Integrity protection)."""
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(10, 10)
        
        self.manager.offload_to_cpu(tensor1, "same_name")
        
        with self.assertRaises(ValueError):
            self.manager.offload_to_cpu(tensor2, "same_name")
    
    def test_restore_nonexistent_raises(self):
        """Restoring nonexistent tensor should raise."""
        with self.assertRaises(KeyError):
            self.manager.restore_to_gpu("nonexistent", torch.device("cpu"))


class TestGradientAccumulation(unittest.TestCase):
    """Tests for gradient accumulation logic."""
    
    def test_should_accumulate_for_intermediate_steps(self):
        """Accumulation steps 1-15 should accumulate for accum_steps=16."""
        mixin = GradientAccumulationMixin()
        
        # Steps 0-14 should accumulate
        for step in range(15):
            self.assertTrue(
                mixin._should_accumulate_impl(step, 16),
                f"Step {step} should accumulate"
            )
    
    def test_should_not_accumulate_on_final_step(self):
        """Step 15 (16th step) should not accumulate."""
        mixin = GradientAccumulationMixin()
        
        self.assertFalse(mixin._should_accumulate_impl(15, 16))
        self.assertFalse(mixin._should_accumulate_impl(31, 16))
    
    def test_loss_scaling(self):
        """Loss should be scaled by accumulation steps."""
        mixin = GradientAccumulationMixin()
        loss = torch.tensor(1.6)
        
        scaled = mixin._scale_loss(loss, 16)
        
        self.assertAlmostEqual(scaled.item(), 0.1, places=5)


class TestDiskFeatureCache(unittest.TestCase):
    """Tests for DiskFeatureCache."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DiskFeatureCache(self.temp_dir)
    
    def tearDown(self):
        self.cache.clear()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve(self):
        """Basic store/retrieve should work."""
        features = torch.randn(64, 256)
        
        self.cache.store("test_key", features)
        retrieved = self.cache.retrieve("test_key")
        
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.allclose(features, retrieved))
    
    def test_exists_check(self):
        """Exists should return correct boolean."""
        self.assertFalse(self.cache.exists("nonexistent"))
        
        self.cache.store("exists_key", torch.randn(10))
        
        self.assertTrue(self.cache.exists("exists_key"))
    
    def test_checksum_validation(self):
        """Checksum should be computed and stored."""
        features = torch.randn(10, 10)
        expected_checksum = compute_checksum(features)
        
        self.cache.store("checksum_key", features)
        
        # Verify checksum was stored
        self.assertIn("checksum_key", self.cache._checksums)
        self.assertEqual(self.cache._checksums["checksum_key"], expected_checksum)
    
    def test_cache_stats(self):
        """Stats should track hits and misses."""
        self.cache.store("hit_key", torch.randn(10))
        
        self.cache.retrieve("hit_key")  # Hit
        self.cache.retrieve("miss_key")  # Miss
        
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)


class TestComponentFreezer(unittest.TestCase):
    """Tests for ComponentFreezer."""
    
    def setUp(self):
        # Create a simple model with named components
        self.model = nn.Module()
        self.model.vision = nn.Linear(10, 10)
        self.model.language = nn.Linear(10, 10)
        self.model.action = nn.Linear(10, 10)
        
        self.freezer = ComponentFreezer({
            "vision": ["vision"],
            "language": ["language"],
            "action": ["action"],
        })
    
    def test_freeze_component(self):
        """Freezing should set requires_grad=False."""
        # Verify initially trainable
        self.assertTrue(self.model.vision.weight.requires_grad)
        
        # Freeze
        frozen_count = self.freezer.freeze_component(self.model, "vision")
        
        # Verify frozen
        self.assertFalse(self.model.vision.weight.requires_grad)
        self.assertGreater(frozen_count, 0)
    
    def test_unfreeze_component(self):
        """Unfreezing should set requires_grad=True."""
        # Freeze first
        self.freezer.freeze_component(self.model, "vision")
        self.assertFalse(self.model.vision.weight.requires_grad)
        
        # Unfreeze
        self.freezer.unfreeze_component(self.model, "vision")
        
        # Verify unfrozen
        self.assertTrue(self.model.vision.weight.requires_grad)
    
    def test_get_trainable_params(self):
        """Should count trainable parameters correctly."""
        total = self.freezer.get_trainable_params(self.model)
        
        self.freezer.freeze_component(self.model, "vision")
        
        after_freeze = self.freezer.get_trainable_params(self.model)
        
        self.assertLess(after_freeze, total)


class TestLowVRAMTrainingStrategy(unittest.TestCase):
    """Tests for LowVRAMTrainingStrategy."""
    
    def setUp(self):
        self.config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
        )
        self.strategy = LowVRAMTrainingStrategy(
            config=self.config,
            device=torch.device("cpu"),
        )
    
    def test_get_config_returns_immutable(self):
        """Config should be accessible."""
        config = self.strategy.get_config()
        
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 4)
    
    def test_should_accumulate_logic(self):
        """Should correctly determine accumulation."""
        # Steps 0, 1, 2 should accumulate
        self.assertTrue(self.strategy.should_accumulate(0))
        self.assertTrue(self.strategy.should_accumulate(1))
        self.assertTrue(self.strategy.should_accumulate(2))
        
        # Step 3 should not (4th step, time to update)
        self.assertFalse(self.strategy.should_accumulate(3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
