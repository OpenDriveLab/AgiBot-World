"""
Feature Cache Implementation
=============================

Disk-based cache for pre-computed vision features.
Security (CIA):
    - Integrity: SHA256 checksum validation
    - Availability: Graceful handling of corrupted files
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from go1.tools.low_vram.core.interfaces import FeatureCache

logger = logging.getLogger(__name__)


def compute_checksum(tensor: Tensor) -> str:
    """Compute SHA256 checksum of tensor data for integrity verification."""
    # Convert to bytes and hash
    data = tensor.detach().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]  # First 16 chars


class DiskFeatureCache(FeatureCache):
    """
    Disk-based feature cache with integrity verification.
    
    Single Responsibility: Feature storage/retrieval only.
    
    File Structure:
        cache_dir/
        ├── features/
        │   ├── {key}.pt          # Tensor files
        │   └── ...
        ├── metadata.json         # Cache metadata
        └── checksums.json        # Integrity checksums
    """
    
    def __init__(self, cache_dir: str, verify_on_load: bool = True):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache storage
            verify_on_load: Whether to verify checksums on retrieval
        """
        self._cache_dir = Path(cache_dir)
        self._features_dir = self._cache_dir / "features"
        self._verify_on_load = verify_on_load
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        # Create directories
        self._features_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create checksums
        self._checksums_path = self._cache_dir / "checksums.json"
        self._checksums: Dict[str, str] = self._load_checksums()
        
        logger.info(f"Feature cache initialized at {cache_dir}")
    
    def _load_checksums(self) -> Dict[str, str]:
        """Load checksums from disk."""
        if self._checksums_path.exists():
            try:
                with open(self._checksums_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Corrupted checksums file, starting fresh")
                return {}
        return {}
    
    def _save_checksums(self) -> None:
        """Save checksums to disk."""
        with open(self._checksums_path, "w") as f:
            json.dump(self._checksums, f, indent=2)
    
    def _get_path(self, key: str) -> Path:
        """Get file path for a key (sanitize key for filesystem)."""
        # Sanitize key to be filesystem-safe
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._features_dir / f"{safe_key}.pt"
    
    def store(
        self,
        key: str,
        features: Tensor,
        checksum: Optional[str] = None
    ) -> None:
        """
        Store features to disk with optional integrity checksum.
        
        Args:
            key: Unique identifier
            features: Tensor to store
            checksum: Optional pre-computed checksum (computes if None)
        """
        path = self._get_path(key)
        
        # Compute checksum if not provided
        if checksum is None:
            checksum = compute_checksum(features)
        
        # Save tensor
        torch.save(features.detach().cpu(), path)
        
        # Save checksum
        self._checksums[key] = checksum
        self._save_checksums()
        
        logger.debug(f"Cached '{key}': {features.shape}, checksum={checksum}")
    
    def retrieve(
        self,
        key: str,
        verify_checksum: bool = True
    ) -> Optional[Tensor]:
        """
        Retrieve features from cache.
        
        Args:
            key: Identifier used during storage
            verify_checksum: Whether to verify integrity
            
        Returns:
            Cached tensor or None if not found/corrupted
        """
        path = self._get_path(key)
        
        if not path.exists():
            self._misses += 1
            return None
        
        try:
            features = torch.load(path, weights_only=True)
        except Exception as e:
            logger.error(f"Failed to load cached features '{key}': {e}")
            self._misses += 1
            return None
        
        # Verify integrity
        if verify_checksum and self._verify_on_load:
            expected = self._checksums.get(key)
            if expected is not None:
                actual = compute_checksum(features)
                if actual != expected:
                    logger.error(
                        f"Checksum mismatch for '{key}': "
                        f"expected {expected}, got {actual}. "
                        "Data may be corrupted!"
                    )
                    self._misses += 1
                    return None
        
        self._hits += 1
        return features
    
    def exists(self, key: str) -> bool:
        """Check if features exist in cache without loading."""
        return self._get_path(key).exists()
    
    def clear(self) -> None:
        """Clear all cached features."""
        if self._features_dir.exists():
            shutil.rmtree(self._features_dir)
            self._features_dir.mkdir(parents=True, exist_ok=True)
        
        self._checksums.clear()
        self._save_checksums()
        
        self._hits = 0
        self._misses = 0
        
        logger.info("Feature cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            f.stat().st_size for f in self._features_dir.glob("*.pt")
        ) if self._features_dir.exists() else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "cached_items": len(list(self._features_dir.glob("*.pt"))),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self._cache_dir),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DiskFeatureCache("
            f"items={stats['cached_items']}, "
            f"size={stats['total_size_mb']:.1f}MB, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )


__all__ = ["DiskFeatureCache", "compute_checksum"]
