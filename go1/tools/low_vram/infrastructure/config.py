"""
Configuration Loading/Saving
=============================

YAML-based configuration for low-VRAM training.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from go1.tools.low_vram.core.interfaces import MemoryConfig, TrainingConfig

logger = logging.getLogger(__name__)


def load_config(
    config_path: Union[str, Path],
) -> tuple[MemoryConfig, TrainingConfig]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (MemoryConfig, TrainingConfig)
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = json.load(f)
    
    memory_config = MemoryConfig(**data.get("memory", {}))
    training_config = TrainingConfig(**data.get("training", {}))
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return memory_config, training_config


def save_config(
    memory_config: MemoryConfig,
    training_config: TrainingConfig,
    config_path: Union[str, Path],
) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        memory_config: Memory configuration
        training_config: Training configuration
        config_path: Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "memory": asdict(memory_config),
        "training": asdict(training_config),
    }
    
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def create_default_config_file(output_path: Union[str, Path]) -> None:
    """Create a default configuration file for reference."""
    from go1.tools.low_vram.infrastructure.factory import create_4gb_config
    
    memory_config, training_config = create_4gb_config()
    save_config(memory_config, training_config, output_path)


__all__ = ["load_config", "save_config", "create_default_config_file"]
