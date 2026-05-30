"""
Low-VRAM Training Framework for GO-1
=====================================

Clean Architecture Package Structure:

    low_vram/
    ├── core/                    # Domain Layer (Pure business logic)
    │   ├── interfaces/          # Abstract interfaces (Dependency Inversion)
    │   ├── entities/            # Domain entities
    │   └── use_cases/           # Application business rules
    ├── adapters/                # Interface Adapters Layer
    │   ├── memory/              # Memory management implementations
    │   └── training/            # Training strategy implementations
    └── infrastructure/          # Frameworks & Drivers Layer
        ├── pytorch/             # PyTorch-specific implementations
        └── config/              # Configuration handling

Design Principles Applied:
    - SOLID Principles (especially Dependency Inversion)
    - Clean Architecture (layered separation)
    - Single Responsibility (one class = one job)
    - Interface Segregation (small, focused interfaces)
    - DRY (shared utilities extracted)

Security Considerations (CIA Triad):
    - Integrity: Checksum validation for cached features
    - Availability: Graceful degradation when memory is low
    - Confidentiality: No sensitive data logged
"""

__version__ = "0.1.0"
__author__ = "AgiBot-World Contributors"

from go1.tools.low_vram.core.interfaces import (
    MemoryManager,
    TrainingStrategy,
    FeatureCache,
)

__all__ = [
    "MemoryManager",
    "TrainingStrategy", 
    "FeatureCache",
]
