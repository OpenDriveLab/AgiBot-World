"""
Model Freezer Implementation
=============================

Handles selective freezing/unfreezing of model components.
Single Responsibility: Parameter freeze state management only.
"""

import logging
from typing import Dict, List, Optional, Set

import torch
from torch import nn

from go1.tools.low_vram.core.interfaces import ModelFreezer

logger = logging.getLogger(__name__)


class ComponentFreezer(ModelFreezer):
    """
    Smart model component freezer with named component support.
    
    Single Responsibility: Only handles freeze/unfreeze operations.
    Open/Closed: Add new component patterns without modifying core logic.
    """
    
    # Default component patterns for GO-1 model
    GO1_COMPONENTS = {
        "vision": ["vision_model"],
        "language": ["language_model"],
        "action_expert": ["action_model"],
        "latent_planner": ["latent_planner"],
        "adapters": ["mlp1", "k_proj", "v_proj", "state_adaptor", "action_adaptor"],
        "embedders": ["time_embedder", "freq_embedder"],
        "final": ["final_layer"],
    }
    
    def __init__(self, component_patterns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize freezer with component patterns.
        
        Args:
            component_patterns: Mapping of component names to module name patterns.
                              Defaults to GO1_COMPONENTS.
        """
        self._patterns = component_patterns or self.GO1_COMPONENTS.copy()
        self._frozen_components: Set[str] = set()
    
    def _get_modules_for_component(
        self,
        model: nn.Module,
        component_name: str
    ) -> List[nn.Module]:
        """Get all modules matching a component name."""
        if component_name not in self._patterns:
            # Try direct attribute access
            if hasattr(model, component_name):
                return [getattr(model, component_name)]
            raise ValueError(f"Unknown component: {component_name}")
        
        modules = []
        for pattern in self._patterns[component_name]:
            if hasattr(model, pattern):
                modules.append(getattr(model, pattern))
        
        return modules
    
    def freeze_component(self, model: nn.Module, component_name: str) -> int:
        """
        Freeze a model component by name.
        
        Args:
            model: The model containing the component
            component_name: Name of the component to freeze
            
        Returns:
            Number of parameters frozen
        """
        modules = self._get_modules_for_component(model, component_name)
        
        frozen_count = 0
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += param.numel()
        
        if frozen_count > 0:
            self._frozen_components.add(component_name)
            logger.info(
                f"Frozen '{component_name}': {frozen_count:,} parameters "
                f"({frozen_count * 4 / 1e6:.1f}MB saved)"
            )
        
        return frozen_count
    
    def unfreeze_component(self, model: nn.Module, component_name: str) -> int:
        """
        Unfreeze a model component by name.
        
        Args:
            model: The model containing the component
            component_name: Name of the component to unfreeze
            
        Returns:
            Number of parameters unfrozen
        """
        modules = self._get_modules_for_component(model, component_name)
        
        unfrozen_count = 0
        for module in modules:
            for param in module.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += param.numel()
        
        if unfrozen_count > 0:
            self._frozen_components.discard(component_name)
            logger.info(f"Unfrozen '{component_name}': {unfrozen_count:,} parameters")
        
        return unfrozen_count
    
    def get_frozen_params(self, model: nn.Module) -> int:
        """Get total number of frozen parameters."""
        return sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
    
    def get_trainable_params(self, model: nn.Module) -> int:
        """Get total number of trainable parameters."""
        return sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    
    def freeze_for_low_vram(self, model: nn.Module) -> Dict[str, int]:
        """
        Apply recommended freeze configuration for 4GB GPU.
        
        Freezes vision and language models, keeps action expert trainable.
        
        Returns:
            Dict of component names to frozen param counts
        """
        results = {}
        
        # Freeze heavy components
        for component in ["vision", "language"]:
            try:
                results[component] = self.freeze_component(model, component)
            except ValueError:
                logger.warning(f"Component '{component}' not found in model")
        
        # Log summary
        total_frozen = sum(results.values())
        trainable = self.get_trainable_params(model)
        
        logger.info(
            f"Low-VRAM freeze complete: "
            f"{total_frozen:,} frozen, {trainable:,} trainable "
            f"(~{trainable * 4 / 1e6:.1f}MB for gradients)"
        )
        
        return results
    
    def get_frozen_components(self) -> Set[str]:
        """Get set of currently frozen component names."""
        return self._frozen_components.copy()
    
    def __repr__(self) -> str:
        return (
            f"ComponentFreezer(frozen={list(self._frozen_components)}, "
            f"patterns={list(self._patterns.keys())})"
        )


__all__ = ["ComponentFreezer"]
