"""
Optimizer Factory

Registry-based optimizer creation with config-driven instantiation.
"""

import logging
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.core.registry import ComponentRegistry
from .lion import Lion
from .ademamix import AdEMAMix

logger = logging.getLogger(__name__)

# Registry for optimizers
optimizer_registry = ComponentRegistry[Optimizer]("optimizer")

# Register built-in PyTorch optimizers
optimizer_registry.register_class("adam", torch.optim.Adam)
optimizer_registry.register_class("adamw", torch.optim.AdamW)
optimizer_registry.register_class("sgd", torch.optim.SGD)
optimizer_registry.register_class("rmsprop", torch.optim.RMSprop)

# Register custom optimizers
optimizer_registry.register_class("lion", Lion)
optimizer_registry.register_class("ademamix", AdEMAMix)

# Register NAdam and RAdam if available (PyTorch 1.10+)
if hasattr(torch.optim, "NAdam"):
    optimizer_registry.register_class("nadam", torch.optim.NAdam)
if hasattr(torch.optim, "RAdam"):
    optimizer_registry.register_class("radam", torch.optim.RAdam)


def create_optimizer(
    params: Iterator[nn.Parameter],
    config: dict[str, Any],
) -> Optimizer:
    """
    Create optimizer from configuration.

    Args:
        params: Model parameters to optimize
        config: Optimizer configuration containing:
            - type: Optimizer type (adam, adamw, lion, ademamix, etc.)
            - learning_rate: Learning rate
            - weight_decay: Weight decay
            - {type}: Type-specific configuration

    Returns:
        Instantiated optimizer

    Raises:
        KeyError: If required config keys are missing
        ValueError: If optimizer type is unknown
    """
    optimizer_type = config["type"]
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]

    # Get type-specific config (must exist in config)
    type_config = config[optimizer_type]

    # Build kwargs
    kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
    }

    # Add type-specific parameters
    if optimizer_type in ["adam", "adamw"]:
        kwargs["betas"] = tuple(type_config["betas"])
        kwargs["eps"] = type_config["eps"]
        kwargs["amsgrad"] = type_config["amsgrad"]

    elif optimizer_type == "sgd":
        kwargs["momentum"] = type_config["momentum"]
        kwargs["dampening"] = type_config["dampening"]
        kwargs["nesterov"] = type_config["nesterov"]

    elif optimizer_type == "lion":
        kwargs["betas"] = tuple(type_config["betas"])

    elif optimizer_type == "ademamix":
        kwargs["betas"] = tuple(type_config["betas"])
        kwargs["alpha"] = type_config["alpha"]
        kwargs["eps"] = type_config["eps"]
        kwargs["T_alpha"] = type_config["T_alpha"]
        kwargs["T_beta3"] = type_config["T_beta3"]

    elif optimizer_type == "rmsprop":
        kwargs["alpha"] = type_config["alpha"]
        kwargs["eps"] = type_config["eps"]
        kwargs["momentum"] = type_config["momentum"]

    elif optimizer_type in ["nadam", "radam"]:
        kwargs["betas"] = tuple(type_config["betas"])
        kwargs["eps"] = type_config["eps"]

    # Get optimizer class and instantiate
    optimizer_cls = optimizer_registry.get(optimizer_type)
    return optimizer_cls(params, **kwargs)


def get_supported_optimizers() -> list[str]:
    """Return list of supported optimizer types."""
    return optimizer_registry.list_registered()
