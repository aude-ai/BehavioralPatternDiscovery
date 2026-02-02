# Core infrastructure for BehavioralPatternDiscovery
#
# Note: Uses lazy imports for torch-dependent modules (interfaces, numerical)
# to support environments without torch (e.g., Hetzner cloud workers).

# These don't require torch
from .registry import ComponentRegistry
from .config import load_config, load_all_configs, validate_required_keys

__all__ = [
    # Config/Registry (no torch)
    "ComponentRegistry",
    "load_config",
    "load_all_configs",
    "validate_required_keys",
    # Interfaces (lazy - requires torch)
    "BaseEncoder",
    "BaseUnification",
    "BaseDecoder",
    "LatentDistribution",
    "BaseDiscriminator",
    "BaseLoss",
    # Numerical (lazy - requires torch)
    "NumericalConstants",
    "safe_div",
    "safe_log",
    "safe_exp",
    "safe_sqrt",
    "safe_std",
    "safe_var",
    "check_finite",
]


def __getattr__(name):
    """Lazy import for classes that require torch."""

    # Interfaces (require torch)
    if name in {
        "BaseEncoder",
        "BaseUnification",
        "BaseDecoder",
        "LatentDistribution",
        "BaseDiscriminator",
        "BaseLoss",
    }:
        from .interfaces import (
            BaseEncoder,
            BaseUnification,
            BaseDecoder,
            LatentDistribution,
            BaseDiscriminator,
            BaseLoss,
        )
        return locals()[name]

    # Numerical utilities (require torch)
    if name in {
        "NumericalConstants",
        "safe_div",
        "safe_log",
        "safe_exp",
        "safe_sqrt",
        "safe_std",
        "safe_var",
        "check_finite",
    }:
        from .numerical import (
            NumericalConstants,
            safe_div,
            safe_log,
            safe_exp,
            safe_sqrt,
            safe_std,
            safe_var,
            check_finite,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
