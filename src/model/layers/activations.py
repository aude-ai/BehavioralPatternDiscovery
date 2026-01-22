"""
Activation Functions

Provides all activation functions used in the model. All activations
are registered in activation_registry for configuration-driven selection.

Supported activations:
- relu: Standard ReLU
- leaky_relu: Leaky ReLU with configurable negative_slope
- swish: Swish/SiLU activation
- gelu: Gaussian Error Linear Unit
- elu: Exponential Linear Unit with configurable alpha
- xielu: Extended Integral ELU with learnable alpha
- cone: Cone activation (arXiv:2405.04459)
- parabolic_cone: Smooth parabolic cone activation
- parameterized_cone: Learnable cone with alpha, beta, gamma
- swiglu: Gated Linear Unit with Swish (changes dimensions)

SHAP Compatibility: All activations use standard PyTorch operations
that maintain gradient flow for SHAP analysis.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.numerical import safe_exp
from src.core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Registry for activation functions
activation_registry = ComponentRegistry[nn.Module]("activation")


@activation_registry.register("relu")
class ReLU(nn.ReLU):
    """Standard ReLU activation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()


@activation_registry.register("leaky_relu")
class LeakyReLU(nn.LeakyReLU):
    """Leaky ReLU with configurable negative slope."""

    def __init__(self, config: dict[str, Any]):
        negative_slope = config["negative_slope"]
        super().__init__(negative_slope=negative_slope)


@activation_registry.register("swish")
class Swish(nn.SiLU):
    """Swish/SiLU activation: x * sigmoid(x)"""

    def __init__(self, config: dict[str, Any]):
        super().__init__()


@activation_registry.register("gelu")
class GELU(nn.GELU):
    """Gaussian Error Linear Unit."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()


@activation_registry.register("elu")
class ELU(nn.ELU):
    """Exponential Linear Unit with configurable alpha."""

    def __init__(self, config: dict[str, Any]):
        alpha = config["alpha"]
        super().__init__(alpha=alpha)


@activation_registry.register("xielu")
class xIELU(nn.Module):
    """
    xIELU: Expanded Integral of ELU

    Formula: xIELU(x) = x if x >= 0 else alpha * (exp(x) - 1)

    Args:
        config: Must contain:
            - alpha: Initial alpha value
            - learnable: If True, alpha is a learnable parameter
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        alpha = config["alpha"]
        learnable = config["learnable"]

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.alpha * (safe_exp(x) - 1))

    def extra_repr(self) -> str:
        return f"alpha={self.alpha.item():.4f}"


@activation_registry.register("cone")
class Cone(nn.Module):
    """
    Cone activation function (arXiv:2405.04459)

    Formula: Cone(z) = 1 - |z - 1|
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - torch.abs(x - 1)


@activation_registry.register("parabolic_cone")
class ParabolicCone(nn.Module):
    """
    Parabolic-Cone activation function (arXiv:2405.04459)

    Formula: ParabolicCone(z) = z(2 - z)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (2 - x)


@activation_registry.register("parameterized_cone")
class ParameterizedCone(nn.Module):
    """
    Parameterized-Cone activation function (arXiv:2405.04459)

    Formula: ParameterizedCone(z) = beta - |z - gamma|^alpha

    Args:
        config: Must contain:
            - alpha: Power parameter
            - beta: Vertical offset
            - gamma: Horizontal center
            - learnable: If True, parameters are learnable
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        alpha = config["alpha"]
        beta = config["beta"]
        gamma = config["gamma"]
        learnable = config["learnable"]

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta", torch.tensor(beta))
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta - torch.abs(x - self.gamma) ** self.alpha

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha.item():.4f}, "
            f"beta={self.beta.item():.4f}, "
            f"gamma={self.gamma.item():.4f}"
        )


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Gated Linear Unit with Swish activation.

    Formula: SwiGLU(x) = Swish(xW) * (xV)

    Note: This activation changes dimensions and is NOT registered
    in the standard registry. Use get_swiglu() factory instead.

    This activation increases parameter count by ~75% compared to
    standard activations. Checkpoints are NOT compatible between
    SwiGLU and non-SwiGLU models.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.W(x)) * self.V(x)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


def make_activation(activation_type: str, config: dict[str, Any]) -> nn.Module:
    """
    Factory function to create an activation module instance.

    Args:
        activation_type: Activation type from activation_registry
        config: Configuration dictionary containing activation-specific parameters.
                For activations that need parameters, the config must contain
                a key matching the activation type with the parameters.

    Returns:
        Instantiated activation module

    Raises:
        KeyError: If activation_type is not registered

    Example:
        config = {
            "type": "leaky_relu",
            "leaky_relu": {"negative_slope": 0.01}
        }
        activation = make_activation("leaky_relu", config["leaky_relu"])
    """
    return activation_registry.create(activation_type, config=config)


def get_swiglu(input_dim: int, output_dim: int) -> SwiGLU:
    """
    Factory function for SwiGLU (which changes dimensions).

    SwiGLU is not in the standard registry because it requires
    input/output dimensions rather than a config dict.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension

    Returns:
        SwiGLU module instance
    """
    return SwiGLU(input_dim, output_dim)


def make_activation_factory(
    activation_type: str,
    config: dict[str, Any]
):
    """
    Create a factory function that returns new activation instances.

    Useful for building layers where each layer needs its own activation
    instance (important for parametric activations with learnable parameters).

    Args:
        activation_type: Activation type string
        config: Configuration for the activation

    Returns:
        A callable that returns a new activation module instance each time
    """
    def factory() -> nn.Module:
        return make_activation(activation_type, config)

    return factory
