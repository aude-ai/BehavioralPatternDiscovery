"""
Numerical Stability Constants and Safe Operations

This module provides centralized constants and utility functions for
numerically stable deep learning operations. All values are chosen
based on research into safe bounds for common operations.

CRITICAL: Always use these constants instead of ad-hoc epsilon values.
"""

import torch
from torch import Tensor


class NumericalConstants:
    """
    Central repository for numerical stability constants.

    These values are carefully chosen to:
    1. Prevent NaN/Inf in forward pass
    2. Maintain gradient flow
    3. Work with both FP32 and BF16/FP16
    """

    # Epsilon values for different operations
    EPS_DIV = 1e-6       # Division: prevents divide-by-zero
    EPS_LOG = 1e-7       # Logarithm: prevents log(0)
    EPS_SQRT = 1e-8      # Square root: prevents sqrt of tiny negatives
    EPS_PROB = 1e-7      # Probability: clamp for log-prob computations
    EPS_NORM = 1e-5      # Normalization: LayerNorm/RMSNorm epsilon

    # Exponential bounds (for torch.exp)
    EXP_CLAMP_MIN = -20.0   # exp(-20) ≈ 2e-9, safe underflow
    EXP_CLAMP_MAX = 20.0    # exp(20) ≈ 485M, safe in FP32

    # Logvar bounds (for VAE reparameterization)
    LOGVAR_CLAMP_MIN = -10.0  # std = exp(-5) ≈ 0.007
    LOGVAR_CLAMP_MAX = 4.0    # std = exp(2) ≈ 7.4

    # Unified layer uses tighter bounds (matches EPP)
    LOGVAR_UNIFIED_MIN = -8.0   # std = exp(-4) ≈ 0.018
    LOGVAR_UNIFIED_MAX = 2.0    # std = exp(1) ≈ 2.7

    # Gamma distribution bounds
    GAMMA_CONC_MIN = 0.5     # Safe for digamma/lgamma (0.5 is safe lower bound)
    GAMMA_CONC_MAX = 100.0   # Prevents overflow in digamma
    GAMMA_RATE_MIN = 0.01    # Safe for division and log
    GAMMA_RATE_MAX = 100.0   # Prevents extreme samples

    # Log-space Gamma bounds (for network output clamping)
    LOG_GAMMA_CONC_MIN = -0.7  # exp(-0.7) ≈ 0.5 (matches GAMMA_CONC_MIN)
    LOG_GAMMA_CONC_MAX = 4.6   # exp(4.6) ≈ 100
    LOG_GAMMA_RATE_MIN = -4.6  # exp(-4.6) ≈ 0.01 (matches GAMMA_RATE_MIN)
    LOG_GAMMA_RATE_MAX = 4.6   # exp(4.6) ≈ 100

    # Cross-entropy logit bounds
    LOGIT_CLAMP = 50.0  # Prevents extreme softmax

    # Gradient/Loss bounds
    LOSS_CLAMP_MAX = 1000.0  # Prevents gradient explosion

    # Sinkhorn OT bounds
    SINKHORN_DIV_MIN = 1e-6  # Division denominator minimum
    SINKHORN_RESULT_MAX = 1e6  # Result maximum after division

    # Von Mises-Fisher distribution bounds
    # kappa is the concentration parameter (kappa=0 is uniform on sphere)
    VMF_KAPPA_MIN = 1e-6          # Below this, treat as uniform
    VMF_KAPPA_MAX = 700.0         # exp(700) is near FP32 max, Bessel safe
    VMF_LOG_KAPPA_MIN = -14.0     # log(1e-6) ≈ -13.8
    VMF_LOG_KAPPA_MAX = 6.55      # log(700) ≈ 6.55

    # Bessel function computation thresholds
    VMF_BESSEL_ASYMPTOTIC_THRESHOLD = 50.0  # Use asymptotic expansion above this
    VMF_BESSEL_SERIES_TERMS = 50            # Max terms for series expansion

    # Sampling parameters
    VMF_REJECTION_MAX_ITER = 100  # Max iterations for rejection sampling
    VMF_UNIFORM_THRESHOLD = 1e-6  # Below this kappa, sample uniform

    # Householder rotation threshold
    VMF_ROTATION_THRESHOLD = 0.999  # mu·e threshold for skipping rotation


def safe_div(numerator: Tensor, denominator: Tensor, eps: float = NumericalConstants.EPS_DIV) -> Tensor:
    """
    Safe division that clamps denominator to prevent divide-by-zero.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor (will be clamped)
        eps: Minimum value for denominator

    Returns:
        numerator / clamped_denominator
    """
    return numerator / denominator.clamp(min=eps)


def safe_log(x: Tensor, eps: float = NumericalConstants.EPS_LOG) -> Tensor:
    """
    Safe logarithm that clamps input to prevent log(0) or log(negative).

    Args:
        x: Input tensor (will be clamped to positive)
        eps: Minimum value for input

    Returns:
        log(clamped_x)
    """
    return torch.log(x.clamp(min=eps))


def safe_exp(x: Tensor,
             min_val: float = NumericalConstants.EXP_CLAMP_MIN,
             max_val: float = NumericalConstants.EXP_CLAMP_MAX) -> Tensor:
    """
    Safe exponential that clamps input to prevent overflow.

    Args:
        x: Input tensor (will be clamped)
        min_val: Minimum value for input
        max_val: Maximum value for input

    Returns:
        exp(clamped_x)
    """
    return torch.exp(x.clamp(min=min_val, max=max_val))


def safer_exp(
    x: Tensor,
    min_val: float | None = None,
    max_val: float | None = None,
) -> Tensor:
    """
    Flexible exponential with optional input clamping.

    Unlike safe_exp which always clamps to [-20, 20], this function allows:
    - No clamping (min_val=None, max_val=None) for cases where underflow is acceptable
    - Custom clamp bounds for specific numerical requirements
    - One-sided clamping (only min or only max)

    Use cases:
    - Bessel function series: Need to allow very negative values (underflow to 0 is OK)
    - Probability computations: May need tighter or looser bounds

    Args:
        x: Input tensor
        min_val: Minimum input value (None = no lower clamp, allows underflow to 0)
        max_val: Maximum input value (None = no upper clamp, risk of overflow)

    Returns:
        exp(optionally_clamped_x)
    """
    if min_val is not None and max_val is not None:
        return torch.exp(x.clamp(min=min_val, max=max_val))
    elif min_val is not None:
        return torch.exp(x.clamp(min=min_val))
    elif max_val is not None:
        return torch.exp(x.clamp(max=max_val))
    else:
        return torch.exp(x)


def safe_sqrt(x: Tensor, eps: float = NumericalConstants.EPS_SQRT) -> Tensor:
    """
    Safe square root that clamps input to prevent sqrt(negative).

    Args:
        x: Input tensor (will be clamped to positive)
        eps: Minimum value for input

    Returns:
        sqrt(clamped_x)
    """
    return torch.sqrt(x.clamp(min=eps))


def safe_std(x: Tensor, dim: int, eps: float = NumericalConstants.EPS_DIV) -> Tensor:
    """
    Safe standard deviation that clamps result to prevent division by zero.

    Args:
        x: Input tensor
        dim: Dimension to compute std over
        eps: Minimum value for result

    Returns:
        clamped std(x, dim)
    """
    return x.std(dim=dim).clamp(min=eps)


def safe_var(x: Tensor, dim: int) -> Tensor:
    """
    Safe variance that returns zero for batch size < 2.

    Args:
        x: Input tensor
        dim: Dimension to compute variance over

    Returns:
        var(x, dim) or zeros if batch too small
    """
    if x.shape[dim] < 2:
        shape = list(x.shape)
        shape.pop(dim)
        return torch.zeros(shape, device=x.device, dtype=x.dtype)
    return x.var(dim=dim)


def safe_normalize(x: Tensor, dim: int = -1, eps: float = NumericalConstants.EPS_DIV) -> Tensor:
    """
    Safe L2 normalization that handles zero-norm vectors.

    For vMF and other spherical distributions, ensures vectors are on unit sphere
    even when input is near-zero.

    Args:
        x: Input tensor to normalize
        dim: Dimension to normalize over
        eps: Minimum norm value to prevent divide-by-zero

    Returns:
        Unit-normalized tensor (or random unit vector if input was zero)
    """
    norm = x.norm(dim=dim, keepdim=True).clamp(min=eps)
    return x / norm


def safe_sinh(x: Tensor,
              min_val: float = NumericalConstants.EXP_CLAMP_MIN,
              max_val: float = NumericalConstants.EXP_CLAMP_MAX) -> Tensor:
    """
    Safe hyperbolic sine that prevents overflow.

    For large |x|, sinh(x) ≈ sign(x) * exp(|x|) / 2, so we clamp input.

    Args:
        x: Input tensor
        min_val: Minimum value for input
        max_val: Maximum value for input

    Returns:
        sinh(clamped_x)
    """
    return torch.sinh(x.clamp(min=min_val, max=max_val))


def safe_log_sinh(x: Tensor, eps: float = NumericalConstants.EPS_LOG) -> Tensor:
    """
    Safe log(sinh(x)) that handles both small and large x.

    For small x: sinh(x) ≈ x, so log(sinh(x)) ≈ log(x)
    For large x: sinh(x) ≈ exp(x)/2, so log(sinh(x)) ≈ x - log(2)

    Args:
        x: Input tensor (must be positive for meaningful result)
        eps: Minimum value for small-x handling

    Returns:
        log(sinh(x)) computed stably
    """
    x = x.clamp(min=eps)
    # Threshold where asymptotic approximation is accurate
    threshold = 20.0
    return torch.where(
        x > threshold,
        x - 0.693147,  # log(2) ≈ 0.693147
        safe_log(torch.sinh(x.clamp(max=threshold)), eps=eps)
    )


def check_finite(x: Tensor, name: str = "tensor") -> None:
    """
    Assert that tensor contains no NaN or Inf values.

    Use during development to catch NaN sources. Remove in production.

    Args:
        x: Tensor to check
        name: Name for error message

    Raises:
        AssertionError: If tensor contains non-finite values
    """
    if not x.isfinite().all():
        non_finite_count = (~x.isfinite()).sum().item()
        raise AssertionError(
            f"Non-finite values in {name}: {non_finite_count} of {x.numel()} elements"
        )
