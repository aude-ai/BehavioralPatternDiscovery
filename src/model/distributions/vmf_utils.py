"""
Utility functions for von Mises-Fisher distribution.

Includes numerically stable Bessel function computation and sampling algorithms.
All operations use centralized numerical stability utilities from src/core/numerical.py.
"""

import logging
import math

import torch
from torch import Tensor

from src.core.numerical import (
    NumericalConstants,
    safe_div,
    safe_exp,
    safe_log,
    safe_normalize,
    safe_sqrt,
    safer_exp,
)

logger = logging.getLogger(__name__)


def log_iv(v: float, z: Tensor) -> Tensor:
    """
    Compute log of modified Bessel function of the first kind I_v(z).

    Uses asymptotic expansion for large z to avoid overflow.
    Numerically stable for all z > 0.

    Args:
        v: Order of Bessel function (typically d/2 - 1 for d-dimensional sphere)
        z: Input tensor (concentration parameter kappa), must be positive

    Returns:
        log(I_v(z)) computed stably
    """
    # Clamp z to safe range - below EPS_LOG we return asymptotic to zero
    z = z.clamp(min=NumericalConstants.EPS_LOG)

    threshold = NumericalConstants.VMF_BESSEL_ASYMPTOTIC_THRESHOLD

    # For v=0, use exponentially-scaled I_0
    if v == 0:
        # I_0(z) = exp(z) * I_0^e(z) where I_0^e is exponentially scaled
        # log(I_0(z)) = z + log(I_0^e(z))
        # torch.special.i0e is stable for all z
        i0e = torch.special.i0e(z)
        # Clamp before log to prevent log(0)
        log_i0e = safe_log(i0e)
        return z + log_i0e

    elif v == 0.5:
        # I_0.5(z) = sqrt(2/(pi*z)) * sinh(z)
        # log(I_0.5(z)) = 0.5*log(2/pi) - 0.5*log(z) + log(sinh(z))
        # For large z: log(sinh(z)) ≈ z - log(2)
        log_prefix = 0.5 * math.log(2.0 / math.pi) - 0.5 * safe_log(z)

        # Compute log(sinh(z)) stably
        log_sinh_z = torch.where(
            z > 20.0,
            z - math.log(2.0),  # Asymptotic: sinh(z) ≈ exp(z)/2
            safe_log(torch.sinh(z.clamp(max=20.0)))
        )
        return log_prefix + log_sinh_z

    else:
        # General case: switch between series and asymptotic
        # Asymptotic expansion: log(I_v(z)) ≈ z - 0.5*log(2*pi*z) - correction
        log_asymptotic = (
            z
            - 0.5 * safe_log(2.0 * math.pi * z)
            - (v * v - 0.25) / (2.0 * z.clamp(min=NumericalConstants.EPS_DIV))
        )

        # Series expansion for small z (computed where needed)
        log_series = _log_iv_series(v, z)

        return torch.where(z > threshold, log_asymptotic, log_series)


def _log_iv_series(v: float, z: Tensor) -> Tensor:
    """
    Compute log(I_v(z)) using series expansion for moderate z.

    I_v(z) = (z/2)^v * sum_{k=0}^inf (z^2/4)^k / (k! * Gamma(v+k+1))

    Computes entirely in log-space using log-sum-exp to avoid:
    1. Underflow in linear space for very small Bessel values
    2. safe_log clamping issues when sum_terms is small but non-zero

    Args:
        v: Order of Bessel function
        z: Input tensor

    Returns:
        log(I_v(z))
    """
    # Clamp z to prevent numerical issues
    z = z.clamp(min=NumericalConstants.EPS_LOG)

    log_half_z = safe_log(z / 2.0)
    log_z_sq_4 = 2 * safe_log(z) - math.log(4.0)

    # Compute series terms entirely in log space using log-sum-exp
    # log(term_k) = k*log(z^2/4) - lgamma(k+1) - lgamma(v+k+1)
    # But we can write: log(term_k) = log(term_0) + k*log(z^2/4) - sum_{j=1}^k log(j) - sum_{j=1}^k log(v+j)

    # Collect log of all terms for log-sum-exp
    log_gamma_v1 = math.lgamma(v + 1)

    # Start with term k=0: log(term_0) = -lgamma(v+1)
    log_terms = [torch.full_like(z, -log_gamma_v1)]

    # Running log(term_k)
    log_term = -log_gamma_v1

    for k in range(1, NumericalConstants.VMF_BESSEL_SERIES_TERMS):
        # Update: log(term_k) = log(term_{k-1}) + log(z^2/4) - log(k) - log(v+k)
        log_term = log_term + log_z_sq_4 - math.log(k) - math.log(v + k)
        log_terms.append(log_term.clone())

        # Early termination: if log_term is very negative, subsequent terms won't matter
        # Stop when term contributes less than 1e-15 relative to first term
        if (log_term - log_terms[0]).max().item() < -35:  # exp(-35) < 1e-15
            break

    # Stack all log terms: (num_terms, batch_size)
    log_terms_stacked = torch.stack(log_terms, dim=0)

    # Log-sum-exp to compute log(sum of terms) stably
    # log(sum(exp(log_terms))) = max(log_terms) + log(sum(exp(log_terms - max)))
    max_log_term = log_terms_stacked.max(dim=0).values  # (batch_size,)
    log_sum = max_log_term + torch.log(
        torch.sum(torch.exp(log_terms_stacked - max_log_term.unsqueeze(0)), dim=0)
    )

    # Final result: log(I_v(z)) = v*log(z/2) + log(sum)
    return v * log_half_z + log_sum


def log_vmf_normalizer(kappa: Tensor, dim: int) -> Tensor:
    """
    Compute log of vMF normalization constant C_d(kappa).

    C_d(kappa) = kappa^(d/2-1) / ((2*pi)^(d/2) * I_{d/2-1}(kappa))

    log(C_d) = (d/2-1)*log(kappa) - (d/2)*log(2*pi) - log(I_{d/2-1}(kappa))

    Args:
        kappa: Concentration parameter (any shape)
        dim: Dimensionality of the ambient space (sphere is S^{dim-1})

    Returns:
        log(C_d(kappa)) with same shape as kappa
    """
    half_dim = dim / 2.0
    order = half_dim - 1.0

    # Clamp kappa to safe range
    kappa = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)

    # Compute each term
    log_kappa_term = order * safe_log(kappa)
    log_pi_term = half_dim * math.log(2.0 * math.pi)
    log_bessel_term = log_iv(order, kappa)

    return log_kappa_term - log_pi_term - log_bessel_term


def log_sphere_surface_area(dim: int) -> float:
    """
    Compute log of surface area of unit sphere S^{dim-1}.

    Surface area = 2 * pi^{d/2} / Gamma(d/2)
    log(surface_area) = log(2) + (d/2)*log(pi) - lgamma(d/2)

    Args:
        dim: Ambient dimension (sphere is S^{dim-1})

    Returns:
        log(surface_area)
    """
    half_dim = dim / 2.0
    return math.log(2.0) + half_dim * math.log(math.pi) - math.lgamma(half_dim)


def log_uniform_density(dim: int) -> float:
    """
    Compute log of uniform density on unit sphere S^{dim-1}.

    Uniform density = 1 / surface_area

    Args:
        dim: Ambient dimension

    Returns:
        log(1/surface_area)
    """
    return -log_sphere_surface_area(dim)


def sample_vmf(mu: Tensor, kappa: Tensor) -> Tensor:
    """
    Sample from vMF distribution using rejection sampling (Ulrich 1984).

    This is the main sampling function - numerically stable for all kappa.

    Args:
        mu: Mean direction (batch, dim), must be unit normalized
        kappa: Concentration (batch,) or (batch, 1)

    Returns:
        Samples on unit sphere (batch, dim)
    """
    batch_size, dim = mu.shape
    device = mu.device
    dtype = mu.dtype

    # Ensure kappa is (batch, 1)
    if kappa.dim() == 1:
        kappa = kappa.unsqueeze(-1)

    # Identify uniform cases (very small kappa)
    is_uniform = kappa.squeeze(-1) < NumericalConstants.VMF_UNIFORM_THRESHOLD

    # Sample w (the component along mu direction)
    w = _sample_w_rejection(kappa, dim, device, dtype)

    # Sample v uniformly on S^{dim-2}
    v = torch.randn(batch_size, dim - 1, device=device, dtype=dtype)
    v = safe_normalize(v, dim=-1)

    # Construct sample in canonical frame (mu = e_dim)
    # x = sqrt(1 - w^2) * [v; 0] + w * e_dim
    sqrt_term = safe_sqrt(1.0 - w * w)  # (batch, 1)

    x_canonical = torch.zeros(batch_size, dim, device=device, dtype=dtype)
    x_canonical[:, :-1] = sqrt_term * v
    x_canonical[:, -1] = w.squeeze(-1)

    # Rotate to align with mu using Householder reflection
    x = _householder_rotate(x_canonical, mu)

    # For uniform case, just sample random direction
    if is_uniform.any():
        uniform_samples = torch.randn(batch_size, dim, device=device, dtype=dtype)
        uniform_samples = safe_normalize(uniform_samples, dim=-1)
        x = torch.where(is_uniform.unsqueeze(-1), uniform_samples, x)

    return x


def _sample_w_rejection(
    kappa: Tensor, dim: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    Sample the w component (projection onto mean direction) using rejection sampling.

    Based on Ulrich (1984) algorithm.

    Args:
        kappa: Concentration (batch, 1)
        dim: Ambient dimension
        device: Torch device
        dtype: Torch dtype

    Returns:
        w samples (batch, 1) in range [-1, 1]
    """
    batch_size = kappa.shape[0]
    m = dim - 1  # Dimension of tangent sphere

    # Clamp kappa for numerical stability
    kappa = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)

    # Compute rejection sampling parameters
    # b = (-2*kappa + sqrt(4*kappa^2 + m^2)) / m
    kappa_sq = kappa * kappa
    discriminant = safe_sqrt(4.0 * kappa_sq + m * m)
    b = safe_div(-2.0 * kappa + discriminant, torch.full_like(kappa, m))

    # a = (m + 2*kappa + sqrt(4*kappa^2 + m^2)) / 4
    a = (m + 2.0 * kappa + discriminant) / 4.0

    # d = 4*a*b / (1 + b) - m*log(m)
    d = safe_div(4.0 * a * b, 1.0 + b) - m * math.log(m)

    # Initialize output
    w = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    accepted = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for iteration in range(NumericalConstants.VMF_REJECTION_MAX_ITER):
        # Sample epsilon in (0, 1) - avoid boundaries
        eps = torch.rand(batch_size, 1, device=device, dtype=dtype)
        eps = eps.clamp(min=1e-7, max=1.0 - 1e-7)

        # Compute candidate w
        # w = (1 - (1+b)*eps) / (1 - (1-b)*eps)
        numerator = 1.0 - (1.0 + b) * eps
        denominator = 1.0 - (1.0 - b) * eps
        w_candidate = safe_div(numerator, denominator)

        # Clamp w to valid range [-1, 1]
        w_candidate = w_candidate.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)

        # Compute acceptance probability
        # t = 2*a*b / (1 - (1-b)*eps)
        t = safe_div(2.0 * a * b, denominator)
        t = t.clamp(min=NumericalConstants.EPS_LOG)

        # log(acceptance) = m*log(t) - t + d
        log_accept_prob = m * safe_log(t) - t + d

        # Accept/reject
        u = torch.rand(batch_size, 1, device=device, dtype=dtype)
        log_u = safe_log(u)
        accept = (log_u < log_accept_prob).squeeze(-1)

        # Update accepted samples
        newly_accepted = accept & (~accepted)
        w = torch.where(newly_accepted.unsqueeze(-1), w_candidate, w)
        accepted = accepted | newly_accepted

        # Early termination when all samples accepted
        if accepted.all().item():
            break

    # Fallback for non-converged samples: use mode
    if not accepted.all().item():
        fallback_w = safe_div(b, 1.0 + b)
        w = torch.where(accepted.unsqueeze(-1), w, fallback_w)

    return w


def _householder_rotate(x: Tensor, mu: Tensor) -> Tensor:
    """
    Rotate x so that e_dim (last basis vector) aligns with mu.

    Uses Householder reflection for numerical stability.

    Args:
        x: Points in canonical frame (batch, dim) where mean is e_dim
        mu: Target direction (batch, dim), must be unit normalized

    Returns:
        Rotated points (batch, dim)
    """
    dim = mu.shape[-1]
    device = mu.device
    dtype = mu.dtype

    # e_dim = [0, 0, ..., 0, 1]
    e = torch.zeros(dim, device=device, dtype=dtype)
    e[-1] = 1.0

    # Check if mu is already aligned with e (no rotation needed)
    mu_dot_e = mu[:, -1]  # (batch,)
    no_rotation_needed = mu_dot_e > NumericalConstants.VMF_ROTATION_THRESHOLD

    # Check if mu is anti-aligned with e (flip needed)
    flip_needed = mu_dot_e < -NumericalConstants.VMF_ROTATION_THRESHOLD

    # Householder vector: u = normalize(e - mu)
    u = e.unsqueeze(0) - mu  # (batch, dim)
    u = safe_normalize(u, dim=-1)

    # Householder reflection: H(u) @ x = x - 2 * (u^T x) * u
    u_dot_x = (u * x).sum(dim=-1, keepdim=True)  # (batch, 1)
    x_rotated = x - 2.0 * u_dot_x * u

    # Handle edge cases
    # If mu ≈ e: no rotation needed, return x
    # If mu ≈ -e: flip sign
    x_flipped = -x

    # Apply conditions
    result = torch.where(no_rotation_needed.unsqueeze(-1), x, x_rotated)
    result = torch.where(flip_needed.unsqueeze(-1), x_flipped, result)

    # Ensure output is normalized (numerical safety)
    result = safe_normalize(result, dim=-1)

    return result
