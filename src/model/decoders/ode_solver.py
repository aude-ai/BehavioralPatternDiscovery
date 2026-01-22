"""
ODE Solvers, Timestep Sampling, and Optimal Transport for Flow Matching

Provides:
- TimestepSampler: Sampling strategies for training timesteps
- ODESolver: ODE solvers for inference
- sinkhorn_coupling: Optimal transport coupling for straighter flow paths
"""

import logging
from typing import Callable, Literal

import torch

logger = logging.getLogger(__name__)


def sinkhorn_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    eps: float,
    n_iters: int,
) -> torch.Tensor:
    """
    Compute optimal transport coupling between noise (x0) and targets (x1).

    Uses Sinkhorn algorithm to find optimal assignment that minimizes
    transport cost. This creates straighter flow paths during training.

    Args:
        x0: (batch, dim) noise samples
        x1: (batch, dim) target samples
        eps: Entropic regularization (smaller = more optimal but less stable)
        n_iters: Number of Sinkhorn iterations

    Returns:
        (batch,) indices to reorder x0 to match x1
    """
    batch_size = x0.shape[0]
    device = x0.device

    # Cost matrix: squared L2 distance
    C = torch.cdist(x0, x1, p=2) ** 2

    # Sinkhorn kernel - clamp exponent to prevent overflow/underflow
    K = torch.exp((-C / eps).clamp(min=-20, max=20))

    # Initialize marginals
    u = torch.ones(batch_size, device=device) / batch_size
    v = torch.ones(batch_size, device=device) / batch_size

    # Sinkhorn iterations with safe division and result clamping
    for _ in range(n_iters):
        denom_u = (K @ v).clamp(min=1e-6)
        u = (1.0 / denom_u).clamp(max=1e6)
        denom_v = (K.T @ u).clamp(min=1e-6)
        v = (1.0 / denom_v).clamp(max=1e6)

    # Transport plan
    P = torch.diag(u) @ K @ torch.diag(v)

    # Get assignment (argmax per row)
    indices = P.argmax(dim=1)
    return indices


class TimestepSampler:
    """
    Timestep sampling strategies for flow matching training.

    Supported distributions:
    - uniform: Uniform sampling in [0, 1]
    - logit_normal: Concentrates samples away from boundaries (center-focused)
    - u_shaped: Concentrates samples at boundaries t=0 and t=1 (boundary-focused)
    """

    def __init__(
        self,
        distribution: Literal["uniform", "logit_normal", "u_shaped"],
        logit_normal_mean: float,
        logit_normal_std: float,
        u_shaped_concentration: float,
    ):
        """
        Args:
            distribution: Sampling distribution type
            logit_normal_mean: Mean for logit-normal distribution
            logit_normal_std: Std for logit-normal distribution
            u_shaped_concentration: Concentration for U-shaped Beta distribution
        """
        self.distribution = distribution
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.u_shaped_concentration = u_shaped_concentration

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps in [0, 1].

        Args:
            batch_size: Number of timesteps to sample
            device: Torch device

        Returns:
            (batch_size,) tensor of timesteps
        """
        if self.distribution == "uniform":
            return torch.rand(batch_size, device=device)

        elif self.distribution == "logit_normal":
            # t = sigmoid(N(mean, std))
            # Concentrates samples away from 0 and 1 (center-focused)
            normal_samples = torch.randn(batch_size, device=device)
            normal_samples = normal_samples * self.logit_normal_std + self.logit_normal_mean
            return torch.sigmoid(normal_samples)

        elif self.distribution == "u_shaped":
            # Beta(a, a) with a < 1 gives U-shaped distribution
            # Higher concentration = stronger focus on boundaries
            # concentration=2 gives a=0.5 (strong U-shape)
            # concentration=1 gives a=1.0 (uniform)
            a = 1.0 / self.u_shaped_concentration
            beta_dist = torch.distributions.Beta(a, a)
            return beta_dist.sample((batch_size,)).to(device)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class ODESolver:
    """
    ODE solver for flow matching inference.

    Solves dx/dt = v(x, t) from t=0 to t=1.

    Supported methods:
    - euler: First-order Euler method
    - heun: Second-order Heun's method (predictor-corrector)
    - midpoint: Second-order midpoint method
    """

    def __init__(self, method: Literal["euler", "heun", "midpoint"]):
        """
        Args:
            method: ODE solver method
        """
        self.method = method

    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Solve ODE dx/dt = velocity_fn(x, t) from t=0 to t=1.

        Args:
            velocity_fn: Function (x_t, t) -> velocity
            x_0: (batch_size, dim) initial condition (noise)
            num_steps: Number of integration steps

        Returns:
            (batch_size, dim) final state x_1 (reconstructed data)

        Raises:
            ValueError: If num_steps < 1
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        if self.method == "euler":
            return self._euler(velocity_fn, x_0, num_steps)
        elif self.method == "heun":
            return self._heun(velocity_fn, x_0, num_steps)
        elif self.method == "midpoint":
            return self._midpoint(velocity_fn, x_0, num_steps)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _euler(
        self,
        velocity_fn: Callable,
        x_0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Euler method (first-order)."""
        x_t = x_0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            v = velocity_fn(x_t, t)
            x_t = x_t + dt * v

        return x_t

    def _heun(
        self,
        velocity_fn: Callable,
        x_0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Heun's method (second-order predictor-corrector)."""
        x_t = x_0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            t_next = torch.full((x_t.shape[0],), (i + 1) * dt, device=x_t.device)

            # Predictor
            v1 = velocity_fn(x_t, t)
            x_pred = x_t + dt * v1

            # Corrector
            v2 = velocity_fn(x_pred, t_next)
            x_t = x_t + dt * (v1 + v2) / 2

        return x_t

    def _midpoint(
        self,
        velocity_fn: Callable,
        x_0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Midpoint method (second-order)."""
        x_t = x_0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            t_mid = torch.full((x_t.shape[0],), (i + 0.5) * dt, device=x_t.device)

            # Midpoint evaluation
            v1 = velocity_fn(x_t, t)
            x_mid = x_t + 0.5 * dt * v1
            v_mid = velocity_fn(x_mid, t_mid)
            x_t = x_t + dt * v_mid

        return x_t
