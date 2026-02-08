"""
AdEMAMix Optimizer

Adaptive EMA Mix optimizer that combines fast and slow EMAs.

Reference:
    "The AdEMAMix Optimizer: Better, Faster, Older"
    Pagliardini et al., 2024 (arXiv:2409.03137)
"""
from __future__ import annotations

import math
import torch
from torch.optim import Optimizer


class AdEMAMix(Optimizer):
    """
    AdEMAMix optimizer: Adaptive EMA Mix.

    Combines two EMAs (fast and slow) to better utilize past gradients.
    The slow EMA allows gradients from tens of thousands of steps ago
    to still influence the current update.

    The update rule:
        m1_t = β1 * m1_{t-1} + (1 - β1) * g_t    (fast EMA)
        m2_t = β3 * m2_{t-1} + (1 - β3) * g_t    (slow EMA)
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        m1_hat = m1_t / (1 - β1^t)
        v_hat = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - lr * (m1_hat + α * m2_t) / (sqrt(v_hat) + ε)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        T_alpha: int | None = None,
        T_beta3: int | None = None,
    ):
        """
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients (β1, β2, β3) where:
                   - β1: Fast EMA momentum
                   - β2: Second moment EMA momentum
                   - β3: Slow EMA momentum
            alpha: Weight for slow EMA contribution
            weight_decay: Decoupled weight decay
            eps: Term added to denominator for numerical stability
            T_alpha: Scheduler for alpha ramp-up (None = constant)
            T_beta3: Scheduler for β3 ramp-up (None = constant)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta3 parameter: {betas[2]}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            weight_decay=weight_decay,
            eps=eps,
            T_alpha=T_alpha,
            T_beta3=T_beta3,
        )
        super().__init__(params, defaults)
        self._step = 0

    def _get_alpha(self, alpha: float, T_alpha: int | None) -> float:
        """Get scheduled alpha value."""
        if T_alpha is None or T_alpha <= 0:
            return alpha
        return min(self._step * alpha / T_alpha, alpha)

    def _get_beta3(
        self,
        beta1: float,
        beta3: float,
        T_beta3: int | None,
    ) -> float:
        """Get scheduled beta3 value (log-linear interpolation)."""
        if T_beta3 is None or T_beta3 <= 0 or self._step >= T_beta3:
            return beta3

        t_ratio = self._step / T_beta3
        log_beta1 = math.log(beta1)
        log_beta3 = math.log(beta3)
        log_beta = log_beta1 + t_ratio * (log_beta3 - log_beta1)
        return min(math.exp(log_beta), beta3)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            alpha_t = self._get_alpha(group["alpha"], group["T_alpha"])
            beta3_t = self._get_beta3(beta1, beta3, group["T_beta3"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)  # Fast EMA
                    state["m2"] = torch.zeros_like(p)  # Slow EMA
                    state["v"] = torch.zeros_like(p)   # Second moment

                state["step"] += 1
                step = state["step"]

                m1, m2, v = state["m1"], state["m2"], state["v"]

                # Update fast EMA
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m1_hat = m1 / (1 - beta1 ** step)

                # Update slow EMA
                m2.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)

                # Update second moment
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                v_hat = v / (1 - beta2 ** step)

                # Compute denominator
                denom = v_hat.sqrt().add_(group["eps"])

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # Update parameters
                p.addcdiv_(m1_hat + alpha_t * m2, denom, value=-group["lr"])

        return loss
