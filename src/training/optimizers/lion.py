"""
Lion Optimizer

Evolved Sign Momentum optimizer.
Uses sign-based updates with momentum, requiring less memory than Adam.

Reference:
    "Symbolic Discovery of Optimization Algorithms"
    Chen et al., 2023 (arXiv:2302.06675)
"""

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer: Evolved Sign Momentum.

    Uses sign-based updates with momentum, requiring less memory than Adam.
    Recommended: 3-10x smaller LR and 3-10x larger weight decay than AdamW.

    The update rule:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        θ_t = θ_{t-1} - lr * sign(m_t) - lr * λ * θ_{t-1}
        m_t = β2 * m_t + (1 - β2) * g_t  (for next step)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running average of gradient
                   (momentum for update, momentum for EMA)
            weight_decay: Decoupled weight decay (L2 regularization)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

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

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Decoupled weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Compute update direction
                update = exp_avg * beta1 + grad * (1 - beta1)

                # Sign-based update
                p.add_(update.sign_(), alpha=-group["lr"])

                # Update EMA for next step
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
