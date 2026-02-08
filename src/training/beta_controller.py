"""
Beta Controller for Dynamic β Adjustment

Supports two controller types:
1. Dual Gradient Descent (default) - Lagrangian approach to constrained optimization
2. PI Controller (legacy) - Proportional-Integral control

The controller adjusts β to satisfy the constraint: KL ≈ capacity
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseBetaController(ABC):
    """Abstract base class for beta controllers."""

    @abstractmethod
    def update(self, current_kl: float, target_capacity: float) -> float:
        """Update beta based on current KL and target capacity."""
        pass

    @abstractmethod
    def get_beta(self) -> float:
        """Get current beta value."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state."""
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, float]:
        """Get state for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state: dict[str, float]) -> None:
        """Load state from checkpoint."""
        pass


class DualGradientController(BaseBetaController):
    """
    Dual Gradient Descent controller for beta adjustment.

    Treats beta as a Lagrange multiplier and performs gradient ascent
    on the dual variable to satisfy the constraint KL = capacity.

    Update rule:
        constraint = KL - capacity
        beta = beta + lr_dual * constraint
        beta = clamp(beta, beta_min, beta_max)

    When KL > capacity: constraint > 0, beta increases, pushes KL down
    When KL < capacity: constraint < 0, beta decreases, lets KL rise
    When KL = capacity: constraint = 0, beta unchanged (equilibrium)
    """

    def __init__(
        self,
        lr_dual: float,
        beta_init: float,
        beta_min: float,
        beta_max: float,
        use_constraint_ema: bool,
        constraint_ema_decay: float,
        name: str = "",
    ):
        """
        Args:
            lr_dual: Learning rate for dual variable updates
            beta_init: Initial beta value
            beta_min: Minimum beta value
            beta_max: Maximum beta value
            use_constraint_ema: Whether to smooth constraint with EMA
            constraint_ema_decay: Decay factor for constraint EMA (0.9 = 10% new, 90% old)
            name: Controller name for logging
        """
        self.lr_dual = lr_dual
        self.beta = beta_init
        self.beta_init = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.use_constraint_ema = use_constraint_ema
        self.constraint_ema_decay = constraint_ema_decay
        self.name = name

        # EMA state
        self.constraint_ema: float | None = None

    def update(self, current_kl: float, target_capacity: float) -> float:
        """
        Update beta based on constraint violation.

        Args:
            current_kl: Current KL divergence value
            target_capacity: Target capacity C

        Returns:
            Updated beta value
        """
        constraint = current_kl - target_capacity

        # Optionally smooth constraint with EMA
        if self.use_constraint_ema:
            if self.constraint_ema is None:
                self.constraint_ema = constraint
            else:
                self.constraint_ema = (
                    self.constraint_ema_decay * self.constraint_ema
                    + (1.0 - self.constraint_ema_decay) * constraint
                )
            constraint = self.constraint_ema

        # Gradient ascent on dual variable
        self.beta = self.beta + self.lr_dual * constraint

        # Project to valid range
        self.beta = max(self.beta_min, min(self.beta_max, self.beta))

        return self.beta

    def get_beta(self) -> float:
        return self.beta

    def reset(self) -> None:
        self.beta = self.beta_init
        self.constraint_ema = None

    def state_dict(self) -> dict[str, float]:
        state: dict[str, float] = {"beta": self.beta}
        if self.use_constraint_ema and self.constraint_ema is not None:
            state["constraint_ema"] = self.constraint_ema
        return state

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.beta = state["beta"]
        if "constraint_ema" in state:
            self.constraint_ema = state["constraint_ema"]


class PIController(BaseBetaController):
    """
    Legacy PI Controller for beta adjustment.

    Kept for comparison and backwards compatibility.
    Uses proportional-integral control with optional leaky integral.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        beta_init: float,
        beta_min: float,
        beta_max: float,
        use_absolute_error: bool,
        beta_decay: float,
        name: str = "",
    ):
        self.kp = kp
        self.ki = ki
        self.beta = beta_init
        self.beta_init = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.use_absolute_error = use_absolute_error
        self.beta_decay = beta_decay
        self.name = name
        self.prev_error = 0.0

    def update(self, current_kl: float, target_capacity: float) -> float:
        if self.use_absolute_error:
            error = abs(current_kl - target_capacity)
        else:
            error = current_kl - target_capacity

        delta = self.kp * (error - self.prev_error) + self.ki * error
        self.beta = self.beta + delta

        if self.beta_decay < 1.0:
            self.beta = (
                self.beta_decay * self.beta
                + (1.0 - self.beta_decay) * self.beta_min
            )

        self.beta = max(self.beta_min, min(self.beta_max, self.beta))
        self.prev_error = error
        return self.beta

    def get_beta(self) -> float:
        return self.beta

    def reset(self) -> None:
        self.beta = self.beta_init
        self.prev_error = 0.0

    def state_dict(self) -> dict[str, float]:
        return {"beta": self.beta, "prev_error": self.prev_error}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.beta = state["beta"]
        self.prev_error = state["prev_error"]


class BetaControllerManager:
    """
    Manages beta controllers for per-encoder, per-level control.

    Structure:
        - N encoders × M levels = N*M encoder-level controllers
        - 1 unified controller
        - Total: N*M + 1 controllers

    Supports dynamic hierarchy levels - level names and counts are
    read from config rather than being hardcoded.

    Supports both Dual Gradient Descent and PI Controller via config.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Full config containing:
                - beta_controller section (with beta_init as list)
                - model.encoder.num_encoders
                - model.encoder.hierarchical.levels (for level names)
        """
        beta_config = config["beta_controller"]

        self.enabled = beta_config["enabled"]
        if not self.enabled:
            logger.info("Beta Controller disabled")
            return

        self.controller_type = beta_config["type"]
        self.beta_min = beta_config["beta_min"]
        self.beta_max = beta_config["beta_max"]

        # Parse beta_init from list format into dict by name
        beta_init_list = beta_config["beta_init"]
        beta_init_config = {item["name"]: item["value"] for item in beta_init_list}

        self.num_encoders = config["model"]["encoder"]["num_encoders"]

        # Get level names from model config (dynamic)
        levels_config = config["model"]["encoder"]["hierarchical"]["levels"]
        self.level_names = [level["name"] for level in levels_config]

        # Create controllers based on type
        self.encoder_level_controllers: dict[str, BaseBetaController] = {}

        for enc_idx in range(1, self.num_encoders + 1):
            enc_name = f"enc{enc_idx}"
            for level in self.level_names:
                key = f"{enc_name}_{level}"
                if level not in beta_init_config:
                    raise KeyError(
                        f"[BetaControllerManager] Missing beta_init for level '{level}'. "
                        f"Available: {list(beta_init_config.keys())}"
                    )
                level_beta_init = beta_init_config[level]
                self.encoder_level_controllers[key] = self._create_controller(
                    beta_config=beta_config,
                    beta_init=level_beta_init,
                    name=key,
                )

        # Unified controller
        if "unified" not in beta_init_config:
            raise KeyError(
                f"[BetaControllerManager] Missing beta_init for 'unified'. "
                f"Available: {list(beta_init_config.keys())}"
            )
        unified_beta_init = beta_init_config["unified"]
        self.unified_controller = self._create_controller(
            beta_config=beta_config,
            beta_init=unified_beta_init,
            name="unified",
        )

        total_controllers = len(self.encoder_level_controllers) + 1
        logger.info(
            f"Initialized BetaControllerManager with {total_controllers} "
            f"{self.controller_type} controllers "
            f"({self.num_encoders} encoders × {len(self.level_names)} levels + unified)"
        )

    def _create_controller(
        self,
        beta_config: dict[str, Any],
        beta_init: float,
        name: str,
    ) -> BaseBetaController:
        """Create a controller based on config type."""
        controller_type = beta_config["type"]

        if controller_type == "dual_gradient":
            dg_config = beta_config["dual_gradient"]
            return DualGradientController(
                lr_dual=dg_config["lr_dual"],
                beta_init=beta_init,
                beta_min=beta_config["beta_min"],
                beta_max=beta_config["beta_max"],
                use_constraint_ema=dg_config["use_constraint_ema"],
                constraint_ema_decay=dg_config["constraint_ema_decay"],
                name=name,
            )
        elif controller_type == "pi":
            pi_config = beta_config["pi"]
            return PIController(
                kp=pi_config["kp"],
                ki=pi_config["ki"],
                beta_init=beta_init,
                beta_min=beta_config["beta_min"],
                beta_max=beta_config["beta_max"],
                use_absolute_error=pi_config["use_absolute_error"],
                beta_decay=pi_config["beta_decay"],
                name=name,
            )
        else:
            raise ValueError(
                f"Unknown beta controller type: '{controller_type}'. "
                f"Available: ['dual_gradient', 'pi']"
            )

    def update_encoder_level(
        self,
        enc_name: str,
        level: str,
        current_kl: float,
        target_capacity: float,
    ) -> float:
        """Update encoder-level controller and return new beta."""
        if not self.enabled:
            return 1.0

        key = f"{enc_name}_{level}"
        if key not in self.encoder_level_controllers:
            raise KeyError(
                f"Unknown encoder-level: '{key}'. "
                f"Available: {list(self.encoder_level_controllers.keys())}"
            )

        return self.encoder_level_controllers[key].update(current_kl, target_capacity)

    def update_unified(
        self,
        current_kl: float,
        target_capacity: float,
    ) -> float:
        """Update unified controller and return new beta."""
        if not self.enabled:
            return 1.0

        return self.unified_controller.update(current_kl, target_capacity)

    def get_beta(self, enc_name: str, level: str) -> float:
        """Get beta for a specific encoder-level."""
        if not self.enabled:
            return 1.0

        key = f"{enc_name}_{level}"
        if key not in self.encoder_level_controllers:
            raise KeyError(
                f"Unknown encoder-level: '{key}'. "
                f"Available: {list(self.encoder_level_controllers.keys())}"
            )

        return self.encoder_level_controllers[key].get_beta()

    def get_unified_beta(self) -> float:
        """Get beta for unified layer."""
        if not self.enabled:
            return 1.0

        return self.unified_controller.get_beta()

    def get_level_avg_betas(self) -> dict[str, float]:
        """Get average beta per level for compact logging."""
        if not self.enabled:
            return {}

        # Initialize with dynamic level names
        level_betas: dict[str, list[float]] = {level: [] for level in self.level_names}
        for key, controller in self.encoder_level_controllers.items():
            # Key format is "enc{n}_{level}"
            level = key.split("_", 1)[1]  # Split only on first underscore
            if level in level_betas:
                level_betas[level].append(controller.get_beta())

        avg_betas = {}
        for level, betas in level_betas.items():
            avg_betas[f"beta_{level}"] = sum(betas) / len(betas) if betas else 1.0
        avg_betas["beta_unified"] = self.unified_controller.get_beta()

        return avg_betas

    def get_all_betas(self) -> dict[str, float]:
        """Get all beta values for metrics logging."""
        if not self.enabled:
            return {}

        betas = {}
        for key, controller in self.encoder_level_controllers.items():
            betas[f"beta_{key}"] = controller.get_beta()
        betas["beta_unified"] = self.unified_controller.get_beta()

        return betas

    def reset_all(self) -> None:
        """Reset all controllers."""
        if not self.enabled:
            return

        for controller in self.encoder_level_controllers.values():
            controller.reset()
        self.unified_controller.reset()

    def state_dict(self) -> dict[str, Any]:
        """Get manager state for checkpointing."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "controller_type": self.controller_type,
            "encoder_levels": {
                key: ctrl.state_dict()
                for key, ctrl in self.encoder_level_controllers.items()
            },
            "unified": self.unified_controller.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load manager state from checkpoint."""
        if not state.get("enabled", False):
            return

        for key, ctrl_state in state.get("encoder_levels", {}).items():
            if key in self.encoder_level_controllers:
                self.encoder_level_controllers[key].load_state_dict(ctrl_state)

        if state.get("unified"):
            self.unified_controller.load_state_dict(state["unified"])
