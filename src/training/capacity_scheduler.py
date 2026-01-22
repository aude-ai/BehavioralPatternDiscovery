"""
Adaptive Capacity Scheduler

Per-level capacity scheduling with warmup KL tracking.

Phases:
1. Warmup: No KL penalty, track actual KL values
2. Ramp Down: Start at measured average KL, decrease to floor target
3. Ramp Up: Increase from floor to final target
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CapacityTracker:
    """
    Tracks capacity for a single level (bottom, mid, top, or unified).

    Phases:
    1. Warmup (epochs 0 to warmup_end): No penalty, collect KL samples
    2. Ramp Down (warmup_end to ramp_down_end): Decrease from avg KL to floor
    3. Ramp Up (ramp_down_end to ramp_up_end): Increase from floor to target
    4. Hold (after ramp_up_end): Stay at target
    """

    def __init__(
        self,
        name: str,
        floor_target: float,
        final_target: float,
        warmup_epochs: int,
        ramp_down_epochs: int,
        ramp_up_epochs: int,
    ):
        """
        Args:
            name: Identifier (e.g., "bottom", "mid", "top", "unified")
            floor_target: Minimum capacity at end of ramp down
            final_target: Final capacity at end of ramp up
            warmup_epochs: Epochs to collect KL without penalty
            ramp_down_epochs: Epochs to ramp from avg KL to floor
            ramp_up_epochs: Epochs to ramp from floor to final
        """
        self.name = name
        self.floor_target = floor_target
        self.final_target = final_target
        self.warmup_epochs = warmup_epochs
        self.ramp_down_epochs = ramp_down_epochs
        self.ramp_up_epochs = ramp_up_epochs

        # Phase boundaries
        self.warmup_end = warmup_epochs
        self.ramp_down_end = self.warmup_end + ramp_down_epochs
        self.ramp_up_end = self.ramp_down_end + ramp_up_epochs

        # Warmup KL tracking
        self.warmup_kl_samples: list[float] = []
        self.warmup_avg_kl: float = 0.0
        self.warmup_complete = False

    def record_warmup_kl(self, kl_value: float):
        """Record KL value during warmup phase."""
        if not self.warmup_complete:
            self.warmup_kl_samples.append(kl_value)

    def finalize_warmup(self):
        """Compute average KL from warmup samples."""
        if self.warmup_kl_samples:
            self.warmup_avg_kl = sum(self.warmup_kl_samples) / len(self.warmup_kl_samples)
        else:
            # Fallback if no samples (shouldn't happen)
            self.warmup_avg_kl = self.final_target
        self.warmup_complete = True
        logger.info(
            f"CapacityTracker[{self.name}]: Warmup complete. "
            f"Avg KL={self.warmup_avg_kl:.2f} from {len(self.warmup_kl_samples)} samples. "
            f"Will ramp {self.warmup_avg_kl:.2f} → {self.floor_target:.2f} → {self.final_target:.2f}"
        )

    def get_capacity(self, epoch: int) -> float:
        """
        Get current capacity for given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Current capacity target (0.0 during warmup = no penalty)
        """
        if epoch < self.warmup_end:
            # Warmup: return 0 to disable KL penalty
            return 0.0

        # Finalize warmup on first call after warmup ends
        if not self.warmup_complete:
            self.finalize_warmup()

        if epoch < self.ramp_down_end:
            # Ramp down: avg_kl → floor
            if self.ramp_down_epochs <= 0:
                return self.floor_target
            progress = (epoch - self.warmup_end) / self.ramp_down_epochs
            return self.warmup_avg_kl + progress * (self.floor_target - self.warmup_avg_kl)

        elif epoch < self.ramp_up_end:
            # Ramp up: floor → final
            if self.ramp_up_epochs <= 0:
                return self.final_target
            progress = (epoch - self.ramp_down_end) / self.ramp_up_epochs
            return self.floor_target + progress * (self.final_target - self.floor_target)

        else:
            # Hold at final
            return self.final_target

    def in_warmup(self, epoch: int) -> bool:
        """Check if currently in warmup phase."""
        return epoch < self.warmup_end

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "warmup_kl_samples": self.warmup_kl_samples,
            "warmup_avg_kl": self.warmup_avg_kl,
            "warmup_complete": self.warmup_complete,
        }

    def load_state_dict(self, state: dict[str, Any]):
        """Load state from checkpoint."""
        self.warmup_kl_samples = state["warmup_kl_samples"]
        self.warmup_avg_kl = state["warmup_avg_kl"]
        self.warmup_complete = state["warmup_complete"]


class CapacitySchedulerManager:
    """
    Manages capacity schedulers for each level.

    Supports dynamic hierarchy levels - level names and counts are
    read from config rather than being hardcoded.

    All encoders at the same level share the same capacity schedule.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Full config containing:
                - capacity section with schedule settings (targets as list format)
                - model.encoder.hierarchical.levels (for level names)
        """
        capacity_config = config["capacity"]

        # Epoch settings (same for all trackers)
        self.warmup_epochs = capacity_config["warmup_epochs"]
        self.ramp_down_epochs = capacity_config["ramp_down_epochs"]
        self.ramp_up_epochs = capacity_config["ramp_up_epochs"]

        # Parse targets from list format into dict by name
        floor_targets_list = capacity_config["floor_targets"]
        final_targets_list = capacity_config["final_targets"]
        floor_targets = {item["name"]: item["value"] for item in floor_targets_list}
        final_targets = {item["name"]: item["value"] for item in final_targets_list}

        # Get level names from model config (dynamic)
        levels_config = config["model"]["encoder"]["hierarchical"]["levels"]
        self.level_names = [level["name"] for level in levels_config]

        # Create per-level trackers (levels + unified)
        self.trackers: dict[str, CapacityTracker] = {}

        for level in self.level_names + ["unified"]:
            if level not in floor_targets:
                raise KeyError(
                    f"[CapacitySchedulerManager] Missing floor_target for level '{level}'. "
                    f"Available: {list(floor_targets.keys())}"
                )
            if level not in final_targets:
                raise KeyError(
                    f"[CapacitySchedulerManager] Missing final_target for level '{level}'. "
                    f"Available: {list(final_targets.keys())}"
                )
            self.trackers[level] = CapacityTracker(
                name=level,
                floor_target=floor_targets[level],
                final_target=final_targets[level],
                warmup_epochs=self.warmup_epochs,
                ramp_down_epochs=self.ramp_down_epochs,
                ramp_up_epochs=self.ramp_up_epochs,
            )

        logger.info(
            f"CapacitySchedulerManager: {len(self.trackers)} level trackers "
            f"({', '.join(self.level_names)} + unified). "
            f"Phases: warmup={self.warmup_epochs}, "
            f"ramp_down={self.ramp_down_epochs}, "
            f"ramp_up={self.ramp_up_epochs}"
        )

    def record_kl(self, level: str, kl_value: float, epoch: int):
        """
        Record KL value for a level tracker (only used during warmup).

        Args:
            level: Level name ("bottom", "mid", "top", "unified")
            kl_value: Current KL divergence value
            epoch: Current epoch
        """
        if level in self.trackers and self.trackers[level].in_warmup(epoch):
            self.trackers[level].record_warmup_kl(kl_value)

    def get_capacity(self, level: str, epoch: int) -> float:
        """
        Get capacity for a specific level.

        Args:
            level: Level name ("bottom", "mid", "top", "unified")
            epoch: Current epoch

        Returns:
            Current capacity target
        """
        if level not in self.trackers:
            raise KeyError(f"Unknown level: {level}")
        return self.trackers[level].get_capacity(epoch)

    def in_warmup(self, epoch: int) -> bool:
        """Check if still in warmup phase."""
        return epoch < self.warmup_epochs

    def get_all_capacities(self, epoch: int, num_encoders: int) -> dict[str, float]:
        """
        Get current capacities for all encoder-level combinations.

        Returns dict with keys like "enc1_bottom", "enc2_mid", etc.
        All encoders at the same level get the same capacity.

        Args:
            epoch: Current epoch
            num_encoders: Number of encoders

        Returns:
            Dict mapping encoder-level keys to capacity values
        """
        capacities = {}

        for enc_idx in range(1, num_encoders + 1):
            enc_name = f"enc{enc_idx}"
            for level in self.level_names:
                key = f"{enc_name}_{level}"
                capacities[key] = self.trackers[level].get_capacity(epoch)

        capacities["unified"] = self.trackers["unified"].get_capacity(epoch)

        return capacities

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {
            key: tracker.state_dict()
            for key, tracker in self.trackers.items()
        }

    def load_state_dict(self, state: dict[str, Any]):
        """Load state from checkpoint."""
        for key, tracker_state in state.items():
            if key in self.trackers:
                self.trackers[key].load_state_dict(tracker_state)
