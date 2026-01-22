"""
Latent Monitoring

Provides metrics and warnings for detecting latent collapse during training.
Monitors active units, latent statistics, and tensor flow through the model.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class LatentMonitor:
    """
    Monitors latent space health to detect collapse.

    Level names and encoder names are required at construction time
    (obtained from the model via get_level_names() and get_encoder_names()).

    Tracks:
        - Input/Output tensor statistics
        - Active units per level: Dimensions with std > threshold
        - Latent statistics: Mean absolute activation and std per encoder
        - Health warnings: Alerts when active ratio drops below threshold
    """

    def __init__(
        self,
        enabled: bool,
        level_names: list[str],
        encoder_names: list[str],
    ):
        """
        Args:
            enabled: Whether monitoring is enabled
            level_names: Ordered list of encoder level names (required)
            encoder_names: List of encoder names (required)
        """
        self.enabled = enabled
        self.active_unit_threshold = 0.01
        self.warn_threshold = 0.1

        # Dynamic level and encoder names (required, no defaults)
        self.level_names = level_names
        self.encoder_names = encoder_names

        # Store tensor stats from last forward pass
        self.last_tensor_stats: dict[str, dict[str, float]] = {}

        if self.enabled:
            logger.info(
                f"LatentMonitor: active_threshold={self.active_unit_threshold}, "
                f"warn_if_ratio<{self.warn_threshold}"
            )

    def compute_active_units(self, z: torch.Tensor) -> tuple[int, int]:
        """
        Count dimensions where std > threshold.

        Args:
            z: Latent tensor (batch, dim)

        Returns:
            Tuple of (active_count, total_count)
        """
        stds = z.std(dim=0)
        active = int((stds > self.active_unit_threshold).sum().item())
        return active, z.shape[-1]

    def capture_tensor_stats(
        self,
        x_input: torch.Tensor,
        x_output: torch.Tensor,
        encodings: dict[str, Any],
    ) -> None:
        """
        Capture tensor statistics for monitoring.

        Args:
            x_input: Input tensor
            x_output: Reconstructed output tensor
            encodings: Model encodings from forward pass
        """
        if not self.enabled:
            return

        self.last_tensor_stats = {}

        # Input stats
        self.last_tensor_stats["input"] = self._tensor_stats(x_input)

        # Output stats
        self.last_tensor_stats["output"] = self._tensor_stats(x_output)

        # Per-encoder, per-level latent stats
        for enc_name, enc_data in encodings.items():
            if not enc_name.startswith("enc"):
                continue

            for level in self.level_names:
                if level not in enc_data:
                    continue

                z = enc_data[level]["z"]
                key = f"{enc_name}_{level}"
                stats = self._tensor_stats(z)
                active, total = self.compute_active_units(z)
                stats["active"] = active
                stats["total"] = total
                self.last_tensor_stats[key] = stats

        # Unified latent
        if "unified" in encodings:
            z = encodings["unified"]["z"]
            stats = self._tensor_stats(z)
            active, total = self.compute_active_units(z)
            stats["active"] = active
            stats["total"] = total
            self.last_tensor_stats["unified"] = stats

    @staticmethod
    def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
        """Compute statistics for a tensor."""
        with torch.no_grad():
            return {
                "min": t.min().item(),
                "max": t.max().item(),
                "mean": t.mean().item(),
                "std": t.std().item(),
            }

    def check_health(self) -> list[str]:
        """
        Check for health warnings based on captured stats.

        Returns:
            List of warning messages
        """
        if not self.enabled or not self.last_tensor_stats:
            return []

        warnings = []

        # Check each encoder level
        for key, stats in self.last_tensor_stats.items():
            if "active" not in stats:
                continue

            active = stats["active"]
            total = stats["total"]
            ratio = active / total if total > 0 else 0

            if ratio < self.warn_threshold:
                warnings.append(
                    f"{key}: {active}/{total} active ({ratio:.0%} < {self.warn_threshold:.0%})"
                )

        return warnings

    def format_monitoring_line(self) -> str | None:
        """
        Format monitoring data as a single log line.

        Returns:
            Formatted string for logging, or None if monitoring disabled
        """
        if not self.enabled or not self.last_tensor_stats:
            return None

        parts = []

        # Input/Output stats (brief)
        if "input" in self.last_tensor_stats:
            s = self.last_tensor_stats["input"]
            parts.append(f"In:[{s['min']:.2f},{s['max']:.2f}]")

        if "output" in self.last_tensor_stats:
            s = self.last_tensor_stats["output"]
            parts.append(f"Out:[{s['min']:.2f},{s['max']:.2f}]")

        # Active units per encoder (compact format)
        for enc_name in self.encoder_names:
            enc_parts = []
            for level in self.level_names:
                key = f"{enc_name}_{level}"
                if key in self.last_tensor_stats:
                    stats = self.last_tensor_stats[key]
                    active = stats["active"]
                    total = stats["total"]
                    enc_parts.append(f"{level[0].upper()}:{active}/{total}")

            if enc_parts:
                parts.append(f"{enc_name}[{' '.join(enc_parts)}]")

        # Unified
        if "unified" in self.last_tensor_stats:
            stats = self.last_tensor_stats["unified"]
            active = stats["active"]
            total = stats["total"]
            parts.append(f"uni[{active}/{total}]")

        # Mean activations per level (aggregated across encoders)
        level_means = {}
        for level in self.level_names:
            means = []
            for enc_name in self.encoder_names:
                key = f"{enc_name}_{level}"
                if key in self.last_tensor_stats:
                    means.append(self.last_tensor_stats[key]["mean"])
            if means:
                level_means[level] = sum(means) / len(means)

        if "unified" in self.last_tensor_stats:
            level_means["unified"] = self.last_tensor_stats["unified"]["mean"]

        if level_means:
            mean_strs = [f"{k[0].upper()}:{v:.3f}" for k, v in level_means.items()]
            parts.append(f"μ[{' '.join(mean_strs)}]")

        # Std per level (aggregated)
        level_stds = {}
        for level in self.level_names:
            stds = []
            for enc_name in self.encoder_names:
                key = f"{enc_name}_{level}"
                if key in self.last_tensor_stats:
                    stds.append(self.last_tensor_stats[key]["std"])
            if stds:
                level_stds[level] = sum(stds) / len(stds)

        if "unified" in self.last_tensor_stats:
            level_stds["unified"] = self.last_tensor_stats["unified"]["std"]

        if level_stds:
            std_strs = [f"{k[0].upper()}:{v:.4f}" for k, v in level_stds.items()]
            parts.append(f"σ[{' '.join(std_strs)}]")

        return " | ".join(parts)

    def format_monitoring_line_compact(self) -> str | None:
        """
        Format monitoring data as a compact log line with range/mean/std per level.

        Format: In[rng μ σ] | Out[rng μ σ] | B[rng μ σ] | M[rng μ σ] | T[rng μ σ] | U[rng μ σ]

        Returns:
            Formatted string for logging, or None if monitoring disabled
        """
        if not self.enabled or not self.last_tensor_stats:
            return None

        parts = []

        # Helper to format stats for a single level
        def format_level_stats(label: str, stats: dict[str, float]) -> str:
            rng = f"{stats['min']:.2f},{stats['max']:.2f}"
            return f"{label}[{rng} μ:{stats['mean']:.2f} σ:{stats['std']:.2f}]"

        # Input stats
        if "input" in self.last_tensor_stats:
            parts.append(format_level_stats("In", self.last_tensor_stats["input"]))

        # Output stats
        if "output" in self.last_tensor_stats:
            parts.append(format_level_stats("Out", self.last_tensor_stats["output"]))

        # Per-level stats (aggregated across encoders)
        for level in self.level_names:
            label = level[0].upper()
            level_stats = {"min": [], "max": [], "mean": [], "std": []}
            for enc_name in self.encoder_names:
                key = f"{enc_name}_{level}"
                if key in self.last_tensor_stats:
                    s = self.last_tensor_stats[key]
                    level_stats["min"].append(s["min"])
                    level_stats["max"].append(s["max"])
                    level_stats["mean"].append(s["mean"])
                    level_stats["std"].append(s["std"])

            if level_stats["mean"]:
                aggregated = {
                    "min": min(level_stats["min"]),
                    "max": max(level_stats["max"]),
                    "mean": sum(level_stats["mean"]) / len(level_stats["mean"]),
                    "std": sum(level_stats["std"]) / len(level_stats["std"]),
                }
                parts.append(format_level_stats(label, aggregated))

        # Unified stats
        if "unified" in self.last_tensor_stats:
            parts.append(format_level_stats("U", self.last_tensor_stats["unified"]))

        return " | ".join(parts)
