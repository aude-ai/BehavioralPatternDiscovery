"""
SHAP Analysis

Extracts hierarchical SHAP weights showing how patterns at each level
compose into patterns at the next level. Dynamically iterates over
encoders and levels from the model.

Level transitions are computed as:
- level[i] -> level[i+1] for each consecutive level pair
- final_level (all encoders concatenated) -> unified

Designed for remote execution (Modal) - accepts data directly and
returns results without file I/O.
"""

import logging
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from src.model.base import BaseVAE

if TYPE_CHECKING:
    from src.model.distributions import LatentDistribution

logger = logging.getLogger(__name__)


class LevelTransitionWrapper(nn.Module):
    """
    Wrapper for SHAP: source_level_z -> target_level_z.

    Works with any consecutive level pair in the hierarchy.
    """

    def __init__(
        self,
        encoder: nn.Module,
        source_level: str,
        target_level: str,
        distribution: "LatentDistribution",
    ):
        """
        Args:
            encoder: HierarchicalEncoder instance
            source_level: Name of source level (e.g., "bottom")
            target_level: Name of target level (e.g., "mid")
            distribution: Latent distribution for getting means
        """
        super().__init__()
        self.target_level_module = encoder.levels[target_level]
        self.distribution = distribution

    def forward(self, source_z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: source_z -> target_z.

        Args:
            source_z: (batch, source_dim) latent from source level

        Returns:
            (batch, target_dim) mean latent at target level
        """
        target_output = self.target_level_module(source_z, training=False)
        return self.distribution.get_mean(target_output["params"])


class FinalLevelToUnifiedWrapper(nn.Module):
    """
    Wrapper for SHAP: concatenated final_level_z (all encoders) -> unified_z.
    """

    def __init__(self, vae: BaseVAE):
        """
        Args:
            vae: MultiEncoderVAE instance
        """
        super().__init__()
        self.unification = vae.unification
        self.distribution = vae.unified_distribution

    def forward(self, final_concat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: concatenated final latents -> unified_z.

        Args:
            final_concat: (batch, final_dim * num_encoders) concatenated latents

        Returns:
            (batch, unified_dim) mean unified latent
        """
        unified_output = self.unification(final_concat)
        return self.distribution.get_mean(unified_output["params"])


class SHAPAnalyzer:
    """Extract hierarchical SHAP weights between levels."""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Config containing shap section
        """
        shap_config = config.get("shap", {})
        self.explainer_type = shap_config.get("explainer_type", "gradient")
        self.background_samples = shap_config.get("background_samples", 100)
        self.shap_samples = shap_config.get("shap_samples", 50)

    def analyze(
        self,
        vae: BaseVAE,
        activations: dict[str, np.ndarray],
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Compute hierarchical SHAP weights.

        Dynamically iterates over encoders and levels from the model.

        Args:
            vae: Trained VAE implementing BaseVAE interface
            activations: Output from BatchScorer.score_all()
            progress_callback: Optional callback(progress) for progress updates

        Returns:
            Hierarchical weights dictionary with structure:
            {
                "enc1": {"level0_to_level1": {...}, "level1_to_level2": {...}},
                "enc2": {...},
                "final_to_unified": {...},
                "metadata": {
                    "level_names": [...],
                    "encoder_names": [...],
                    "level_dims": {...},
                    "unified_dim": int,
                    "num_encoders": int,
                }
            }
        """
        try:
            import shap  # noqa: F401
        except ImportError:
            raise ImportError("SHAP library required. Install with: pip install shap")

        vae.eval()
        device = next(vae.parameters()).device

        # Get dynamic structure from model
        level_names = vae.get_level_names()
        encoder_names = vae.get_encoder_names()

        logger.info(f"SHAP Analysis: {len(encoder_names)} encoders, {len(level_names)} levels")
        logger.info(f"  Encoders: {encoder_names}")
        logger.info(f"  Levels: {level_names}")

        hierarchical_weights = {}
        total_transitions = len(encoder_names) * (len(level_names) - 1) + 1
        completed_transitions = 0

        # Per-encoder hierarchical weights
        for enc_name in encoder_names:
            enc_idx = int(enc_name.replace("enc", "")) - 1
            encoder = vae.encoders[enc_idx]

            logger.info(f"Computing hierarchical weights for {enc_name}...")
            enc_weights = {}

            # Compute weights for each level transition
            for i in range(len(level_names) - 1):
                source_level = level_names[i]
                target_level = level_names[i + 1]
                transition_key = f"{source_level}_to_{target_level}"

                wrapper = LevelTransitionWrapper(
                    encoder=encoder,
                    source_level=source_level,
                    target_level=target_level,
                    distribution=vae.distribution,
                )

                source_key = f"{enc_name}_{source_level}"
                weights = self._compute_level_weights(
                    wrapper=wrapper,
                    source_activations=activations[source_key],
                    device=device,
                )
                enc_weights[transition_key] = weights
                logger.info(f"  Computed {transition_key} weights")

                completed_transitions += 1
                if progress_callback:
                    progress_callback(completed_transitions / total_transitions)

            hierarchical_weights[enc_name] = enc_weights
            logger.info(f"Completed hierarchical weights for {enc_name}")

        # Final level -> Unified weights
        logger.info("Computing final_level -> unified weights...")
        final_level = level_names[-1]

        final_concat = np.concatenate(
            [activations[f"{enc_name}_{final_level}"] for enc_name in encoder_names],
            axis=1,
        )

        final_to_unified_wrapper = FinalLevelToUnifiedWrapper(vae)
        final_to_unified = self._compute_level_weights(
            wrapper=final_to_unified_wrapper,
            source_activations=final_concat,
            device=device,
        )
        hierarchical_weights["final_to_unified"] = final_to_unified
        logger.info("Computed final_level -> unified weights")

        if progress_callback:
            progress_callback(1.0)

        # Add metadata for downstream consumers
        hierarchical_weights["metadata"] = {
            "level_names": level_names,
            "encoder_names": encoder_names,
            "level_dims": {level: vae.latent_dims[level] for level in level_names},
            "unified_dim": vae.unified_dim,
            "num_encoders": len(encoder_names),
        }

        return hierarchical_weights

    def _compute_level_weights(
        self,
        wrapper: nn.Module,
        source_activations: np.ndarray,
        device: torch.device,
    ) -> dict[str, list[dict[str, float]]]:
        """
        Compute SHAP weights for a single level transition.

        Args:
            wrapper: Level wrapper module
            source_activations: (n_samples, source_dim) activations
            device: Torch device

        Returns:
            Dictionary mapping target dimension to list of source contributions
        """
        import shap

        wrapper.to(device)
        wrapper.eval()

        n_samples = len(source_activations)

        # Select background samples
        bg_indices = np.random.choice(
            n_samples,
            min(self.background_samples, n_samples),
            replace=False,
        )
        background = torch.from_numpy(
            source_activations[bg_indices]
        ).float().to(device)

        # Select samples for SHAP computation
        shap_indices = np.random.choice(
            n_samples,
            min(self.shap_samples, n_samples),
            replace=False,
        )
        shap_data = torch.from_numpy(
            source_activations[shap_indices]
        ).float().to(device)

        # Create explainer
        if self.explainer_type == "gradient":
            explainer = shap.GradientExplainer(wrapper, background)
        elif self.explainer_type == "deep":
            explainer = shap.DeepExplainer(wrapper, background)
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")

        # Compute SHAP values
        shap_values = explainer.shap_values(shap_data)

        # Handle different return formats
        if isinstance(shap_values, list):
            # Stack outputs: (n_samples, source_dim, target_dim)
            shap_values = np.stack(shap_values, axis=-1)
        elif hasattr(shap_values, "numpy"):
            shap_values = shap_values.numpy()
        elif hasattr(shap_values, "values"):
            shap_values = shap_values.values

        # Average across samples: (source_dim, target_dim)
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Ensure correct shape
        if len(mean_shap.shape) == 1:
            # Single output dimension
            mean_shap = mean_shap.reshape(-1, 1)

        source_dim, target_dim = mean_shap.shape
        weights = {}

        for target_idx in range(target_dim):
            target_key = f"dim_{target_idx}"
            contributions = []

            for source_idx in range(source_dim):
                weight = float(mean_shap[source_idx, target_idx])
                if weight > 0.001:  # Filter near-zero weights
                    contributions.append({
                        f"dim_{source_idx}": weight
                    })

            # Sort by weight descending
            contributions.sort(
                key=lambda x: list(x.values())[0],
                reverse=True,
            )
            weights[target_key] = contributions

        return weights
