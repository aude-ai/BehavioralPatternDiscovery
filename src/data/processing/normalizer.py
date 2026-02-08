"""
Unified Normalizer

Chainable normalization pipeline for embeddings.
Replaces per-encoder normalization and variance transforms.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import stats

from src.core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Registry for individual normalizations
normalization_registry = ComponentRegistry("normalization")


class BaseNormalization(ABC):
    """Interface for all normalization implementations."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseNormalization":
        """Fit normalization parameters from data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get fitted parameters for persistence."""
        pass

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, Any]) -> "BaseNormalization":
        """Reconstruct from saved parameters."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Normalization name for logging."""
        pass


@normalization_registry.register("l2")
class L2Normalization(BaseNormalization):
    """
    L2 normalization - project vectors to unit sphere.

    This replicates the built-in normalization from:
    - Jina v3: normalize_embeddings=True
    - NV-Embed v2: torch.nn.functional.normalize(embeddings, p=2, dim=1)

    Each vector is divided by its L2 norm, resulting in unit-length vectors.
    """

    def __init__(self, config: dict | None = None):
        self._fitted = False

    def fit(self, X: np.ndarray) -> "L2Normalization":
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return X / norms

    def get_params(self) -> dict[str, Any]:
        return {"type": "l2", "fitted": self._fitted}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "L2Normalization":
        instance = cls()
        instance._fitted = params["fitted"]
        return instance

    @property
    def name(self) -> str:
        return "l2"


@normalization_registry.register("maxabs")
class MaxAbsNormalization(BaseNormalization):
    """MaxAbs scaling - scale each dimension to [-1, 1] by max absolute value."""

    def __init__(self, config: dict | None = None):
        self.max_abs: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MaxAbsNormalization":
        self.max_abs = np.abs(X).max(axis=0)
        self.max_abs = np.maximum(self.max_abs, 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.max_abs is None:
            raise RuntimeError("MaxAbsNormalization not fitted")
        return X / self.max_abs

    def get_params(self) -> dict[str, Any]:
        return {"type": "maxabs", "max_abs": self.max_abs.tolist()}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "MaxAbsNormalization":
        instance = cls()
        instance.max_abs = np.array(params["max_abs"])
        return instance

    @property
    def name(self) -> str:
        return "maxabs"


@normalization_registry.register("standard")
class StandardNormalization(BaseNormalization):
    """Standard scaling - zero mean, unit variance per dimension."""

    def __init__(self, config: dict | None = None):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardNormalization":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std = np.maximum(self.std, 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardNormalization not fitted")
        return (X - self.mean) / self.std

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "standard",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "StandardNormalization":
        instance = cls()
        instance.mean = np.array(params["mean"])
        instance.std = np.array(params["std"])
        return instance

    @property
    def name(self) -> str:
        return "standard"


@normalization_registry.register("zca")
class ZCANormalization(BaseNormalization):
    """ZCA whitening with configurable regularization (soft-ZCA)."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.eps = config["eps"]
        self.mean: np.ndarray | None = None
        self.whitening_matrix: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZCANormalization":
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        cov = np.cov(X_centered, rowvar=False)
        U, S, _ = np.linalg.svd(cov)

        S_inv_sqrt = np.diag(1.0 / np.sqrt(S + self.eps))
        self.whitening_matrix = U @ S_inv_sqrt @ U.T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.whitening_matrix is None:
            raise RuntimeError("ZCANormalization not fitted")
        X_centered = X - self.mean
        return X_centered @ self.whitening_matrix.T

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "zca",
            "eps": self.eps,
            "mean": self.mean.tolist(),
            "whitening_matrix": self.whitening_matrix.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "ZCANormalization":
        instance = cls({"eps": params["eps"]})
        instance.mean = np.array(params["mean"])
        instance.whitening_matrix = np.array(params["whitening_matrix"])
        return instance

    @property
    def name(self) -> str:
        return "zca"


@normalization_registry.register("quantile")
class QuantileNormalization(BaseNormalization):
    """Quantile transform to target distribution."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.output_distribution = config["output_distribution"]
        self.n_quantiles = config["n_quantiles"]
        self.quantiles: np.ndarray | None = None
        self.references: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "QuantileNormalization":
        n_samples = X.shape[0]
        n_quantiles = min(self.n_quantiles, n_samples)

        self.quantiles = np.percentile(
            X, np.linspace(0, 100, n_quantiles), axis=0
        )

        if self.output_distribution == "normal":
            self.references = stats.norm.ppf(
                np.linspace(0.001, 0.999, n_quantiles)
            )
        else:
            self.references = np.linspace(0, 1, n_quantiles)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.quantiles is None or self.references is None:
            raise RuntimeError("QuantileNormalization not fitted")

        result = np.zeros_like(X)
        for dim in range(X.shape[1]):
            result[:, dim] = np.interp(
                X[:, dim], self.quantiles[:, dim], self.references
            )
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "quantile",
            "output_distribution": self.output_distribution,
            "n_quantiles": self.n_quantiles,
            "quantiles": self.quantiles.tolist(),
            "references": self.references.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "QuantileNormalization":
        instance = cls({
            "output_distribution": params["output_distribution"],
            "n_quantiles": params["n_quantiles"],
        })
        instance.quantiles = np.array(params["quantiles"])
        instance.references = np.array(params["references"])
        return instance

    @property
    def name(self) -> str:
        return "quantile"


@normalization_registry.register("isotropic")
class IsotropicNormalization(BaseNormalization):
    """Isotropic scaling - scale all vectors to target L2 norm."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.target_l2_norm = config["target_l2_norm"]
        self._fitted = False

    def fit(self, X: np.ndarray) -> "IsotropicNormalization":
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        current_l2 = np.linalg.norm(X, axis=1, keepdims=True)
        current_l2 = np.maximum(current_l2, 1e-8)
        return X * (self.target_l2_norm / current_l2)

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "isotropic",
            "target_l2_norm": self.target_l2_norm,
            "fitted": self._fitted,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "IsotropicNormalization":
        instance = cls({"target_l2_norm": params["target_l2_norm"]})
        instance._fitted = params["fitted"]
        return instance

    @property
    def name(self) -> str:
        return "isotropic"


@normalization_registry.register("winsorize")
class WinsorizeNormalization(BaseNormalization):
    """Winsorization - clip outliers to percentile bounds."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.lower_percentile = config["lower_percentile"]
        self.upper_percentile = config["upper_percentile"]
        self.lower_bounds: np.ndarray | None = None
        self.upper_bounds: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "WinsorizeNormalization":
        self.lower_bounds = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds is None or self.upper_bounds is None:
            raise RuntimeError("WinsorizeNormalization not fitted")
        return np.clip(X, self.lower_bounds, self.upper_bounds)

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "winsorize",
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_bounds": self.lower_bounds.tolist(),
            "upper_bounds": self.upper_bounds.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "WinsorizeNormalization":
        instance = cls({
            "lower_percentile": params["lower_percentile"],
            "upper_percentile": params["upper_percentile"],
        })
        instance.lower_bounds = np.array(params["lower_bounds"])
        instance.upper_bounds = np.array(params["upper_bounds"])
        return instance

    @property
    def name(self) -> str:
        return "winsorize"


class NormalizationPipeline:
    """
    Chainable normalization pipeline.

    Applies normalizations in the order specified in config.
    All parameters are saved for inference-time application.
    """

    def __init__(self, config: dict):
        """
        Initialize pipeline from config.

        Args:
            config: Full config dict with processing.normalization section
        """
        norm_config = config["processing"]["normalization"]
        pipeline_str = norm_config["pipeline"].strip()

        self.normalization_names: list[str] = []
        self.normalizations: list[BaseNormalization] = []

        if pipeline_str:
            names = [n.strip() for n in pipeline_str.split(",")]
            for name in names:
                if not name:
                    continue
                self.normalization_names.append(name)
                norm_class = normalization_registry.get(name)
                norm_instance = norm_class(norm_config.get(name, {}))
                self.normalizations.append(norm_instance)

        self._fitted = False

        if self.normalization_names:
            logger.info(f"Normalization pipeline: {' -> '.join(self.normalization_names)}")
        else:
            logger.info("Normalization pipeline: none (raw embeddings)")

    def fit(self, X: np.ndarray) -> "NormalizationPipeline":
        """Fit all normalizations sequentially."""
        current = X
        for norm in self.normalizations:
            norm.fit(current)
            current = norm.transform(current)
            logger.info(f"Fitted {norm.name}: shape={current.shape}, std={current.std():.6f}")

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all normalizations in order."""
        if not self._fitted and self.normalizations:
            raise RuntimeError("NormalizationPipeline not fitted")

        current = X
        for norm in self.normalizations:
            current = norm.transform(current)

        # Ensure consistent float32 output (matches encoder output dtype)
        return current.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X)
        return self.transform(X)

    def get_params(self) -> dict[str, Any]:
        """Get all parameters for persistence."""
        return {
            "pipeline": self.normalization_names,
            "normalizations": [norm.get_params() for norm in self.normalizations],
            "fitted": self._fitted,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "NormalizationPipeline":
        """Reconstruct pipeline from saved parameters."""
        instance = object.__new__(cls)
        instance.normalization_names = params["pipeline"]
        instance.normalizations = []
        instance._fitted = params["fitted"]

        for norm_params in params["normalizations"]:
            norm_type = norm_params["type"]
            norm_class = normalization_registry.get(norm_type)
            norm_instance = norm_class.from_params(norm_params)
            instance.normalizations.append(norm_instance)

        return instance

    @property
    def is_empty(self) -> bool:
        """Check if pipeline has no normalizations."""
        return len(self.normalizations) == 0
