# Model components for BehavioralPatternDiscovery

from .base import BaseVAE
from .vae import MultiEncoderVAE, EncoderView

from .layers import (
    activation_registry,
    normalization_registry,
    make_activation,
    make_norm_layer,
    make_layer_block,
    compute_layer_dimensions,
)

from .distributions import (
    distribution_registry,
    GaussianDistribution,
    GammaDistribution,
)

from .encoders import (
    encoder_registry,
    HierarchicalEncoder,
)

from .unification import (
    unification_registry,
    MLPUnification,
)

from .decoders import (
    decoder_registry,
    BaseDecoder,
    DecoderTrainingOutput,
    DecoderValidationOutput,
    FlowMatchingDecoder,
    SphericalFlowMatchingDecoder,
    MLPDecoder,
    ODESolver,
    TimestepSampler,
)

from .discriminators import (
    discriminator_registry,
    TCDiscriminator,
    PCDiscriminator,
)

__all__ = [
    # VAE
    "BaseVAE",
    "MultiEncoderVAE",
    "EncoderView",
    # Layers
    "activation_registry",
    "normalization_registry",
    "make_activation",
    "make_norm_layer",
    "make_layer_block",
    "compute_layer_dimensions",
    # Distributions
    "distribution_registry",
    "GaussianDistribution",
    "GammaDistribution",
    # Encoders
    "encoder_registry",
    "HierarchicalEncoder",
    # Unification
    "unification_registry",
    "MLPUnification",
    # Decoders
    "decoder_registry",
    "BaseDecoder",
    "DecoderTrainingOutput",
    "DecoderValidationOutput",
    "FlowMatchingDecoder",
    "SphericalFlowMatchingDecoder",
    "MLPDecoder",
    "ODESolver",
    "TimestepSampler",
    # Discriminators
    "discriminator_registry",
    "TCDiscriminator",
    "PCDiscriminator",
]
