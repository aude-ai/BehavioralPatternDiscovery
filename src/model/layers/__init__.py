# Layer components: activations, normalizations, blocks

from .activations import (
    activation_registry,
    make_activation,
    get_swiglu,
    xIELU,
    Cone,
    ParabolicCone,
    ParameterizedCone,
    SwiGLU,
)

from .normalizations import (
    normalization_registry,
    make_norm_layer,
    RMSNorm,
    LayerNorm,
    DynamicTanh,
    ScaledNorm,
)

from .blocks import (
    LinearBlock,
    make_layer_block,
    compute_layer_dimensions,
)

__all__ = [
    # Activations
    "activation_registry",
    "make_activation",
    "get_swiglu",
    "xIELU",
    "Cone",
    "ParabolicCone",
    "ParameterizedCone",
    "SwiGLU",
    # Normalizations
    "normalization_registry",
    "make_norm_layer",
    "RMSNorm",
    "LayerNorm",
    "DynamicTanh",
    "ScaledNorm",
    # Blocks
    "LinearBlock",
    "make_layer_block",
    "compute_layer_dimensions",
]
