# Discriminator implementations

from .tc import TCDiscriminator, discriminator_registry
from .pc import PCDiscriminator
from .diversity import DiversityDiscriminator, DiversityDiscriminatorLoss

__all__ = [
    "TCDiscriminator",
    "PCDiscriminator",
    "DiversityDiscriminator",
    "DiversityDiscriminatorLoss",
    "discriminator_registry",
]
