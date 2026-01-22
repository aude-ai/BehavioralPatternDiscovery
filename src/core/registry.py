"""
Component Registry System

Provides a unified way to register and retrieve swappable components.
All major components (encoders, decoders, distributions, etc.) use this
registry pattern to enable configuration-driven component selection.
"""

import logging
from typing import TypeVar, Generic, Callable, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """
    Registry for swappable components.

    Each component type (encoder, decoder, distribution, etc.) has its own
    registry instance. Components are registered with a string name and
    can be retrieved or instantiated by that name.

    Example:
        # Create a registry for encoders
        encoder_registry = ComponentRegistry[BaseEncoder]("encoder")

        # Register an implementation
        @encoder_registry.register("hierarchical")
        class HierarchicalEncoder(BaseEncoder):
            ...

        # Create an instance from config
        encoder = encoder_registry.create("hierarchical", config=encoder_config)
    """

    def __init__(self, component_type: str):
        """
        Initialize registry for a component type.

        Args:
            component_type: Human-readable name for error messages (e.g., "encoder")
        """
        self._component_type = component_type
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a component class.

        Args:
            name: String identifier for this implementation

        Returns:
            Decorator function that registers the class

        Example:
            @encoder_registry.register("hierarchical")
            class HierarchicalEncoder(BaseEncoder):
                ...
        """
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(
                    f"[{self._component_type}] Name '{name}' already registered "
                    f"to {self._registry[name].__name__}. Cannot register {cls.__name__}."
                )
            self._registry[name] = cls
            logger.debug(f"[{self._component_type}] Registered '{name}' -> {cls.__name__}")
            return cls
        return decorator

    def register_class(self, name: str, cls: type[T]) -> None:
        """
        Register a component class directly (non-decorator form).

        Useful for registering existing classes like PyTorch built-ins.

        Args:
            name: String identifier for this implementation
            cls: The class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self._registry:
            raise ValueError(
                f"[{self._component_type}] Name '{name}' already registered "
                f"to {self._registry[name].__name__}. Cannot register {cls.__name__}."
            )
        self._registry[name] = cls
        logger.debug(f"[{self._component_type}] Registered '{name}' -> {cls.__name__}")

    def get(self, name: str) -> type[T]:
        """
        Get a component class by name.

        Args:
            name: Registered name of the component

        Returns:
            The component class

        Raises:
            KeyError: If name is not registered
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"[{self._component_type}] Unknown type: '{name}'. "
                f"Available: {available}"
            )
        return self._registry[name]

    def create(self, name: str, **kwargs: Any) -> T:
        """
        Create a component instance from config.

        Args:
            name: Registered name of the component
            **kwargs: Arguments passed to component constructor

        Returns:
            Instantiated component

        Raises:
            KeyError: If name is not registered
        """
        cls = self.get(name)
        return cls(**kwargs)

    def list_registered(self) -> list[str]:
        """
        List all registered component names.

        Returns:
            List of registered names
        """
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._registry)
