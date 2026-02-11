# Pattern identification components for BehavioralPatternDiscovery
#
# GPU-dependent modules (BatchScorer, SHAPAnalyzer, WordAttributor) must be
# imported directly from their submodules to avoid torch dependency on CPU-only servers.

from .message_scorer import MessageScorer
from .pattern_naming import PatternNamer
from .population_stats import PopulationStats
from .prompt_builder import PromptBuilder
from .prompt_templates import (
    SYSTEM_CONTEXT,
    DATA_SOURCE_DESCRIPTIONS,
    TEAM_CONTEXT,
    DISENTANGLEMENT_CONTEXT,
    ABSTRACTION_LEVELS,
    LEVEL_CONTEXTS,
    get_abstraction_for_level,
)

__all__ = [
    # Message Scoring
    "MessageScorer",
    # Pattern Naming
    "PatternNamer",
    # Prompt System
    "PromptBuilder",
    "SYSTEM_CONTEXT",
    "DATA_SOURCE_DESCRIPTIONS",
    "TEAM_CONTEXT",
    "DISENTANGLEMENT_CONTEXT",
    "ABSTRACTION_LEVELS",
    "LEVEL_CONTEXTS",
    "get_abstraction_for_level",
    # Population Statistics
    "PopulationStats",
]
