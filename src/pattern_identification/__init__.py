# Pattern identification components for BehavioralPatternDiscovery

from .batch_scorer import BatchScorer
from .message_assigner import MessageAssigner, PatternExamples
from .word_attributor import WordAttributor
from .shap_analysis import SHAPAnalyzer, LevelTransitionWrapper, FinalLevelToUnifiedWrapper
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
    # Batch Scoring
    "BatchScorer",
    # Message Assignment
    "MessageAssigner",
    "PatternExamples",
    # Word Attribution
    "WordAttributor",
    # SHAP Analysis
    "SHAPAnalyzer",
    "LevelTransitionWrapper",
    "FinalLevelToUnifiedWrapper",
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
