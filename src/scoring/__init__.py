"""
Scoring Module

Individual engineer scoring and LLM-based report generation.

GPU modules (IndividualScorer) must be imported directly from submodules:
    from src.scoring.individual_scorer import IndividualScorer

CPU-safe modules can be imported from this package:
    from src.scoring import ReportGenerator, ExplanationGenerator
"""

from .report_generator import ReportGenerator
from .explanation_generator import ExplanationGenerator

__all__ = [
    "ReportGenerator",
    "ExplanationGenerator",
]
