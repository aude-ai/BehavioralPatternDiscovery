"""
Data Collection Module

Handles fetching activities from MongoDB, NDJSON, and managing engineer data.
"""

from .mongodb_loader import MongoDBLoader
from .engineer_manager import EngineerManager
from .ndjson_loader import NDJSONLoader, load_activities_from_ndjson
from .text_cleanup import TextCleaner
from .data_source import resolve_data_source

__all__ = [
    "MongoDBLoader",
    "EngineerManager",
    "NDJSONLoader",
    "load_activities_from_ndjson",
    "TextCleaner",
    "resolve_data_source",
]
