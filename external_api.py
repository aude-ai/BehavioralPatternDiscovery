"""
External API for BehavioralPatternDiscovery

This API is designed for integration with the larger evaluation system.
It provides:
1. NDJSON data ingestion
2. Pattern export for category mapping
3. Message search by pattern
4. Model metadata

Run with: uvicorn external_api:app --host 0.0.0.0 --port 8001
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yaml

from src.data.collection.ndjson_loader import NDJSONLoader
from src.external.pattern_export import PatternExporter
from src.external.message_search import MessageSearcher
from src.external.schemas import (
    PatternExport,
    PatternListResponse,
    MessageSearchResponse,
    ModelMetadata,
    EncoderMetadata,
    NDJSONIngestRequest,
    IngestResponse,
    UserPatternMessagesRequest,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config() -> dict:
    """Load and merge configuration files."""
    config = {}
    config_dir = Path("config")

    for config_file in ["data.yaml", "model.yaml", "external.yaml"]:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                config.update(file_config)

    return config


config = load_config()
DATA_DIR = Path("data")

# Initialize components
pattern_exporter = PatternExporter(DATA_DIR)
message_searcher = MessageSearcher(DATA_DIR)


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="BehavioralPatternDiscovery External API",
    description="External integration API for pattern export and message search",
    version="1.0.0",
    docs_url="/external/docs",
    redoc_url="/external/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Metadata Endpoints
# =============================================================================

@app.get("/external/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "BehavioralPatternDiscovery External API"
    }


@app.get("/external/model/metadata", response_model=ModelMetadata)
def get_model_metadata():
    """
    Get metadata about the trained VAE model.

    Returns information about:
    - Model version
    - Architecture (encoders, levels, dimensions)
    - Training stats
    - Pattern counts
    """
    # Load model config
    model_config_path = Path("config/model.yaml")
    if not model_config_path.exists():
        raise HTTPException(status_code=404, detail="Model config not found")

    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Build encoder metadata
    encoders = []
    patterns_per_level = {}
    total_patterns = 0

    encoder_configs = model_config.get("encoders", {})
    for enc_name, enc_config in encoder_configs.items():
        levels = []
        dims_per_level = {}

        hierarchy = enc_config.get("hierarchy", {})
        for level_name, level_config in hierarchy.items():
            levels.append(level_name)
            dim = level_config.get("latent_dim", 0)
            dims_per_level[level_name] = dim

            level_key = f"{enc_name}_{level_name}"
            patterns_per_level[level_key] = dim
            total_patterns += dim

        encoders.append(EncoderMetadata(
            name=enc_name,
            levels=levels,
            dimensions_per_level=dims_per_level,
        ))

    # Add unified dimension
    unified_dim = model_config.get("unification", {}).get("unified_dim", 0)
    patterns_per_level["unified"] = unified_dim
    total_patterns += unified_dim

    return ModelMetadata(
        model_version_id=pattern_exporter._get_model_version(),
        trained_at=None,
        num_encoders=len(encoders),
        encoders=encoders,
        unified_dimension=unified_dim,
        num_training_messages=0,
        num_engineers=0,
        total_patterns=total_patterns,
        patterns_per_level=patterns_per_level,
    )


# =============================================================================
# Data Ingestion Endpoints
# =============================================================================

@app.post("/external/ingest/ndjson", response_model=IngestResponse)
def ingest_ndjson(request: NDJSONIngestRequest):
    """
    Ingest activities from NDJSON file(s).

    This converts data from the unified data loader format into
    the format used by this project's preprocessing pipeline.
    """
    try:
        loader_config = {
            "input_path": request.input_path,
            "excluded_sections": config["collection"]["ndjson"]["excluded_sections"],
            "engineer_id_mapping": request.engineer_id_mapping_path,
            "parser_configs": config["collection"]["parsers"],
            "text_cleanup": config["collection"]["text_cleanup"],
        }

        loader = NDJSONLoader(loader_config)

        if request.window_start and request.window_end:
            df = loader.load_windowed(request.window_start, request.window_end)
        else:
            df = loader.load()

        # Save to activities.csv format
        output_path = DATA_DIR / "data" / "collection" / "activities.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        # Compute stats
        sources = df["source"].value_counts().to_dict()

        logger.info(f"Ingested {len(df)} activities from NDJSON to {output_path}")

        return IngestResponse(
            status="success",
            activities_loaded=len(df),
            unique_engineers=df["engineer_id"].nunique(),
            sources=sources,
            output_path=str(output_path),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("NDJSON ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Pattern Export Endpoints
# =============================================================================

@app.get("/external/patterns", response_model=PatternListResponse)
def get_all_patterns(
    top_k_examples: int = Query(5, ge=1, le=20, description="Examples per pattern"),
    level: Optional[str] = Query(None, description="Filter by level: bottom, mid, top, unified"),
    encoder: Optional[str] = Query(None, description="Filter by encoder name"),
):
    """
    Export all discovered patterns.

    Returns patterns in a format suitable for:
    1. Computing semantic similarity to CTO-defined categories
    2. Human review for sign (+/-) assignment
    3. Building the cluster -> category mapping
    """
    try:
        export = pattern_exporter.export_all_patterns(top_k_examples)

        # Filter by level if specified
        if level:
            export.patterns = [p for p in export.patterns if p.level == level]

        # Filter by encoder if specified
        if encoder:
            export.patterns = [p for p in export.patterns if p.encoder == encoder]

        export.pattern_count = len(export.patterns)

        return export

    except Exception as e:
        logger.exception("Pattern export failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/external/patterns/{pattern_id}", response_model=PatternExport)
def get_pattern(
    pattern_id: str,
    top_k_examples: int = Query(10, ge=1, le=50),
):
    """
    Get a single pattern by ID.

    Pattern IDs are formatted as:
    - "unified_0", "unified_1", ... for unified patterns
    - "enc1_bottom_0", "enc1_mid_5", ... for encoder-specific patterns
    """
    pattern = pattern_exporter.export_pattern(pattern_id, top_k_examples)
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"Pattern not found: {pattern_id}")
    return pattern


# =============================================================================
# Message Search Endpoints
# =============================================================================

@app.get("/external/patterns/{pattern_id}/messages", response_model=MessageSearchResponse)
def get_pattern_messages(
    pattern_id: str,
    top_k: int = Query(20, ge=1, le=100),
    engineer_id: Optional[str] = Query(None, description="Filter to specific engineer"),
):
    """
    Get messages that activate a specific pattern.

    Useful for:
    1. Understanding what a pattern captures
    2. Generating evidence snippets for evaluations
    3. Semantic similarity computation
    """
    try:
        return message_searcher.search_by_pattern(
            pattern_id=pattern_id,
            top_k=top_k,
            engineer_id=engineer_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Message search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/external/messages/user-pattern", response_model=MessageSearchResponse)
def get_user_pattern_messages(request: UserPatternMessagesRequest):
    """
    Get a specific user's messages for a pattern.

    This is specifically for generating evidence snippets
    in individual evaluations.
    """
    try:
        return message_searcher.get_user_pattern_messages(
            engineer_id=request.engineer_id,
            pattern_id=request.pattern_id,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("User pattern message search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/external/messages/{message_id}")
def get_message(message_id: str):
    """Get full details for a specific message."""
    message = message_searcher.get_message_by_id(message_id)
    if message is None:
        raise HTTPException(status_code=404, detail=f"Message not found: {message_id}")
    return message


@app.get("/external/patterns/list/ids")
def list_pattern_ids():
    """Get list of all available pattern IDs."""
    try:
        pattern_ids = message_searcher.get_all_patterns()
        return {
            "count": len(pattern_ids),
            "pattern_ids": pattern_ids
        }
    except Exception as e:
        logger.exception("Failed to list pattern IDs")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8001)

    logger.info(f"Starting External API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
