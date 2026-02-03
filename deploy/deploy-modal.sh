#!/bin/bash
# Deploy Modal apps for BehavioralPatternDiscovery
# Run from the behavioral-pattern-discovery directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"
MODAL="$VENV_DIR/bin/modal"

# Check venv exists
if [ ! -f "$MODAL" ]; then
    echo "Error: Modal not found. Run ./deploy/setup.sh first."
    exit 1
fi

cd "$PROJECT_DIR"

echo "=== Deploying Modal Apps ==="

echo ""
echo "Deploying processing pipeline (A100 GPU)..."
"$MODAL" deploy cloud/modal_apps/processing/app.py

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Your Modal apps are now live and will scale to zero when idle."
echo "View them at: https://modal.com/apps"
echo ""
echo "Deployed functions:"
echo "  - run_processing_pipeline: Full B.1-B.8 pipeline"
echo "  - score_individual: One-off individual scoring"
echo "  - ScoringService: Warm service for individual scoring"
echo "  - get_r2_status: Check R2 file status"
echo "  - delete_project_r2_files: Clean up R2 files"
