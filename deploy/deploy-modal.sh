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
echo "Deploying embedding app (T4 GPU)..."
"$MODAL" deploy cloud/modal_apps/embedding/app.py

echo ""
echo "Deploying VAE app (A10G GPU)..."
"$MODAL" deploy cloud/modal_apps/vae/app.py

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Your Modal apps are now live and will scale to zero when idle."
echo "View them at: https://modal.com/apps"
