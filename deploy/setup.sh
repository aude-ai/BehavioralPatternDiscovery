#!/bin/bash
# Setup script for BehavioralPatternDiscovery cloud deployment
# Run this from the behavioral-pattern-discovery directory on your Hetzner server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"
MODAL="$VENV_DIR/bin/modal"
PIP="$VENV_DIR/bin/pip"

echo "=== BPD Cloud Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install modal
echo "Installing Modal CLI..."
"$PIP" install --upgrade pip
"$PIP" install modal

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Modal CLI installed at: $MODAL"
echo ""
echo "Next steps:"
echo ""
echo "1. Authenticate with Modal (one-time, opens browser):"
echo "   $MODAL token new"
echo ""
echo "2. Create Modal secret:"
echo "   $MODAL secret create hetzner-internal-key \\"
echo "       HETZNER_INTERNAL_KEY=your-secret-key \\"
echo "       HETZNER_BASE_URL=https://yourdomain.com/bpd"
echo ""
echo "3. Deploy Modal apps:"
echo "   ./deploy/deploy-modal.sh"
echo ""
echo "4. Start CPU server:"
echo "   cp deploy/.env.example .env"
echo "   # Edit .env with your keys"
echo "   docker-compose -f deploy/docker-compose.yml up -d"
