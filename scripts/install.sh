#!/bin/bash
# Installation script for Adaptive Compressed World Model Framework

set -e  # Exit immediately if any command fails

echo "=========================================================="
echo "Adaptive Compressed World Model Framework Installation"
echo "=========================================================="

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root directory: $PROJECT_ROOT"

# Create and activate virtual environment
echo -e "\n[1/5] Setting up virtual environment..."
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    python -m venv "$PROJECT_ROOT/venv"
fi

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Update pip
echo -e "\n[2/5] Updating pip..."
pip install --upgrade pip

# Install dependencies
echo -e "\n[3/5] Installing dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# Install the package in development mode
echo -e "\n[4/5] Installing package in development mode..."
pip install -e "$PROJECT_ROOT"

# Setup Ollama (optional)
echo -e "\n[5/5] Setting up Ollama..."
python "$PROJECT_ROOT/scripts/setup_ollama.py"

echo -e "\nInstallation complete!"
echo "To activate the environment, run:"
echo "  source $PROJECT_ROOT/venv/bin/activate"
echo ""
echo "To run the knowledge system demo:"
echo "  python $PROJECT_ROOT/examples/knowledge_system_demo.py"
echo ""
echo "Enjoy using the Adaptive Compressed World Model Framework!"
