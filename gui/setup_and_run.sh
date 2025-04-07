#!/bin/bash

# Setup and Run Script for ACWMF GUI
# This script installs all dependencies and runs the backend and frontend

# Set base directory to script location
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$BASE_DIR")"

# Print header
echo "====================================================="
echo "  Adaptive Compressed World Model Framework (ACWMF)  "
echo "====================================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python environment
echo "Checking Python environment..."
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Using existing virtual environment."
    source "$PROJECT_ROOT/venv/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }
else
    echo "Creating new virtual environment..."
    python -m venv "$PROJECT_ROOT/venv" || { echo "Failed to create virtual environment"; exit 1; }
    source "$PROJECT_ROOT/venv/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r "$PROJECT_ROOT/requirements.txt"

# Check if npm is installed
if ! command_exists npm; then
    echo "npm is not installed. Please install Node.js and npm first."
    echo "On Ubuntu/Pop!_OS: sudo apt install nodejs npm"
    exit 1
fi

# Install npm dependencies
echo "Installing npm dependencies for frontend..."
cd "$BASE_DIR/frontend" || { echo "Failed to navigate to frontend directory"; exit 1; }
npm install

# Build the frontend
echo "Building the frontend..."
npm run build

# Run the server
echo "Starting the ACWMF server..."
cd "$BASE_DIR" || { echo "Failed to navigate to GUI directory"; exit 1; }
python app.py

# The script will exit when the server is stopped
