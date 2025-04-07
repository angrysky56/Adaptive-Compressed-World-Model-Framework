#!/bin/bash

# Set base directory to script location
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GUI_DIR="$BASE_DIR/gui"
FRONTEND_DIR="$GUI_DIR/frontend"

# Print header
echo "====================================================="
echo "  Rebuilding ACWMF Frontend  "
echo "====================================================="
echo ""

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Cannot rebuild frontend."
    echo "On Ubuntu/Pop!_OS: sudo apt install nodejs npm"
    exit 1
fi

# Navigate to frontend directory
cd "$FRONTEND_DIR" || { 
    echo "Failed to navigate to frontend directory"; 
    exit 1; 
}

# Clean build directory if it exists
if [ -d "$FRONTEND_DIR/build" ]; then
    echo "Cleaning existing build..."
    rm -rf "$FRONTEND_DIR/build"
fi

# Install dependencies and rebuild
echo "Installing npm dependencies..."
npm install

echo "Building frontend..."
npm run build

echo "Frontend rebuild complete!"
echo "Now run ./start.sh to start the server with the updated frontend."
