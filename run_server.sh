#!/bin/bash

# Set base directory to script location
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GUI_DIR="$BASE_DIR/gui"

# Activate virtual environment
source "$BASE_DIR/venv/bin/activate" || { 
    echo "Failed to activate virtual environment"; 
    exit 1; 
}

# Start the server
cd "$GUI_DIR" || { 
    echo "Failed to navigate to GUI directory"; 
    exit 1; 
}

echo "Starting the ACWMF Knowledge Graph server..."
python app.py
