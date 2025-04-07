#!/bin/bash

# Adaptive Compressed World Model Framework (ACWMF) Startup Script
# This script initializes the environment and starts the application

# Set base directory to script location
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GUI_DIR="$BASE_DIR/gui"

# Print header
echo "====================================================="
echo "  Adaptive Compressed World Model Framework (ACWMF)  "
echo "====================================================="
echo ""

# Activate virtual environment
if [ -d "$BASE_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$BASE_DIR/venv/bin/activate" || { 
        echo "Failed to activate virtual environment"; 
        echo "Try running: python -m venv venv"; 
        exit 1; 
    }
else
    echo "Creating new virtual environment..."
    python -m venv "$BASE_DIR/venv" || { 
        echo "Failed to create virtual environment"; 
        exit 1; 
    }
    source "$BASE_DIR/venv/bin/activate" || { 
        echo "Failed to activate virtual environment"; 
        exit 1; 
    }
    
    # Install pip, setuptools and wheel
    echo "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    echo "Installing requirements..."
    pip install -r "$BASE_DIR/requirements.txt"
fi

# Check if GUI should be started
if [ -d "$GUI_DIR" ]; then
    echo "Starting the ACWMF GUI..."
    cd "$GUI_DIR" || { 
        echo "Failed to navigate to GUI directory"; 
        exit 1; 
    }
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        echo "Warning: npm is not installed. The frontend will not be built automatically."
        echo "On Ubuntu/Pop!_OS: sudo apt install nodejs npm"
    else
        # Build the frontend if it doesn't exist
        if [ ! -d "$GUI_DIR/frontend/build" ]; then
            echo "Building the frontend..."
            cd "$GUI_DIR/frontend" || { 
                echo "Failed to navigate to frontend directory"; 
                exit 1; 
            }
            npm install && npm run build || {
                echo "Failed to build the frontend";
                echo "Try running: cd $GUI_DIR/frontend && npm install && npm run build";
            }
            cd "$GUI_DIR" || { 
                echo "Failed to navigate to GUI directory"; 
                exit 1; 
            }
        fi
    fi
    
    # Start the server
    echo "Starting the Flask server..."
    python app.py
else
    echo "GUI directory not found at $GUI_DIR"
    exit 1
fi
