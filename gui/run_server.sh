#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
fi

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Run the server
echo "Starting the server..."
python app.py
