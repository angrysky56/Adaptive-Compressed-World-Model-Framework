#!/bin/bash

# Start the Next.js version of the Adaptive Compressed World Model Framework

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Activate virtual environment if it exists
if [ -d "$DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$DIR/venv/bin/activate"
else
    echo "Warning: Virtual environment not found at $DIR/venv"
    echo "Continuing without virtual environment activation..."
fi

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
pip install -r "$DIR/requirements.txt"
pip install -r "$DIR/gui/requirements.txt"

# Skip the editable package installation which is causing errors
echo "Skipping package installation (not required for web interface)..."
# Add the source directory to PYTHONPATH instead
export PYTHONPATH="$DIR:$PYTHONPATH"

# Start Flask backend in the background
echo "Starting Flask backend server..."
python "$DIR/gui/app.py" &
FLASK_PID=$!

# Give Flask a moment to start
sleep 2

# Check if the backend is running
if ps -p $FLASK_PID > /dev/null; then
    echo "Flask backend started successfully (PID: $FLASK_PID)"
else
    echo "Failed to start Flask backend server!"
    exit 1
fi

# Start Next.js frontend
echo "Starting Next.js frontend..."
cd "$DIR/gui/nextjs-frontend"

# Check if Node.js dependencies are installed, install if not
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Start the Next.js development server
echo "Starting Next.js development server..."
npm run dev

# Function to handle script termination
cleanup() {
    echo "Stopping services..."
    kill $FLASK_PID
    exit 0
}

# Register the cleanup function for script termination
trap cleanup SIGINT SIGTERM

# Wait for the frontend to exit
wait
