#!/bin/bash

# Start the Next.js version of the Adaptive Compressed World Model Framework

# Set stricter error handling
set -e

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run the enhanced cleanup script first
echo "Running thorough cleanup to ensure no Flask processes are running..."
chmod +x "$DIR/cleanup_flask.sh"
"$DIR/cleanup_flask.sh"

# Activate virtual environment if it exists
if [ -d "$DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$DIR/venv/bin/activate"
else
    echo "Warning: Virtual environment not found at $DIR/venv"
    echo "Continuing without virtual environment activation..."
fi

# Install updated dependencies with explicit versions for better compatibility
echo "Installing backend requirements with updated versions..."
pip install flask==2.3.3 flask-cors==4.0.0 numpy==1.26.4 networkx==3.2.1 redis==5.0.1 \
           scikit-learn==1.4.0 scipy==1.12.0 matplotlib==3.8.2 pandas==2.2.0 \
           PyPDF2==3.0.1 python-docx==0.8.11 openpyxl==3.1.2 pdfminer.six==20221105 \
           pdfplumber==0.10.3

# Only install torch if it's not already installed (to avoid long installation times)
if ! pip freeze | grep -q torch; then
    echo "Installing PyTorch..."
    pip install torch==2.1.2
fi

# Add the source directory to PYTHONPATH
export PYTHONPATH="$DIR:$PYTHONPATH"

# Create and ensure permissions for data directory
DATA_DIR="$DIR/data/knowledge_gui"
mkdir -p "$DATA_DIR"
chmod -R 755 "$DATA_DIR"

# Use a more robust PID management approach
PID_FILE="/tmp/acwmf_flask.pid"
LOCK_FILE="/tmp/acwmf_flask.lock"

# Create a lock file to ensure exclusive access
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another instance is already running the start script. Please run cleanup first."
    exit 1
fi

# Start Flask backend with explicit environment variables
echo "Starting Flask backend server..."
FLASK_ENV=development FLASK_DEBUG=0 python "$DIR/gui/app.py" &
FLASK_PID=$!

# Save PID to file for future cleanup
echo $FLASK_PID > "$PID_FILE"
echo "Flask server started with PID: $FLASK_PID"

# Give Flask a moment to start and verify it's running
sleep 3
if ! ps -p $FLASK_PID > /dev/null; then
    echo "ERROR: Flask server failed to start or exited immediately!"
    rm -f "$PID_FILE"
    exit 1
fi

# Check if port 5000 is being used by our process
if ! lsof -i:5000 -sTCP:LISTEN | grep -q $FLASK_PID; then
    echo "WARNING: Flask server started but is not listening on port 5000!"
    echo "Another process might be using that port."
    ps -p $FLASK_PID -f
    lsof -i:5000
fi

echo "-------------------------------------------"
echo "IMPORTANT: Flask backend runs on port 5000, but you should access the app via:"
echo "           http://localhost:3000"
echo "-------------------------------------------"

# Start Next.js frontend
echo "Starting Next.js frontend..."
cd "$DIR/gui/nextjs-frontend"

# Check if Node.js dependencies are installed, install if not
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Define a robust cleanup function
cleanup() {
    echo "Performing thorough cleanup before exit..."
    
    # Kill the Flask process we started
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            echo "Stopping Flask server (PID: $PID)..."
            kill -15 $PID 2>/dev/null  # Try graceful shutdown first
            sleep 1
            # If still running, force kill
            if ps -p $PID > /dev/null; then
                echo "Force killing Flask server..."
                kill -9 $PID 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi
    
    # Run the cleanup script to catch any processes we missed
    "$DIR/cleanup_flask.sh"
    
    # Release the lock file
    flock -u 9
    rm -f "$LOCK_FILE"
    
    echo "All services stopped."
    exit 0
}

# Register cleanup for multiple signals
trap cleanup SIGINT SIGTERM EXIT

# Start the Next.js development server
echo "Starting Next.js development server..."
npm run dev

# Wait for the frontend to exit
wait
