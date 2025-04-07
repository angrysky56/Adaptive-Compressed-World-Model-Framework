#!/bin/bash

# Enhanced script to find and kill ALL running Flask processes without exceptions

echo "Performing thorough Flask backend process cleanup..."
echo "NOTE: This script stops the Flask backend on port 5000, not the Next.js frontend on port 3000"

# First, try lsof to check what's using port 5000
PORT_PIDS=$(lsof -ti:5000 2>/dev/null)
if [ ! -z "$PORT_PIDS" ]; then
    echo "Found processes using port 5000: $PORT_PIDS"
    for pid in $PORT_PIDS; do
        echo "Killing process $pid using port 5000..."
        kill -9 $pid 2>/dev/null
    done
fi

# Find any Python processes that contain "app.py" in their command line
APP_PIDS=$(ps -ef | grep -E 'python.*app\.py' | grep -v grep | awk '{print $2}')
if [ ! -z "$APP_PIDS" ]; then
    echo "Found Flask app.py processes: $APP_PIDS"
    for pid in $APP_PIDS; do
        echo "Killing app.py process $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

# Find any Python processes that might be Flask-related
FLASK_PIDS=$(ps -ef | grep -E 'python.*flask' | grep -v grep | awk '{print $2}')
if [ ! -z "$FLASK_PIDS" ]; then
    echo "Found Flask-related Python processes: $FLASK_PIDS"
    for pid in $FLASK_PIDS; do
        echo "Killing Flask process $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

# Remove PID file
PID_FILE="/tmp/acwmf_flask.pid"
if [ -f "$PID_FILE" ]; then
    echo "Removing PID file..."
    rm -f "$PID_FILE"
fi

# Double-check port 5000 is free
sleep 1
PORT_CHECK=$(lsof -i:5000 2>/dev/null)
if [ ! -z "$PORT_CHECK" ]; then
    echo "WARNING: Port 5000 is still in use after cleanup!"
    echo "$PORT_CHECK"
else
    echo "Port 5000 is free."
fi

# Final check for any lingering Python processes matching our app
FINAL_CHECK=$(ps -ef | grep -E 'python.*(/gui/app\.py|flask)' | grep -v grep)
if [ ! -z "$FINAL_CHECK" ]; then
    echo "WARNING: Some Flask processes may still be running:"
    echo "$FINAL_CHECK"
else
    echo "No Flask processes detected."
fi

echo "Cleanup complete!"
