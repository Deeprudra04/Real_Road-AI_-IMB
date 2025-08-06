#!/bin/bash

echo "Rice Leaf Disease Detection - Quick Start"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import flask, tensorflow, numpy" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Run the quick start script
echo
echo "Starting application..."
python3 run_app.py 