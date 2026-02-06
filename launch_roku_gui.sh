#!/bin/bash
# Roku GUI Launcher

cd "$(dirname "$0")"

echo "🤖 Launching Roku AI GUI..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Launch GUI
echo "Starting Roku..."
python interfaces/roku_gui.py
