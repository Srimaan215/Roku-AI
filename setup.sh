#!/bin/bash
# Project Roku - Setup Script
# ===========================

set -e

echo "🤖 Setting up Project Roku..."
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python 3.11 if not present
if ! command -v python3.11 &> /dev/null; then
    echo "🐍 Installing Python 3.11..."
    brew install python@3.11
fi

# Install build tools
echo "🔧 Installing build tools..."
brew install cmake portaudio

# Create virtual environment
echo "📁 Creating virtual environment..."
python3.11 -m venv ~/roku-env

# Activate and install dependencies
echo "📥 Installing Python dependencies..."
source ~/roku-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📂 Creating project directories..."
mkdir -p models/base
mkdir -p models/lora
mkdir -p data/conversations
mkdir -p data/context
mkdir -p data/smart_home
mkdir -p data/user_profile
mkdir -p data/training

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source ~/roku-env/bin/activate"
echo "  2. Download model (see techstack.md for instructions)"
echo "  3. Run CLI: python interfaces/cli.py"
echo ""
