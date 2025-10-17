#!/bin/bash

# AI Humanizer Startup Script

echo "Starting AI Humanizer..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Docker is installed (optional)
if command -v docker &> /dev/null; then
    echo "Docker detected. Starting with Docker Compose..."
    docker-compose up -d
else
    echo "Docker not found. Starting with Python directly..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Download NLP models
    echo "Downloading NLP models..."
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
    
    # Start the application
    echo "Starting AI Humanizer server..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi