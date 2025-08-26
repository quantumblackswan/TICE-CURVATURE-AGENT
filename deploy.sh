#!/bin/bash

# TICE Curvature Agent Deployment Script
# Usage: PORT=8000 TICE_SECRET_KEY=mysecret ./deploy.sh

set -e

echo "🧠 TICE Curvature Agent Deployment Starting..."

# Set default values if not provided
export PORT=${PORT:-8000}
export TICE_SECRET_KEY=${TICE_SECRET_KEY:-"ticepluginsecretkey"}

echo "📋 Configuration:"
echo "  Port: $PORT"
echo "  Secret Key: [REDACTED]"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Docker detected, building container..."
    
    # Build Docker image
    docker build -t tice-curvature-agent .
    
    # Stop existing container if running
    docker stop tice-agent 2>/dev/null || true
    docker rm tice-agent 2>/dev/null || true
    
    # Run container
    echo "🚀 Starting TICE Agent container..."
    docker run -d \
        --name tice-agent \
        -p $PORT:8000 \
        -e TICE_SECRET_KEY="$TICE_SECRET_KEY" \
        --restart unless-stopped \
        tice-curvature-agent
    
    echo "✅ TICE Agent deployed via Docker on port $PORT"
    echo "🌐 API available at: http://localhost:$PORT"
    
else
    echo "🐍 Docker not available, using local Python..."
    
    # Install dependencies if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
    fi
    
    # Run the API directly
    echo "🚀 Starting TICE Agent API..."
    export TICE_SECRET_KEY="$TICE_SECRET_KEY"
    python "TICE plug newest.py" &
    
    API_PID=$!
    echo "✅ TICE Agent deployed with PID $API_PID on port $PORT"
    echo "🌐 API available at: http://localhost:$PORT"
    echo "⏹️  To stop: kill $API_PID"
fi

echo "🧠 ΞΛ Quantum Sentinel deployed successfully."