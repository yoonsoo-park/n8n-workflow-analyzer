#!/usr/bin/env bash
# n8n Workflow Analyzer Deployment Script
#
# This script deploys the n8n Workflow Analyzer in a local environment.
# It sets up the Python virtual environment, installs dependencies,
# and starts the API server and web interface.
#
# Usage:
#   ./deploy.sh [options]
#
# Options:
#   --help          Show this help message and exit
#   --dev           Deploy in development mode
#   --prod          Deploy in production mode (default)
#   --api-only      Deploy only the API server
#   --web-only      Deploy only the web interface
#   --port PORT     Specify the API server port (default: 5000)
#   --web-port PORT Specify the web interface port (default: 5173)

set -e

# Default options
MODE="prod"
DEPLOY_API=true
DEPLOY_WEB=true
API_PORT=5000
WEB_PORT=5173

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      echo "n8n Workflow Analyzer Deployment Script"
      echo ""
      echo "Usage:"
      echo "  ./deploy.sh [options]"
      echo ""
      echo "Options:"
      echo "  --help          Show this help message and exit"
      echo "  --dev           Deploy in development mode"
      echo "  --prod          Deploy in production mode (default)"
      echo "  --api-only      Deploy only the API server"
      echo "  --web-only      Deploy only the web interface"
      echo "  --port PORT     Specify the API server port (default: 5000)"
      echo "  --web-port PORT Specify the web interface port (default: 5173)"
      exit 0
      ;;
    --dev)
      MODE="dev"
      shift
      ;;
    --prod)
      MODE="prod"
      shift
      ;;
    --api-only)
      DEPLOY_API=true
      DEPLOY_WEB=false
      shift
      ;;
    --web-only)
      DEPLOY_API=false
      DEPLOY_WEB=true
      shift
      ;;
    --port)
      API_PORT="$2"
      shift
      shift
      ;;
    --web-port)
      WEB_PORT="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists python3; then
  echo "Error: python3 is required but not installed."
  exit 1
fi

if ! command_exists pip3; then
  echo "Error: pip3 is required but not installed."
  exit 1
fi

if $DEPLOY_WEB && ! command_exists node; then
  echo "Error: node is required but not installed."
  exit 1
fi

if $DEPLOY_WEB && ! command_exists npm; then
  echo "Error: npm is required but not installed."
  exit 1
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "Activating Python virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Deploy API server
if $DEPLOY_API; then
  echo "Deploying API server..."
  
  # Install API server dependencies
  cd api_server
  pip install -r requirements.txt
  
  # Start API server
  if [ "$MODE" = "dev" ]; then
    echo "Starting API server in development mode on port $API_PORT..."
    export FLASK_ENV=development
    export FLASK_APP=src/main.py
    export FLASK_RUN_PORT=$API_PORT
    python -m flask run --host=0.0.0.0 &
  else
    echo "Starting API server in production mode on port $API_PORT..."
    export FLASK_ENV=production
    export FLASK_APP=src/main.py
    
    # Check if gunicorn is installed
    if ! command_exists gunicorn; then
      echo "Installing gunicorn..."
      pip install gunicorn
    fi
    
    # Start API server with gunicorn
    gunicorn -w 4 -b 0.0.0.0:$API_PORT src.main:app &
  fi
  
  cd ..
  echo "API server started on http://localhost:$API_PORT"
fi

# Deploy web interface
if $DEPLOY_WEB; then
  echo "Deploying web interface..."
  
  # Install web interface dependencies
  cd web_ui
  npm install
  
  # Start web interface
  if [ "$MODE" = "dev" ]; then
    echo "Starting web interface in development mode on port $WEB_PORT..."
    export VITE_API_URL=http://localhost:$API_PORT
    npm run dev -- --port $WEB_PORT &
  else
    echo "Building web interface for production..."
    export VITE_API_URL=http://localhost:$API_PORT
    npm run build
    
    # Check if serve is installed
    if ! command_exists serve; then
      echo "Installing serve..."
      npm install -g serve
    fi
    
    # Start web interface with serve
    echo "Starting web interface in production mode on port $WEB_PORT..."
    serve -s dist -l $WEB_PORT &
  fi
  
  cd ..
  echo "Web interface started on http://localhost:$WEB_PORT"
fi

echo "Deployment complete!"
echo ""
if $DEPLOY_API; then
  echo "API server: http://localhost:$API_PORT"
fi
if $DEPLOY_WEB; then
  echo "Web interface: http://localhost:$WEB_PORT"
fi
echo ""
echo "Press Ctrl+C to stop the servers."

# Wait for user to press Ctrl+C
wait

