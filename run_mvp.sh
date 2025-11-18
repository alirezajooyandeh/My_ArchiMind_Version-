#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PORT=8090

# Parse optional -p <port>
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    *)
      shift 1
      ;;
  esac
done

# Create venv if needed
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  echo "Installing requirements..."
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

# Start uvicorn with backend.main:app
echo "Starting ArchiMind server on port $PORT..."
exec uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" --reload

