#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI backend on port 8000..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for the backend to start
sleep 3

# Start Streamlit frontend in the foreground
echo "Starting Streamlit frontend on port 8501..."
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
