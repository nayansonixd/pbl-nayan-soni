"""
run_server.py
=============
Simple launcher — run this from the project root to start the API server.

Usage:
    python run_server.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
