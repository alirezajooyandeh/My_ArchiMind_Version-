#!/usr/bin/env python3
"""Simple script to run the application."""
import uvicorn
from backend.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,
    )

