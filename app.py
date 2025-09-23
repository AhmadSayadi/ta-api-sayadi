# Alias file for Render deployment compatibility
# This file imports the FastAPI app from main.py

from main import app

# Export the app for uvicorn
__all__ = ['app']