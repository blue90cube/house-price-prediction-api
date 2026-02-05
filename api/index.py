"""
Vercel Serverless Entry Point for House Price Prediction API.

This module serves as the entry point for Vercel's Python serverless functions.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app
from app.main import app

# Vercel handler
handler = app
