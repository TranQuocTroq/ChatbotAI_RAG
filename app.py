"""
DocBrain RAG - FastAPI HTML Web App Entry Point
Serves pure Modern HTML/CSS Web App interface (FastAPI).
Completely independent of Gradio or audio libraries.
"""
import sys, os
import uvicorn
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import src.core.win_fix
from src.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
