"""
Eye-NAV UI – Quick Launcher
============================
Starts the Flask backend from the ui/backend directory.

Usage:
    cd ui
    python run.py

    # To use live files from main_controller (instead of mock data):
    DATA_SOURCE=file python run.py      # Linux / macOS
    $env:DATA_SOURCE="file"; python run.py   # PowerShell
"""

import os
import sys
import subprocess
from pathlib import Path

BACKEND_DIR = Path(__file__).parent / "backend"

if __name__ == "__main__":
    os.chdir(BACKEND_DIR)
    cmd = [sys.executable, "app.py"] + sys.argv[1:]
    print(f"[Launcher] Starting backend from: {BACKEND_DIR}")
    subprocess.run(cmd)
