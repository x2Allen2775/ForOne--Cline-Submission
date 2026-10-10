#!/usr/bin/env python3
"""
Launcher for the Anemia-Screen Multimodal Pipeline.

Usage:
    python run_pipeline.py
"""
import sys
from scripts.analyze_cli import main

if __name__ == "__main__":
    sys.exit(main())
