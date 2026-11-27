#!/usr/bin/env python3
"""Unified entry point for NTO Case 2A from workspace root.

Usage:
    python main.py
    python main.py --train
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
solution_dir = root_dir / 'solution'
if str(solution_dir) not in sys.path:
    sys.path.insert(0, str(solution_dir))

import main as sol_main

if __name__ == '__main__':
    sol_main.main()
