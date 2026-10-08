# -*- coding: utf-8 -*-
"""
Shared pytest configuration: lets test files import the scripts under src/.

Each script was originally a "directly run script", not an "importable package".
For tests to call them, Python must first be told "the code is in the src/ folder".
This file is auto-loaded by pytest when running tests, and does just this one thing.
"""
import sys
from pathlib import Path

# Project root (one level above tests/)
ROOT = Path(__file__).resolve().parent.parent

# Add src/ to Python's search path so tests can `import pressure_simulator` etc.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
