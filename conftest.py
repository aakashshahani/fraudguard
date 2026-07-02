"""Make the project root importable so `from src.data_prep import ...` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
