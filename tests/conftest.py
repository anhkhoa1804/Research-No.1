"""Ensures the repo root (where the openvocab_rel/ package lives) is on
sys.path regardless of the working directory pytest is invoked from, since
this repo has no pyproject.toml/setup.py package install step yet."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
