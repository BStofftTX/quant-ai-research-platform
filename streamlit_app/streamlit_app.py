"""Community Cloud entrypoint for the slim, offline Quant-AI demonstration."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

run_demo = import_module("app").run_demo


run_demo()
