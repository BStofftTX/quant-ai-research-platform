"""Deprecated compatibility entrypoint.

The supported Streamlit Community Cloud entrypoint is
``streamlit_app/streamlit_app.py``. This thin shim remains temporarily so an
older deployment cannot silently serve a divergent interface. It delegates to
the canonical implementation in ``app.py`` and must not receive new UI work.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

run_demo = import_module("app").run_demo


run_demo()
