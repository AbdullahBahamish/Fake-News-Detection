from __future__ import annotations

import sys
from pathlib import Path

try:
    from .train import main
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.train import main


if __name__ == "__main__":
    main()
