from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from .config import PATHS, TRAINING_CONFIG


RANDOM_STATE = TRAINING_CONFIG.random_state


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_results_directories() -> dict[str, Path]:
    return {
        "results": ensure_directory(PATHS.results_dir),
        "plots": ensure_directory(PATHS.plots_dir),
        "reports": ensure_directory(PATHS.reports_dir),
    }


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def estimate_matrix_size_mb(matrix) -> float:
    if hasattr(matrix, "data") and hasattr(matrix, "indptr") and hasattr(matrix, "indices"):
        size_bytes = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    else:
        size_bytes = np.asarray(matrix).nbytes
    return float(size_bytes / (1024 * 1024))
