from __future__ import annotations

import sys

from .modeldefs import ALLOWED_MODEL_KEYS, normalize_model_key
from .train import main as train_main


def run_model_module(model_key: str) -> None:
    normalized = normalize_model_key(model_key)
    if normalized not in ALLOWED_MODEL_KEYS:
        allowed = ", ".join(sorted(ALLOWED_MODEL_KEYS))
        raise ValueError(f"Unsupported model '{model_key}'. Allowed models: {allowed}")
    train_main([*sys.argv[1:], "--model", normalized])
