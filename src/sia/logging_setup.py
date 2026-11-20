"""Logging setup: console + rotating file."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Third-party loggers that flood the console at INFO level.
_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers")


def setup_logging(level: str = "INFO", logs_dir: Path | None = None) -> None:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if called again.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            logs_dir / "sia.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence chatty libraries (their own WARNING/ERROR messages still appear).
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)