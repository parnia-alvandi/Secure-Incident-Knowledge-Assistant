"""Typed configuration loader.

This is the ONLY module that reads config.yaml. Other modules receive a
`Settings` object, so models/paths/thresholds can change without touching logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    docs_dir: Path
    vector_db_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ModelsConfig:
    embedding: str
    reranker: str


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float
    num_ctx: int
    num_predict: int


@dataclass(frozen=True)
class IngestionConfig:
    manifest_path: Path
    supported_extensions: tuple[str, ...]
    min_chars_per_page: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    models: ModelsConfig
    llm: LLMConfig
    ingestion: IngestionConfig
    logging: LoggingConfig


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from YAML. Relative paths are resolved against PROJECT_ROOT."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file is empty or malformed: {path}")

    try:
        p = raw["paths"]
        paths = PathsConfig(
            docs_dir=_resolve(p["docs_dir"]),
            vector_db_dir=_resolve(p["vector_db_dir"]),
            logs_dir=_resolve(p["logs_dir"]),
        )
        models = ModelsConfig(**raw["models"])
        llm = LLMConfig(
            provider=raw["llm"]["provider"],
            model=raw["llm"]["model"],
            temperature=float(raw["llm"]["temperature"]),
            num_ctx=int(raw["llm"]["num_ctx"]),
            num_predict=int(raw["llm"]["num_predict"]),
        )
        ing = raw["ingestion"]
        ingestion = IngestionConfig(
            manifest_path=_resolve(ing["manifest_path"]),
            supported_extensions=tuple(e.lower() for e in ing["supported_extensions"]),
            min_chars_per_page=int(ing["min_chars_per_page"]),
        )
        logging_cfg = LoggingConfig(level=str(raw["logging"]["level"]).upper())
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc

    return Settings(paths=paths, models=models, llm=llm,
                    ingestion=ingestion, logging=logging_cfg)


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p