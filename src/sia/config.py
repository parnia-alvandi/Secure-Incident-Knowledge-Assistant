"""Typed configuration loader.

This is the ONLY module that reads config.yaml. Other modules receive a
`Settings` object, so models/paths/thresholds can change without touching logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    keep_alive: str = "30m"     # how long Ollama keeps the model in RAM after a request
    timeout_s: float = 300.0    # client timeout for one generation


@dataclass(frozen=True)
class IngestionConfig:
    manifest_path: Path
    supported_extensions: tuple[str, ...]
    min_chars_per_page: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class ChunkingSettings:
    """Chunking parameters (tokens are counted with the embedding model's tokenizer)."""
    chunk_tokens: int = 256
    overlap_tokens: int = 40
    min_tokens: int = 40


@dataclass(frozen=True)
class EmbeddingSettings:
    batch_size: int = 32
    max_seq_length: int = 512
    offline: bool = False      # True = never contact the Hugging Face Hub (model must be cached)
    # BGE v1.5 retrieval prefix for QUERIES only (passages are embedded without it)
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class IndexSettings:
    collection_name: str = "sia_chunks"


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5
    # Insufficient-evidence gate: if the best cosine score is below this value the
    # LLM is not called. None = gate disabled (until calibrated in phase 4c).
    min_score: float | None = None
    use_query_instruction: bool = True


@dataclass(frozen=True)
class AnsweringSettings:
    excerpt_chars: int = 350


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    models: ModelsConfig
    llm: LLMConfig
    ingestion: IngestionConfig
    logging: LoggingConfig
    # Phase 3 sections are optional in config.yaml: defaults apply when absent.
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    index: IndexSettings = field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    answering: AnsweringSettings = field(default_factory=AnsweringSettings)


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
            keep_alive=str(raw["llm"].get("keep_alive", "30m")),
            timeout_s=float(raw["llm"].get("timeout_s", 300.0)),
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

    chunking = ChunkingSettings(**_section(raw, "chunking", ChunkingSettings))
    embedding = EmbeddingSettings(**_section(raw, "embedding", EmbeddingSettings))
    index = IndexSettings(**_section(raw, "index", IndexSettings))
    retrieval = RetrievalSettings(**_section(raw, "retrieval", RetrievalSettings))
    answering = AnsweringSettings(**_section(raw, "answering", AnsweringSettings))
    if retrieval.top_k < 1:
        raise ValueError("retrieval.top_k must be >= 1")
    if retrieval.min_score is not None and not -1.0 <= retrieval.min_score <= 1.0:
        raise ValueError("retrieval.min_score must be None or between -1 and 1")
    if answering.excerpt_chars < 50:
        raise ValueError("answering.excerpt_chars must be >= 50")
    if not 0 <= chunking.overlap_tokens < chunking.chunk_tokens:
        raise ValueError("chunking.overlap_tokens must be >= 0 and < chunking.chunk_tokens")
    if chunking.chunk_tokens <= 0 or chunking.min_tokens < 0:
        raise ValueError("chunking.chunk_tokens must be > 0 and chunking.min_tokens >= 0")

    return Settings(paths=paths, models=models, llm=llm, ingestion=ingestion,
                    logging=logging_cfg, chunking=chunking, embedding=embedding, index=index,
                    retrieval=retrieval, answering=answering)


def _section(raw: dict, name: str, cls: type) -> dict:
    """Optional YAML section -> kwargs. Unknown keys are an error (typo protection)."""
    values = raw.get(name) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in config section '{name}': {sorted(unknown)}")
    return dict(values)


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p