"""Tests for configuration loading."""
from pathlib import Path

import pytest

from sia.config import PROJECT_ROOT, load_settings


def test_default_config_loads():
    s = load_settings()
    assert s.llm.provider == "ollama"
    assert s.llm.num_ctx > 0
    assert s.models.embedding


def test_relative_paths_resolved_to_project_root():
    s = load_settings()
    assert s.paths.docs_dir.is_absolute()
    assert s.paths.docs_dir == PROJECT_ROOT / "data" / "docs"


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


def test_missing_key_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("paths:\n  docs_dir: a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(bad)