# -*- coding: utf-8 -*-
"""Repository hygiene: files that must never end up in the public repo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gitignore_rules():
    return (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_private_notes_are_ignored():
    rules = _gitignore_rules()
    assert ".workbuddy/" in rules
    assert any(line.startswith("TECHNICAL_REVIEW") for line in rules)


def test_downloaded_datasets_are_ignored():
    assert "pupu_data/" in _gitignore_rules()


def test_generated_outputs_are_ignored():
    assert "outputs/" in _gitignore_rules()


def test_required_public_files_exist():
    for name in ["README.md", "README.zh-CN.md", "LICENSE", "CITATION.cff", "pyproject.toml"]:
        assert (ROOT / name).exists(), f"missing required file: {name}"
