"""Fixtures shared across test modules."""

import os

import pytest


@pytest.fixture
def no_anthropic_credentials(monkeypatch, tmp_path):
    """Hide every Anthropic credential and setting the test machine may have."""
    for name in list(os.environ):
        if name.startswith("ANTHROPIC_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile
