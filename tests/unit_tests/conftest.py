"""Keep unit tests independent of a contributor's local environment defaults."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_papermind_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("PAPERMIND_"):
            monkeypatch.delenv(name)
