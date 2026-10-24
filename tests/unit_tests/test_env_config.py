"""Environment defaults and per-call overrides share one configuration path."""

import pytest

from shared.configuration import get_config


def test_environment_defaults_and_runtime_override(monkeypatch):
    monkeypatch.setenv("PAPERMIND_EMBEDDING_MODEL", "local/model")
    monkeypatch.setenv("PAPERMIND_SEARCH_KWARGS", '{"k": 3}')
    monkeypatch.setenv("PAPERMIND_SPLITTER_CONFIGURATION", '{"chunk_size": 256}')
    result = get_config({"configurable": {"search_kwargs": {"k": 1}, "custom": True}})
    assert result["embedding_model"] == "local/model"
    assert result["search_kwargs"] == {"k": 1}
    assert result["splitter_configuration"] == {"chunk_size": 256}
    assert result["custom"] is True


@pytest.mark.parametrize("raw", ["invalid", "[]", "null", "8"])
def test_invalid_environment_object(monkeypatch, raw):
    monkeypatch.setenv("PAPERMIND_SEARCH_KWARGS", raw)
    with pytest.raises(ValueError, match="PAPERMIND_SEARCH_KWARGS"):
        get_config()
