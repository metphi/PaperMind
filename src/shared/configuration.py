"""Runtime configuration defaults shared by all three graphs.

The graphs no longer materialize a dataclass: nodes read their knobs straight
from ``config["configurable"]`` via :func:`get_config` and fall back to the
default constants below.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

# Load .env from the project root on first import (no-op if already loaded,
# and never overrides existing environment variables).
load_dotenv()


def get_config(config: RunnableConfig | None = None) -> dict:
    """Merge environment defaults with runtime overrides (top-level keys).

    JSON settings must be objects. Runtime dictionaries replace the whole
    corresponding environment object; custom runtime keys are preserved.
    """
    defaults = {}
    for key in (
        "retriever_provider", "embedding_model", "embedding_device",
        "chroma_path", "chroma_collection", "query_model", "response_model",
        "search_kwargs", "splitter_configuration",
    ):
        name = f"PAPERMIND_{key.upper()}"
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        if key in ("search_kwargs", "splitter_configuration"):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} must contain a valid JSON object") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{name} must contain a JSON object")
            defaults[key] = value
        else:
            defaults[key] = raw.strip()
    defaults.update((config or {}).get("configurable") or {})
    return defaults


DEFAULT_EMBEDDING_MODEL = "E:/python_work/PDFReader_Agent/bge-m3"
"""Local sentence-transformers model used for embeddings."""

DEFAULT_EMBEDDING_DEVICE = "cpu"
"""Device used for local embedding inference (e.g. 'cpu' or 'cuda')."""

RETRIEVER_PROVIDERS = ("elastic-local", "elastic", "pinecone", "mongodb", "chroma")
"""Every retriever provider the shared layer can instantiate."""

DEFAULT_RETRIEVER_PROVIDER = "chroma"
"""The default vector-store provider."""

DEFAULT_CHROMA_PATH = "./chroma_db"
"""Local directory for Chroma persistent storage."""

DEFAULT_CHROMA_COLLECTION = "langchain_index"
"""Chroma collection name."""

DEFAULT_SEARCH_KWARGS = {"k": 8}
"""Default retriever ``search_kwargs`` (k=8, scanned 2026-08-12 on the eval
set — k=8 is the marginal turning point). Applied when the caller does not
supply ``search_kwargs`` in configurable; eval passes it explicitly."""

