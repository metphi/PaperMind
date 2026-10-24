"""Shared primitives for the chunker graph.

Functions:
    normalize_source: Normalize a document's ``source`` identity.
    sanitize_metadata: Reduce arbitrary metadata to Chroma-safe scalar values.
"""

from pathlib import Path

# Metadata values allowed to pass through unchanged (Chroma requires scalar
# metadata: str/int/float/bool only — no None, list, tuple, or nested dict).
_SCALAR_TYPES = (str, int, float, bool)


def normalize_source(source: str, *, local: bool = True) -> str:
    """Normalize a document identity to a canonical string.

    A document's ``source`` is its **identity** (the thing re-chunking a
    document keys on when purging stale vectors). Two representations of the
    same document must normalize to the same string.

    - Local mode (default): ``source`` is a filesystem path. It is resolved to
      an absolute path so ``papers/a.pdf`` and ``E:/.../papers/a.pdf`` collapse
      to one identity.
    - Online mode (``local=False``): ``source`` is a client-provided doc_id
      with no filesystem meaning, so it is passed through verbatim.

    Args:
        source (str): The document identity (local path or online doc_id).
        local (bool): True when ``source`` is a real filesystem path.

    Returns:
        str: The canonical identity string.

    Raises:
        ValueError: If ``source`` is empty or not a string.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"source must be a non-empty string, got {source!r}")
    if local:
        return str(Path(source).resolve())
    return source


def sanitize_metadata(metadata: dict) -> dict:
    """Return a copy of ``metadata`` containing only scalar values.

    Vector stores such as Chroma reject None/list/tuple values in metadata, so
    they are dropped; anything else that is not a str/int/float/bool is coerced
    to str. The input dict is not mutated.

    Args:
        metadata (dict): Arbitrary document metadata.

    Returns:
        dict: A new dict whose values are all str/int/float/bool.
    """
    out = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (list, tuple)):
            continue
        if isinstance(value, _SCALAR_TYPES):
            out[key] = value
        else:
            out[key] = str(value)
    return out
