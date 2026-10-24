"""Abstract interfaces for pluggable document loaders and text splitters.

The two ABCs are the **only** surface a user needs to know. Writing a custom
loader or splitter means subclassing the ABC and implementing the single hook
method :meth:`DocumentLoader._load` / :meth:`TextSplitter._split`. Everything
else is handled here:

- **Config-driven pluggability.** ``from_provider(name)`` resolves a provider
  string — a registered short name (built-ins ``"pdf"`` / ``"recursive"``) or a
  dotted module path (``"mypkg.myloader.MyLoader"``). Path providers are
  imported lazily at call time, so user subclasses never need to be registered
  by hand; any class that is merely defined registers itself via
  ``__init_subclass__``.
- **Template methods.** ``load`` / ``split`` are concrete skeletons. They
  delegate the user's logic to the abstract hook, then enforce the shared
  contract automatically: the document identity is normalized and stamped as
  ``source``, and every metadata dict is reduced to Chroma-safe scalar values.
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from shared.chunking import normalize_source, sanitize_metadata


def _resolve_provider(
    provider: str,
    registry: dict[str, type],
    base: type,
) -> Any:
    """Resolve a provider string to a strategy instance.

    Checks the registry first (short names such as ``"pdf"``); otherwise treats
    ``provider`` as a dotted module path and imports the module on demand.
    """
    if provider in registry:
        return registry[provider]()
    module_name, _, class_name = provider.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(
            f"Unknown {base.__name__} provider {provider!r}: not a registered "
            "provider and not a 'module.Class' dotted path."
        )
    try:
        module = importlib.import_module(module_name)
        strategy_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Unknown {base.__name__} provider {provider!r}: could not import "
            f"{module_name}.{class_name}."
        ) from exc
    if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, base)):
        raise ValueError(
            f"Unknown {base.__name__} provider {provider!r}: "
            f"{module_name}.{class_name} is not a {base.__name__} subclass."
        )
    return strategy_cls()


class DocumentLoader(ABC):
    """Turn a document source into page-level raw documents.

    Concrete loaders implement :meth:`_load`; the base class normalizes the
    ``source`` identity, stamps it on every returned document, and sanitizes
    all metadata to scalar values.
    """

    _registry: dict[str, type[DocumentLoader]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register subclasses by their ``provider`` name."""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # keep intermediate abstract subclasses out of the registry
        provider = cls.__dict__.get("provider") or cls.__name__
        DocumentLoader._registry[provider] = cls

    @classmethod
    def from_provider(
        cls, provider: str, config: Any | None = None
    ) -> DocumentLoader:
        """Create a loader from a provider name (short name or dotted path).

        ``config`` is the loader's *private* sub-config — the
        ``loader_configuration`` value under ``configurable`` — passed to
        :meth:`configure` so a loader can read its own parameters without the
        graph knowing about them.
        """
        loader = _resolve_provider(provider, cls._registry, cls)
        if config is not None:
            loader.configure(config)
        return loader

    def configure(self, config: Any) -> None:
        """Read any strategy-specific parameters from the loader's sub-config.

        Optional hook: the default does nothing. Override to pull your own
        parameters out of ``loader_configuration`` (for example ``prefix``);
        only keys you look up are consumed.
        """

    def load(
        self,
        source: str,
        *,
        data: bytes | None = None,
        metadata: dict | None = None,
    ) -> list[Document]:
        """Load a document, normalizing identity and metadata on the way out.

        Computes the canonical ``source`` identity, delegates parsing to
        :meth:`_load`, then stamps the identity and sanitizes metadata on every
        returned document.
        """
        identity = normalize_source(source, local=self._is_local(source, data))
        raw_docs = self._load(source, data=data, metadata=metadata)
        return self._finalize(raw_docs, identity)

    @abstractmethod
    def _load(
        self,
        source: str,
        *,
        data: bytes | None = None,
        metadata: dict | None = None,
    ) -> list[Document]:
        """Parse ``source`` / ``data`` into one raw ``Document`` per unit (page)."""

    def _is_local(self, source: str, data: bytes | None) -> bool:
        """Return True when ``source`` is a filesystem path, not an opaque id.

        Defaults to "local iff no remote bytes were given"; override for
        loaders whose ``source`` means something else (for example a URL).
        """
        return data is None

    def _finalize(self, raw_docs: list[Document], identity: str) -> list[Document]:
        """Stamp the document identity and reduce metadata to scalar values."""
        finalized = []
        for doc in raw_docs:
            meta = dict(doc.metadata or {})
            meta["source"] = identity
            finalized.append(
                doc.model_copy(update={"metadata": sanitize_metadata(meta)})
            )
        return finalized


class TextSplitter(ABC):
    """Split page-level raw documents into retrieval-ready chunks.

    Concrete splitters implement :meth:`_split`; :meth:`split` sanitizes chunk
    metadata afterwards so custom splitters never need to know about the scalar
    contract.
    """

    _registry: dict[str, type[TextSplitter]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register subclasses by their ``provider`` name."""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        provider = cls.__dict__.get("provider") or cls.__name__
        TextSplitter._registry[provider] = cls

    @classmethod
    def from_provider(
        cls, provider: str, config: Any | None = None
    ) -> TextSplitter:
        """Create a splitter from a provider name (short name or dotted path).

        ``config`` is the splitter's *private* sub-config — the
        ``splitter_configuration`` value under ``configurable`` — passed to
        :meth:`configure`.
        """
        splitter = _resolve_provider(provider, cls._registry, cls)
        if config is not None:
            splitter.configure(config)
        return splitter

    def configure(self, config: Any) -> None:
        """Read any strategy-specific parameters from the splitter's sub-config.

        Optional hook: the default does nothing. ``RecursiveSplitter`` reads
        ``chunk_size`` / ``chunk_overlap`` from ``split_configuration`` here.
        """

    def split(self, raw_docs: list[Document]) -> list[Document]:
        """Split raw documents into chunks, sanitizing metadata on the way out."""
        chunks = self._split(raw_docs)
        return self._finalize(chunks)

    @abstractmethod
    def _split(self, raw_docs: list[Document]) -> list[Document]:
        """Split raw page documents into chunks, preserving metadata."""

    def _finalize(self, chunks: list[Document]) -> list[Document]:
        """Reduce every chunk's metadata to scalar values."""
        return [
            doc.model_copy(update={"metadata": sanitize_metadata(doc.metadata or {})})
            for doc in chunks
        ]
