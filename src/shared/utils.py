"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator

import httpx

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_chat_http_client: ContextVar[httpx.AsyncClient | None] = ContextVar(
    "chat_http_client", default=None
)


@asynccontextmanager
async def chat_client_session() -> AsyncIterator[None]:
    """Own the Qwen HTTP client for one async graph invocation or session.

    Child graph tasks inherit the client through their context. Exit this scope
    only after all graph calls and streams have finished.
    """
    if _chat_http_client.get() is not None:
        yield
        return
    async with httpx.AsyncClient() as client:
        token = _chat_http_client.set(client)
        try:
            yield
        finally:
            _chat_http_client.reset(token)


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: list[Document] | None) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    if provider == "qwen":
        # Qwen models are served through an OpenAI-compatible endpoint on
        # Alibaba Cloud DashScope, so wrap it with ChatOpenAI.
        # Explicit timeout + max_retries: without them a stalled DashScope
        # request can hang a run for hours (observed 2026-08-13 in the C-2
        # eval). timeout=180 is generous for long generations.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=os.environ["QWEN_API_KEY"],
            base_url=QWEN_BASE_URL,
            timeout=180,
            max_retries=1,
            http_async_client=_chat_http_client.get(),
        )

    return init_chat_model(model, model_provider=provider)
