from shared.configuration import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVER_PROVIDER,
    get_config,
)


def test_get_config_empty() -> None:
    assert get_config(None) == {}
    assert get_config({}) == {}


def test_default_constants() -> None:
    assert DEFAULT_EMBEDDING_MODEL == "E:/python_work/PDFReader_Agent/bge-m3"
    assert DEFAULT_RETRIEVER_PROVIDER == "chroma"
    assert DEFAULT_CHROMA_PATH == "./chroma_db"
