# Repository Guidelines

## Project Structure & Module Organization
PaperMind is a Python 3.10+ research-paper RAG application using LangGraph and LangChain.
- `src/chunk_graph/`: PDF loading and pluggable text splitting.
- `src/index_graph/`: document embedding and vector-store indexing.
- `src/retrieval_graph/`: routing, research planning, and answers; `researcher_graph/` handles query generation and parallel retrieval.
- `src/shared/`: configuration, retrieval providers, metadata utilities, and shared state.
- `tests/unit_tests/` and `tests/integration_tests/`: pytest suites.
- `scripts/` and `evalset/`: indexing tools, evaluation datasets, corpus, and reports.
- `langgraph.json`: graph registrations. Local PDFs belong in ignored `papers/`.

## Build, Test, and Development Commands
Run commands from the repository root in a virtual environment with the required dependencies installed.
- `python -m venv .venv`: create an isolated Python environment.
- PowerShell: `$env:PYTHONPATH = "$PWD/src"` makes source packages importable.
- `python -m pytest tests/unit_tests -q`: run unit tests.
- `python -m pytest tests/integration_tests -q`: run integration tests after configuring required services and fixtures.
- `python scripts/evalset_validate.py`: validate evaluation data and evidence anchors.
- `langgraph dev`: start the local API with `langgraph-cli[inmem]` installed; newer CLI versions may require Python 3.11+.

This checkout has no package manifest or build configuration. The README's `pip install -e .` requires packaging metadata that is currently absent.

## Coding Style & Naming Conventions
Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Follow existing type annotations and docstrings. Keep graph nodes asynchronous where appropriate. Read runtime overrides through `config["configurable"]` and forward configuration to subgraphs. No formatter or linter configuration is included.

## Testing Guidelines
Name files `test_*.py` and functions `test_*`. Use pytest fixtures and monkeypatching to isolate model calls in unit tests. Async integration tests use `pytest.mark.asyncio`; some require Elasticsearch, model credentials, local embeddings, or PDF fixtures. Add regression coverage for changed behavior; no numeric coverage threshold is configured.

## Commit & Pull Request Guidelines
Git metadata is absent, so existing commit conventions cannot be verified. Use concise, imperative commit subjects. PRs should explain behavior changes, link relevant issues, and list validation commands and required services. Include evaluation comparisons when changing retrieval, chunking, or prompts.

## Security & Configuration Tips
Copy `.env.example` to `.env` for credentials; never commit secrets, personal PDFs, or vector databases. Override machine-specific embedding paths. Keep embedding model, Chroma directory, and collection consistent between indexing and retrieval. Review evaluation outputs before committing; generation evaluations can incur API charges.
