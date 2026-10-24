"""Tests for the runtime-config style from the configurable→runtime rewrite.

Nodes must read their knobs straight from ``config["configurable"]`` (falling
back to the default constants) instead of materializing a dataclass. These tests
drive each config-reading node directly with a fake chat model, so no network or
API key is needed. They also guard iron rule #1: ``conduct_research`` still
forwards the caller's config verbatim to the researcher subgraph.
"""

import asyncio
import importlib
from types import SimpleNamespace

from retrieval_graph.configuration import DEFAULT_QUERY_MODEL, DEFAULT_RESPONSE_MODEL
from shared.configuration import get_config
from shared.utils import format_docs

# ``retrieval_graph.__init__`` re-exports the compiled graph as ``graph``, which
# shadows the submodule. importlib gives us the actual module so we can patch
# its ``load_chat_model`` / ``researcher_graph`` names.
rg = importlib.import_module("retrieval_graph.graph")
rg_researcher = importlib.import_module("retrieval_graph.researcher_graph.graph")


def _run(coro):
    """Run an async node call to completion on a fresh event loop."""
    return asyncio.run(coro)


class FakeModel:
    """Records the spec it was created with and every messages list it saw."""

    def __init__(self, spec, return_value=None):
        self.spec = spec
        self.calls = []
        self._return_value = (
            return_value if return_value is not None else {"type": "general", "logic": ""}
        )

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._return_value


class FakeModelFactory:
    """Stand-in for ``load_chat_model``; records every FakeModel it creates."""

    def __init__(self, return_value=None):
        self.return_value = return_value
        self.instances = []

    def __call__(self, spec):
        model = FakeModel(spec, self.return_value)
        self.instances.append(model)
        return model


def test_get_config():
    assert get_config(None) == {}
    assert get_config({}) == {}
    assert get_config({"configurable": {"a": 1}}) == {"a": 1}


def test_defaults_exposed():
    assert DEFAULT_QUERY_MODEL == "qwen/qwen3.8-max"
    assert DEFAULT_RESPONSE_MODEL == "qwen/qwen3.8-max"


def test_analyze_and_route_query_reads_overrides(monkeypatch):
    factory = FakeModelFactory()
    monkeypatch.setattr(rg, "load_chat_model", factory)

    result = _run(
        rg.analyze_and_route_query(
            SimpleNamespace(messages=[{"role": "user", "content": "hi"}]),
            config={
                "configurable": {
                    "query_model": "qwen/override-model",
                    "router_system_prompt": "CUSTOM_ROUTER",
                }
            },
        )
    )

    assert result["router"]["type"] == "general"
    model = factory.instances[0]
    assert model.spec == "qwen/override-model"
    assert model.calls[0][0]["content"] == "CUSTOM_ROUTER"


def test_ask_for_more_info_reads_overrides(monkeypatch):
    factory = FakeModelFactory()
    monkeypatch.setattr(rg, "load_chat_model", factory)

    _run(
        rg.ask_for_more_info(
            SimpleNamespace(
                messages=[], router={"type": "more-info", "logic": "which paper?"}
            ),
            config={
                "configurable": {
                    "query_model": "qwen/override",
                    "more_info_system_prompt": "CUSTOM {logic}",
                }
            },
        )
    )

    model = factory.instances[0]
    assert model.spec == "qwen/override"
    assert model.calls[0][0]["content"] == "CUSTOM which paper?"


def test_respond_to_general_query_reads_overrides(monkeypatch):
    factory = FakeModelFactory()
    monkeypatch.setattr(rg, "load_chat_model", factory)

    _run(
        rg.respond_to_general_query(
            SimpleNamespace(messages=[], router={"type": "general", "logic": "why?"}),
            config={
                "configurable": {
                    "query_model": "qwen/override",
                    "general_system_prompt": "CUSTOM {logic}",
                }
            },
        )
    )

    model = factory.instances[0]
    assert model.spec == "qwen/override"
    assert model.calls[0][0]["content"] == "CUSTOM why?"


def test_create_research_plan_reads_overrides(monkeypatch):
    factory = FakeModelFactory({"steps": ["step1"]})
    monkeypatch.setattr(rg, "load_chat_model", factory)

    result = _run(
        rg.create_research_plan(
            SimpleNamespace(messages=[]),
            config={
                "configurable": {
                    "query_model": "qwen/override",
                    "research_plan_system_prompt": "CUSTOM_PLAN",
                }
            },
        )
    )

    assert result == {"steps": ["step1"], "documents": "delete"}
    model = factory.instances[0]
    assert model.spec == "qwen/override"
    assert model.calls[0][0]["content"] == "CUSTOM_PLAN"


def test_respond_reads_overrides(monkeypatch):
    factory = FakeModelFactory()
    monkeypatch.setattr(rg, "load_chat_model", factory)

    _run(
        rg.respond(
            SimpleNamespace(messages=[], documents=[]),
            config={
                "configurable": {
                    "response_model": "qwen/override-response",
                    "response_system_prompt": "CUSTOM_{context}",
                }
            },
        )
    )

    model = factory.instances[0]
    assert model.spec == "qwen/override-response"
    assert model.calls[0][0]["content"] == "CUSTOM_" + format_docs([])


def test_generate_queries_reads_overrides(monkeypatch):
    factory = FakeModelFactory({"queries": ["q1"]})
    monkeypatch.setattr(rg_researcher, "load_chat_model", factory)

    result = _run(
        rg_researcher.generate_queries(
            SimpleNamespace(question="what is langgraph?"),
            config={
                "configurable": {
                    "query_model": "qwen/override",
                    "generate_queries_system_prompt": "CUSTOM_GQ",
                }
            },
        )
    )

    assert result["queries"] == ["q1"]
    model = factory.instances[0]
    assert model.spec == "qwen/override"
    assert model.calls[0][0]["content"] == "CUSTOM_GQ"


def test_nodes_fall_back_to_defaults(monkeypatch):
    factory = FakeModelFactory()
    monkeypatch.setattr(rg, "load_chat_model", factory)

    _run(rg.analyze_and_route_query(SimpleNamespace(messages=[]), config=None))

    assert factory.instances[0].spec == DEFAULT_QUERY_MODEL


def test_conduct_research_passes_config_through(monkeypatch):
    """Iron rule #1: the caller's config reaches the researcher subgraph verbatim."""
    seen = {}

    async def fake_ainvoke(payload, config=None):
        seen["config"] = config
        return {"documents": ["d"]}

    monkeypatch.setattr(rg, "researcher_graph", SimpleNamespace(ainvoke=fake_ainvoke))

    cfg = {"configurable": {"query_model": "x"}}
    result = _run(
        rg.conduct_research(
            SimpleNamespace(steps=["s1", "s2"], messages=[]), config=cfg
        )
    )

    assert result == {"documents": ["d"], "steps": ["s2"]}
    assert seen["config"] is cfg
