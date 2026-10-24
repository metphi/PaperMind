"""Default configuration constants for the retrieval agent.

Runtime-config style: the graph no longer materializes a dataclass. Nodes read
their knobs straight from ``config["configurable"]`` via
``shared.configuration.get_config`` and fall back to the constants below.
"""

from __future__ import annotations

from retrieval_graph import prompts

DEFAULT_QUERY_MODEL = "qwen/qwen3.8-max"
DEFAULT_RESPONSE_MODEL = "qwen/qwen3.8-max"

DEFAULT_ROUTER_SYSTEM_PROMPT = prompts.ROUTER_SYSTEM_PROMPT
DEFAULT_MORE_INFO_SYSTEM_PROMPT = prompts.MORE_INFO_SYSTEM_PROMPT
DEFAULT_GENERAL_SYSTEM_PROMPT = prompts.GENERAL_SYSTEM_PROMPT
DEFAULT_RESEARCH_PLAN_SYSTEM_PROMPT = prompts.RESEARCH_PLAN_SYSTEM_PROMPT
DEFAULT_GENERATE_QUERIES_SYSTEM_PROMPT = prompts.GENERATE_QUERIES_SYSTEM_PROMPT
DEFAULT_RESPONSE_SYSTEM_PROMPT = prompts.RESPONSE_SYSTEM_PROMPT
