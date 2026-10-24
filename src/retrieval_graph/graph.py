"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from retrieval_graph.configuration import (
    DEFAULT_GENERAL_SYSTEM_PROMPT,
    DEFAULT_MORE_INFO_SYSTEM_PROMPT,
    DEFAULT_QUERY_MODEL,
    DEFAULT_RESEARCH_PLAN_SYSTEM_PROMPT,
    DEFAULT_RESPONSE_MODEL,
    DEFAULT_RESPONSE_SYSTEM_PROMPT,
    DEFAULT_ROUTER_SYSTEM_PROMPT,
)
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router
from shared.configuration import get_config
from shared.utils import format_docs, load_chat_model

# Safety cap on research-plan length. The plan itself is decided dynamically by
# the LLM (prompt: "研究步数尽可能少"); this only bounds pathological over-plans
# so the sequential conduct_research loop can't run away.
MAX_PLAN_STEPS = 5


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    configurable = get_config(config)
    model = load_chat_model(configurable.get("query_model", DEFAULT_QUERY_MODEL))
    messages = [
        {
            "role": "system",
            "content": configurable.get(
                "router_system_prompt", DEFAULT_ROUTER_SYSTEM_PROMPT
            ),
        }
    ] + state.messages
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "langchain":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configurable = get_config(config)
    model = load_chat_model(configurable.get("query_model", DEFAULT_QUERY_MODEL))
    system_prompt = configurable.get(
        "more_info_system_prompt", DEFAULT_MORE_INFO_SYSTEM_PROMPT
    ).format(logic=state.router["logic"])
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configurable = get_config(config)
    model = load_chat_model(configurable.get("query_model", DEFAULT_QUERY_MODEL))
    system_prompt = configurable.get(
        "general_system_prompt", DEFAULT_GENERAL_SYSTEM_PROMPT
    ).format(logic=state.router["logic"])
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    configurable = get_config(config)
    model = load_chat_model(
        configurable.get("query_model", DEFAULT_QUERY_MODEL)
    ).with_structured_output(Plan)
    messages = [
        {
            "role": "system",
            "content": configurable.get(
                "research_plan_system_prompt", DEFAULT_RESEARCH_PLAN_SYSTEM_PROMPT
            ),
        }
    ] + state.messages
    response = cast(Plan, await model.ainvoke(messages))
    steps = (response["steps"] or [])[:MAX_PLAN_STEPS]
    # 计划由 LLM 动态决定；空计划会让 conduct_research 在 steps[0] 处崩溃，
    # 回退为把用户最近一次提问当作唯一检索步。
    if not steps:
        user_content = next(
            (
                m.content
                for m in reversed(state.messages)
                if getattr(m, "type", None) == "human" and isinstance(m.content, str)
            ),
            "",
        )
        steps = [user_content] if user_content else ["用户的问题"]
    return {"steps": steps, "documents": "delete"}


async def conduct_research(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke(
        {"question": state.steps[0]}, config
    )
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configurable = get_config(config)
    model = load_chat_model(configurable.get("response_model", DEFAULT_RESPONSE_MODEL))
    context = format_docs(state.documents)
    prompt = configurable.get(
        "response_system_prompt", DEFAULT_RESPONSE_SYSTEM_PROMPT
    ).format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
builder = StateGraph(AgentState, input_schema=InputState)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node(create_research_plan)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
