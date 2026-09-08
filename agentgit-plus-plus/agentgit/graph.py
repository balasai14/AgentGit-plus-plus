"""
graph.py — LangGraph workflow definition for AgentGit++

Defines a 4-stage sequential report-writing workflow:
    Plan → Research → Draft → Review

Each stage:
    1. Calls the LLM with a stage-specific prompt
    2. Updates the shared agent state with the output
    3. Creates a checkpoint in the SQLite store
    4. Returns the updated state for the next stage

The graph uses LangGraph's StateGraph for orchestration but manages its
own checkpointing (not LangGraph's built-in checkpointer) so we have
full control over the checkpoint tree, branching, and scoring.

Usage:
    from agentgit.graph import run_workflow
    result = run_workflow("The impact of AI on healthcare")
    result = run_workflow("topic", from_checkpoint_id="abc123")  # resume
"""

import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agentgit.checkpoint_store import CheckpointStore

# Load environment variables from .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Agent State — the shared data structure flowing through the graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """The shared state that flows through all workflow stages.

    Each stage reads what it needs and writes its output back.
    The full state is checkpointed after every stage.

    Attributes:
        topic: The report topic provided by the user.
        plan: Output of the Plan stage (structured outline).
        research: Output of the Research stage (key facts/information).
        draft: Output of the Draft stage (full report draft).
        review: Output of the Review stage (final reviewed version).
        messages: Full message history across all stages.
        current_stage: Name of the stage currently executing.
        checkpoint_ids: Mapping of stage_name → checkpoint_id.
    """
    topic: str
    plan: str
    research: str
    draft: str
    review: str
    messages: list
    current_stage: str
    checkpoint_ids: dict


# ---------------------------------------------------------------------------
# Stage ordering — used for resumption logic
# ---------------------------------------------------------------------------
STAGE_ORDER = ["plan", "research", "draft", "review"]


# ---------------------------------------------------------------------------
# LLM initialization
# ---------------------------------------------------------------------------
def get_llm(model_name: Optional[str] = None, temperature: float = 0.7) -> ChatOpenAI:
    """Create an LLM client with the specified or default model.

    Supports OpenAI, OpenRouter, or any OpenAI-compatible API via
    the OPENAI_API_BASE environment variable.

    Args:
        model_name: Override the model name (defaults to MODEL_NAME env var).
        temperature: Sampling temperature for generation.

    Returns:
        A ChatOpenAI instance ready to invoke.
    """
    base_url = os.getenv("OPENAI_API_BASE")
    kwargs = dict(
        model=model_name or os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=temperature,
        max_tokens=1024,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Stage prompts — each stage has a system prompt and a user prompt template
# ---------------------------------------------------------------------------
STAGE_PROMPTS = {
    "plan": {
        "system": (
            "You are an expert research planner. Create a clear, structured "
            "outline for a short report. Include 3-4 main sections with key "
            "points to cover in each. Be specific and actionable."
        ),
        "user": "Create a detailed plan/outline for a short report on: {topic}",
    },
    "research": {
        "system": (
            "You are an expert researcher. Based on the provided plan, gather "
            "and organize key facts, statistics, and insights. Present the "
            "information in a well-structured format with clear citations of "
            "the types of sources that would support each point."
        ),
        "user": (
            "Research the following topic and gather key information based on "
            "this plan.\n\nTopic: {topic}\n\nPlan:\n{plan}"
        ),
    },
    "draft": {
        "system": (
            "You are an expert report writer. Write a well-structured, "
            "engaging report based on the provided plan and research. Use "
            "clear headings, smooth transitions, and a professional tone. "
            "The report should be approximately 500-800 words."
        ),
        "user": (
            "Write a short report on the following topic using the provided "
            "plan and research.\n\nTopic: {topic}\n\nPlan:\n{plan}\n\n"
            "Research:\n{research}"
        ),
    },
    "review": {
        "system": (
            "You are an expert editor and reviewer. Review the provided draft "
            "report for clarity, accuracy, coherence, and completeness. "
            "Provide specific improvements and then output the final, "
            "polished version of the report."
        ),
        "user": (
            "Review and improve this draft report.\n\nTopic: {topic}\n\n"
            "Draft:\n{draft}\n\nProvide your critique followed by the "
            "improved final version."
        ),
    },
}


# ---------------------------------------------------------------------------
# Node functions — one per stage
# ---------------------------------------------------------------------------

# Module-level store and config, set by run_workflow before graph execution
_store: Optional[CheckpointStore] = None
_run_config: dict = {}
_branch_label: str = "main"


def _make_node(stage_name: str):
    """Factory function that creates a graph node for a given stage.

    Each node:
        1. Builds a prompt from the stage template + current state
        2. Calls the LLM
        3. Stores the output in state
        4. Creates a checkpoint
        5. Returns the updated state

    Args:
        stage_name: One of "plan", "research", "draft", "review".

    Returns:
        A function suitable for use as a LangGraph node.
    """
    def node_fn(state: AgentState) -> dict:
        global _store, _run_config, _branch_label

        prompts = STAGE_PROMPTS[stage_name]

        # Allow config overrides for the system prompt (used in branching)
        system_prompt = _run_config.get("prompt_variant", prompts["system"])
        if _run_config.get("stage_override") != stage_name:
            # Only apply prompt_variant if this is the targeted stage
            system_prompt = prompts["system"]

        # Build the user prompt by filling in state values
        user_prompt = prompts["user"].format(
            topic=state.get("topic", ""),
            plan=state.get("plan", ""),
            research=state.get("research", ""),
            draft=state.get("draft", ""),
        )

        # Get LLM (allow model/temperature overrides from config)
        model_name = None
        temperature = 0.7
        if _run_config.get("stage_override") == stage_name:
            model_name = _run_config.get("model_name")
            temperature = _run_config.get("temperature", 0.7)

        llm = get_llm(model_name=model_name, temperature=temperature)

        # Call the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        output = response.content

        # Build the updated state
        new_messages = list(state.get("messages", []))
        new_messages.append({
            "stage": stage_name,
            "role": "assistant",
            "content": output,
        })

        # Determine parent checkpoint ID (the previous stage's checkpoint)
        checkpoint_ids = dict(state.get("checkpoint_ids", {}))
        stage_idx = STAGE_ORDER.index(stage_name)
        parent_id = None
        if stage_idx > 0:
            prev_stage = STAGE_ORDER[stage_idx - 1]
            parent_id = checkpoint_ids.get(prev_stage)

        # Build the full state to checkpoint
        updated = {
            stage_name: output,
            "messages": new_messages,
            "current_stage": stage_name,
        }

        # Merge with existing state for checkpoint storage
        full_state = {**state, **updated}

        # Create the checkpoint
        if _store is not None:
            cp_id = _store.create_checkpoint(
                parent_id=parent_id,
                stage_name=stage_name,
                state=full_state,
                config=_run_config if _run_config.get("stage_override") == stage_name else {},
                branch_label=_branch_label,
            )
            checkpoint_ids[stage_name] = cp_id
            updated["checkpoint_ids"] = checkpoint_ids

            # Print progress
            print(f"  [{stage_idx + 1}/4] {stage_name.capitalize():12s} ✓ checkpoint: {cp_id}")

        return updated

    node_fn.__name__ = f"{stage_name}_node"
    return node_fn


def build_graph() -> StateGraph:
    """Build the 4-stage LangGraph workflow.

    Returns:
        A compiled StateGraph ready to invoke.
    """
    builder = StateGraph(AgentState)

    # Add nodes for each stage
    for stage in STAGE_ORDER:
        builder.add_node(f"{stage}_node", _make_node(stage))

    # Wire edges: START → plan → research → draft → review → END
    builder.add_edge(START, "plan_node")
    builder.add_edge("plan_node", "research_node")
    builder.add_edge("research_node", "draft_node")
    builder.add_edge("draft_node", "review_node")
    builder.add_edge("review_node", END)

    return builder.compile()


def build_partial_graph(start_stage: str) -> StateGraph:
    """Build a partial graph that starts from a specific stage.

    Used for resumption after rollback — skips stages before start_stage.

    Args:
        start_stage: The stage to start from (e.g., "research" to skip "plan").

    Returns:
        A compiled StateGraph starting from the specified stage.
    """
    start_idx = STAGE_ORDER.index(start_stage)
    remaining_stages = STAGE_ORDER[start_idx:]

    builder = StateGraph(AgentState)

    # Add nodes only for remaining stages
    for stage in remaining_stages:
        builder.add_node(f"{stage}_node", _make_node(stage))

    # Wire edges
    builder.add_edge(START, f"{remaining_stages[0]}_node")
    for i in range(len(remaining_stages) - 1):
        builder.add_edge(
            f"{remaining_stages[i]}_node",
            f"{remaining_stages[i + 1]}_node",
        )
    builder.add_edge(f"{remaining_stages[-1]}_node", END)

    return builder.compile()


def build_single_stage_graph(stage_name: str) -> StateGraph:
    """Build a graph with only a single stage node.

    Used for branching — re-run just one stage with different config.

    Args:
        stage_name: The stage to run.

    Returns:
        A compiled StateGraph with a single node.
    """
    builder = StateGraph(AgentState)
    builder.add_node(f"{stage_name}_node", _make_node(stage_name))
    builder.add_edge(START, f"{stage_name}_node")
    builder.add_edge(f"{stage_name}_node", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_workflow(
    topic: str,
    store: Optional[CheckpointStore] = None,
    config: Optional[dict] = None,
    from_checkpoint_id: Optional[str] = None,
    branch_label: str = "main",
) -> dict:
    """Run the full (or partial) report-writing workflow.

    This is the main entry point for the AgentGit++ workflow.

    Args:
        topic: The report topic.
        store: CheckpointStore instance for persistence.
        config: Optional config overrides (model_name, temperature, etc.).
        from_checkpoint_id: If provided, resume from this checkpoint
                           instead of starting from scratch.
        branch_label: Label for the branch being created.

    Returns:
        The final agent state after all stages complete.
    """
    global _store, _run_config, _branch_label
    _store = store or CheckpointStore()
    _run_config = config or {}
    _branch_label = branch_label

    if from_checkpoint_id and store:
        # Resume from a checkpoint — load its state and start from the next stage
        cp = store.get_checkpoint(from_checkpoint_id)
        if cp is None:
            raise ValueError(f"Checkpoint {from_checkpoint_id} not found")

        # Determine the next stage after the checkpoint's stage
        stage_idx = STAGE_ORDER.index(cp.stage_name)
        if stage_idx >= len(STAGE_ORDER) - 1:
            print(f"  Checkpoint {from_checkpoint_id} is at the final stage. Nothing to resume.")
            return cp.state

        next_stage = STAGE_ORDER[stage_idx + 1]
        print(f"  Resuming from checkpoint {from_checkpoint_id} ({cp.stage_name})")
        print(f"  Starting from stage: {next_stage}")

        # Build a partial graph starting from the next stage
        graph = build_partial_graph(next_stage)

        # Use the checkpoint's state as initial state
        initial_state = {**cp.state}
        initial_state["current_stage"] = next_stage

        result = graph.invoke(initial_state)
        return result

    else:
        # Fresh run — start from scratch
        graph = build_graph()

        initial_state: AgentState = {
            "topic": topic,
            "plan": "",
            "research": "",
            "draft": "",
            "review": "",
            "messages": [],
            "current_stage": "",
            "checkpoint_ids": {},
        }

        result = graph.invoke(initial_state)
        return result
