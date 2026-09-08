"""
branch.py — Create alternative branches from any checkpoint

This module implements Git-like branching for agent workflows:
re-run a stage with different settings (prompt, temperature, model)
and store the result as a sibling checkpoint under the same parent.

This is key to the AgentGit++ concept: you can explore multiple
strategies for the same stage and then use the judge to pick the best.

How it works:
    1. Load the parent checkpoint's state
    2. Re-run the SAME stage (not the next one) with a different config
    3. Store the result as a new checkpoint with the same parent_id
    4. The new checkpoint becomes a "sibling" (branch) of the original

Usage:
    from agentgit.branch import create_branch

    # Create a branch with a different writing style
    new_id = create_branch(
        store, "ghi789",
        new_config={"prompt_variant": "Write in a more academic style..."},
        branch_label="academic-style"
    )
"""

from typing import Optional

from agentgit.checkpoint_store import CheckpointStore
from agentgit.graph import (
    build_single_stage_graph,
    STAGE_ORDER,
    AgentState,
)
import agentgit.graph as graph_module


def create_branch(
    store: CheckpointStore,
    checkpoint_id: str,
    new_config: dict,
    branch_label: Optional[str] = None,
) -> str:
    """Create a new branch by re-running a stage with different config.

    The new checkpoint is stored as a sibling of the original checkpoint
    (both share the same parent_id).

    Args:
        store: The CheckpointStore instance.
        checkpoint_id: The checkpoint to branch FROM. The new branch
                       re-runs this checkpoint's stage with new settings.
        new_config: Configuration overrides. Supported keys:
            - prompt_variant (str): Alternative system prompt for the stage
            - temperature (float): Different sampling temperature
            - model_name (str): Different model to use
        branch_label: Human-readable label for the new branch.
                     Auto-generated if not provided.

    Returns:
        The checkpoint ID of the newly created branch checkpoint.

    Raises:
        ValueError: If the source checkpoint is not found.
    """
    # Load the source checkpoint
    source_cp = store.get_checkpoint(checkpoint_id)
    if source_cp is None:
        raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

    stage_name = source_cp.stage_name
    parent_id = source_cp.parent_id  # Branch shares the same parent

    # Auto-generate branch label if not provided
    if branch_label is None:
        existing_children = store.get_children(parent_id) if parent_id else []
        branch_num = len(existing_children)
        branch_label = f"branch-{branch_num}"

    print(f"\n═══ Creating Branch ═══")
    print(f"  Source checkpoint: {checkpoint_id} ({stage_name})")
    print(f"  Branch label:     {branch_label}")
    if "prompt_variant" in new_config:
        print(f"  Custom prompt:    {new_config['prompt_variant'][:60]}...")
    if "temperature" in new_config:
        print(f"  Temperature:      {new_config['temperature']}")
    if "model_name" in new_config:
        print(f"  Model:            {new_config['model_name']}")
    print()

    # Load the parent checkpoint's state (the state BEFORE this stage ran)
    # This is the input state that we want to re-run the stage on
    if parent_id:
        parent_cp = store.get_checkpoint(parent_id)
        if parent_cp is None:
            raise ValueError(f"Parent checkpoint '{parent_id}' not found.")
        base_state = parent_cp.state
    else:
        # This is a root checkpoint — use the source's state but clear its output
        base_state = {**source_cp.state}
        base_state[stage_name] = ""

    # Set up the graph module's global config for the branched run
    graph_module._store = store
    graph_module._run_config = {
        **new_config,
        "stage_override": stage_name,  # Only apply overrides to this stage
    }
    graph_module._branch_label = branch_label

    # Build and run a single-stage graph
    graph = build_single_stage_graph(stage_name)

    # Prepare initial state — carry over everything from the parent state
    initial_state: AgentState = {
        "topic": base_state.get("topic", ""),
        "plan": base_state.get("plan", ""),
        "research": base_state.get("research", ""),
        "draft": base_state.get("draft", ""),
        "review": base_state.get("review", ""),
        "messages": list(base_state.get("messages", [])),
        "current_stage": stage_name,
        "checkpoint_ids": dict(base_state.get("checkpoint_ids", {})),
    }

    # Run the stage
    result = graph.invoke(initial_state)

    # Reset global config
    graph_module._run_config = {}
    graph_module._branch_label = "main"

    # Return the new checkpoint ID
    new_cp_id = result.get("checkpoint_ids", {}).get(stage_name)
    print(f"\n  ✓ Branch created: {new_cp_id} [{branch_label}]")

    return new_cp_id


def create_multiple_branches(
    store: CheckpointStore,
    checkpoint_id: str,
    configs: list[dict],
    labels: Optional[list[str]] = None,
) -> list[str]:
    """Create multiple branches from the same checkpoint.

    Convenience function for creating several alternative versions
    of a stage's output.

    Args:
        store: The CheckpointStore instance.
        checkpoint_id: The checkpoint to branch from.
        configs: List of config dicts, one per branch.
        labels: Optional list of branch labels (auto-generated if None).

    Returns:
        List of new checkpoint IDs, one per branch.
    """
    labels = labels or [None] * len(configs)
    new_ids = []

    for config, label in zip(configs, labels):
        new_id = create_branch(store, checkpoint_id, config, label)
        new_ids.append(new_id)

    return new_ids
