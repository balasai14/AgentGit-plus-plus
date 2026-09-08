"""
rollback.py — Rollback and resume from any checkpoint

This module implements the "version control" aspect of AgentGit:
instead of restarting the entire workflow on failure, you can roll back
to any prior checkpoint and resume from there.

How it works:
    1. Load the full agent state from the target checkpoint
    2. Determine which stage comes next
    3. Build a partial LangGraph workflow starting from that stage
    4. Execute the remaining stages with the loaded state

Usage:
    from agentgit.rollback import rollback, resume_from

    # Just load the state (no re-execution)
    state = rollback(store, "abc123")

    # Load and immediately resume the workflow
    final_state = resume_from(store, "abc123")
"""

from typing import Optional

from agentgit.checkpoint_store import CheckpointStore, Checkpoint
from agentgit.graph import run_workflow, STAGE_ORDER


def rollback(store: CheckpointStore, checkpoint_id: str) -> dict:
    """Load the full state from a checkpoint without re-executing.

    This is the "git checkout" equivalent — it retrieves the saved state
    so you can inspect it or use it as a starting point.

    Args:
        store: The CheckpointStore instance.
        checkpoint_id: ID of the checkpoint to roll back to.

    Returns:
        The full agent state dictionary from that checkpoint.

    Raises:
        ValueError: If the checkpoint ID is not found.
    """
    cp = store.get_checkpoint(checkpoint_id)
    if cp is None:
        raise ValueError(f"Checkpoint '{checkpoint_id}' not found in store.")

    print(f"  ↩ Rolled back to checkpoint {checkpoint_id}")
    print(f"    Stage: {cp.stage_name} | Branch: {cp.branch_label}")
    print(f"    Created: {cp.created_at}")

    return cp.state


def resume_from(
    store: CheckpointStore,
    checkpoint_id: str,
    config: Optional[dict] = None,
    branch_label: str = "main",
) -> dict:
    """Roll back to a checkpoint and resume the workflow from the next stage.

    This is the most common rollback operation: go back to a known good
    state and re-run everything from there forward.

    Args:
        store: The CheckpointStore instance.
        checkpoint_id: ID of the checkpoint to resume from.
        config: Optional config overrides for the resumed stages.
        branch_label: Branch label for the new checkpoints.

    Returns:
        The final agent state after all remaining stages complete.

    Raises:
        ValueError: If the checkpoint ID is not found.
    """
    cp = store.get_checkpoint(checkpoint_id)
    if cp is None:
        raise ValueError(f"Checkpoint '{checkpoint_id}' not found in store.")

    # Check if we're at the last stage
    stage_idx = STAGE_ORDER.index(cp.stage_name)
    if stage_idx >= len(STAGE_ORDER) - 1:
        print(f"  Checkpoint {checkpoint_id} is at the final stage ('{cp.stage_name}').")
        print(f"  Nothing to resume — returning stored state.")
        return cp.state

    next_stage = STAGE_ORDER[stage_idx + 1]
    print(f"\n═══ Resuming Workflow ═══")
    print(f"  Rolling back to: {cp.stage_name} ({checkpoint_id})")
    print(f"  Resuming from:   {next_stage}")
    print()

    # Use the graph module's run_workflow with from_checkpoint_id
    result = run_workflow(
        topic=cp.state.get("topic", ""),
        store=store,
        config=config,
        from_checkpoint_id=checkpoint_id,
        branch_label=branch_label,
    )

    return result


def get_rollback_options(store: CheckpointStore) -> list[dict]:
    """List all available checkpoints that can be rolled back to.

    Convenience function for the dashboard — returns a simplified list
    of checkpoints with just the info needed for a selection UI.

    Args:
        store: The CheckpointStore instance.

    Returns:
        List of dicts with keys: id, stage_name, branch_label, created_at, score.
    """
    checkpoints = store.get_all_checkpoints()
    return [
        {
            "id": cp.id,
            "stage_name": cp.stage_name,
            "branch_label": cp.branch_label,
            "created_at": cp.created_at,
            "score": cp.score,
        }
        for cp in checkpoints
    ]
