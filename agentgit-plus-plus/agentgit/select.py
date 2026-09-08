"""
select.py — Intelligent branch selection for AgentGit++

This module provides the decision-making layer that makes AgentGit++
"intelligent" — it automatically selects the best branch based on
LLM judge scores and can auto-continue the workflow from the winner.

Key functions:
    - select_best_branch: Pick the highest-scoring sibling branch
    - auto_continue_best: Select best + resume workflow from it
    - check_and_recover: Auto-detect failures and trigger rollback

Usage:
    from agentgit.select import select_best_branch, auto_continue_best

    best = select_best_branch(store, "parent123")
    final = auto_continue_best(store, "parent123")
"""

from typing import Optional

from agentgit.checkpoint_store import CheckpointStore, Checkpoint
from agentgit.judge import score_checkpoint, score_all_children
from agentgit.rollback import resume_from
from agentgit.graph import STAGE_ORDER


def select_best_branch(
    store: CheckpointStore,
    parent_checkpoint_id: str,
) -> Optional[Checkpoint]:
    """Select the highest-scoring branch under a parent checkpoint.

    If any branches haven't been scored yet, they are scored first.
    This is the core "intelligent selection" function of AgentGit++.

    Args:
        store: The CheckpointStore instance.
        parent_checkpoint_id: ID of the parent checkpoint whose
                              children (branches) to compare.

    Returns:
        The Checkpoint object of the best-scoring branch,
        or None if no children exist.
    """
    children = store.get_children(parent_checkpoint_id)

    if not children:
        print(f"  No branches found under checkpoint {parent_checkpoint_id}")
        return None

    # Score any unscored branches
    unscored = [c for c in children if c.score is None]
    if unscored:
        print(f"  Scoring {len(unscored)} unscored branch(es)...")
        for child in unscored:
            score_checkpoint(store, child.id)
        # Reload children with updated scores
        children = store.get_children(parent_checkpoint_id)

    # Find the highest-scoring branch
    scored_children = [c for c in children if c.score is not None]
    if not scored_children:
        print("  No scored branches available.")
        return children[0] if children else None

    best = max(scored_children, key=lambda c: c.score)

    print(f"\n═══ Branch Selection ═══")
    for child in scored_children:
        marker = " ★ BEST" if child.id == best.id else ""
        print(f"  {child.branch_label:20s} ({child.id}): {child.score:.1f}/10{marker}")
    print()

    return best


def auto_continue_best(
    store: CheckpointStore,
    parent_checkpoint_id: str,
    branch_label: str = "main",
) -> dict:
    """Select the best branch and automatically resume the workflow from it.

    This combines intelligent selection with workflow continuation:
    1. Score all branches under the parent
    2. Pick the highest-scoring one
    3. Resume the workflow from that branch's checkpoint

    Args:
        store: The CheckpointStore instance.
        parent_checkpoint_id: ID of the parent checkpoint.
        branch_label: Branch label for the continuation.

    Returns:
        The final agent state after completing the remaining stages.

    Raises:
        ValueError: If no branches exist or selection fails.
    """
    best = select_best_branch(store, parent_checkpoint_id)

    if best is None:
        raise ValueError(
            f"No branches found under checkpoint {parent_checkpoint_id}"
        )

    print(f"  Auto-continuing from best branch: {best.id} ({best.branch_label})")
    print(f"  Score: {best.score:.1f}/10")

    # Resume the workflow from the best branch
    result = resume_from(
        store,
        best.id,
        branch_label=branch_label,
    )

    return result


# ---------------------------------------------------------------------------
# Stretch goal: automatic failure detection and recovery
# ---------------------------------------------------------------------------
def check_and_recover(
    store: CheckpointStore,
    checkpoint_id: str,
    score_threshold: float = 3.0,
) -> str:
    """Check if a checkpoint's output is acceptable; auto-rollback if not.

    This is a simple failure-detection heuristic:
    1. If the output is empty → fail
    2. If the LLM judge score is below the threshold → fail
    3. On failure, roll back to the parent checkpoint

    Args:
        store: The CheckpointStore instance.
        checkpoint_id: ID of the checkpoint to check.
        score_threshold: Minimum acceptable score (default: 3.0).

    Returns:
        The checkpoint ID to continue from. This is either:
        - The same checkpoint_id (if it passed the check)
        - The parent checkpoint ID (if it failed and rolled back)
    """
    cp = store.get_checkpoint(checkpoint_id)
    if cp is None:
        raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

    # Check 1: Empty output
    stage_output = cp.state.get(cp.stage_name, "")
    if not stage_output or stage_output.strip() == "":
        print(f"  ⚠ Checkpoint {checkpoint_id}: empty output detected!")
        if cp.parent_id:
            print(f"  ↩ Auto-rolling back to parent: {cp.parent_id}")
            return cp.parent_id
        print(f"  ⚠ No parent to roll back to — this is the root.")
        return checkpoint_id

    # Check 2: Score below threshold
    score, reasoning = score_checkpoint(store, checkpoint_id)
    if score < score_threshold:
        print(f"  ⚠ Checkpoint {checkpoint_id}: score {score:.1f} < threshold {score_threshold}")
        if cp.parent_id:
            print(f"  ↩ Auto-rolling back to parent: {cp.parent_id}")
            return cp.parent_id
        print(f"  ⚠ No parent to roll back to — this is the root.")
        return checkpoint_id

    print(f"  ✓ Checkpoint {checkpoint_id} passed quality check (score: {score:.1f})")
    return checkpoint_id
