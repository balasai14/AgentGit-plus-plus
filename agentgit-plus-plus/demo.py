"""
demo.py — End-to-end CLI demo for AgentGit++

This script demonstrates the full AgentGit++ system in action:
    1. Runs a 4-stage report-writing workflow with checkpointing
    2. Creates branches with different writing strategies
    3. Scores branches using the LLM judge
    4. Auto-selects the best branch and continues from it

Usage:
    python demo.py "The impact of artificial intelligence on healthcare"
    python demo.py  # Uses default topic

What you'll see:
    - Each stage checkpointed with a unique ID
    - A visual checkpoint tree
    - Branch creation with alternative prompts
    - Judge scores for each branch
    - Automatic selection of the best branch
"""

import sys
import os

# Add the project root to the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentgit.checkpoint_store import CheckpointStore
from agentgit.graph import run_workflow, STAGE_ORDER
from agentgit.branch import create_branch
from agentgit.judge import score_checkpoint, score_all_children
from agentgit.select import select_best_branch, auto_continue_best
from agentgit.rollback import resume_from


# ---------------------------------------------------------------------------
# Styling helpers for terminal output
# ---------------------------------------------------------------------------
def header(text: str):
    """Print a styled section header."""
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}\n")


def subheader(text: str):
    """Print a styled subsection header."""
    print(f"\n{'─' * 40}")
    print(f"  {text}")
    print(f"{'─' * 40}\n")


# ---------------------------------------------------------------------------
# Main demo flow
# ---------------------------------------------------------------------------
def main():
    # Get the topic from command line args or use default
    topic = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "The impact of artificial intelligence on modern healthcare"
    )

    header("AgentGit++ Demo")
    print(f"  Topic: {topic}")
    print(f"  Stages: {' → '.join(s.capitalize() for s in STAGE_ORDER)}")
    print()

    # Initialize the checkpoint store (fresh DB for each demo)
    db_path = "demo_checkpoints.db"
    store = CheckpointStore(db_path)
    store.clear_all()  # Start fresh

    # ──────────────────────────────────────────────────────────
    # PHASE 1: Run the full workflow (AgentGit baseline)
    # ──────────────────────────────────────────────────────────
    header("Phase 1: Running Full Workflow (AgentGit Baseline)")
    print("  Running 4-stage report-writing pipeline...\n")

    result = run_workflow(topic=topic, store=store, branch_label="main")

    subheader("Checkpoint Tree After Full Run")
    print(store.print_tree())

    # Show the final output (truncated)
    subheader("Final Report (first 500 chars)")
    final_review = result.get("review", "")
    print(f"  {final_review[:500]}...")

    # ──────────────────────────────────────────────────────────
    # PHASE 1b: Demonstrate Rollback
    # ──────────────────────────────────────────────────────────
    header("Phase 1b: Demonstrating Rollback")

    # Find the research checkpoint to rollback to
    all_cps = store.get_all_checkpoints()
    research_cp = next(
        (cp for cp in all_cps if cp.stage_name == "research" and cp.branch_label == "main"),
        None,
    )

    if research_cp:
        print(f"  Rolling back to research checkpoint: {research_cp.id}")
        print(f"  This will re-run draft and review stages.\n")

        rollback_result = resume_from(
            store, research_cp.id, branch_label="rollback-1"
        )

        subheader("Checkpoint Tree After Rollback")
        print(store.print_tree())
    else:
        print("  (Could not find research checkpoint to demo rollback)")

    # ──────────────────────────────────────────────────────────
    # PHASE 2: Create branches and score them (AgentGit++)
    # ──────────────────────────────────────────────────────────
    header("Phase 2: Branching & Scoring (AgentGit++ Intelligence)")

    # Find the draft checkpoint on the main branch to branch from
    all_cps = store.get_all_checkpoints()
    draft_cp = next(
        (cp for cp in all_cps if cp.stage_name == "draft" and cp.branch_label == "main"),
        None,
    )

    if draft_cp is None:
        print("  ERROR: Could not find draft checkpoint. Exiting.")
        return

    print(f"  Creating branches from draft checkpoint: {draft_cp.id}")
    print(f"  Each branch re-runs the draft stage with a different strategy.\n")

    # Branch 1: More academic/formal style
    branch1_id = create_branch(
        store,
        draft_cp.id,
        new_config={
            "prompt_variant": (
                "You are an expert academic writer. Write a formal, "
                "scholarly report with precise language, citations-style "
                "references, and a methodical analytical approach. Use "
                "academic tone throughout."
            ),
            "temperature": 0.5,
        },
        branch_label="academic-style",
    )

    # Branch 2: More engaging/journalistic style
    branch2_id = create_branch(
        store,
        draft_cp.id,
        new_config={
            "prompt_variant": (
                "You are a talented science journalist. Write an engaging, "
                "accessible report that tells a compelling story. Use vivid "
                "examples, analogies, and a narrative structure that draws "
                "the reader in. Make complex ideas simple and exciting."
            ),
            "temperature": 0.9,
        },
        branch_label="journalistic-style",
    )

    subheader("Checkpoint Tree With Branches")
    print(store.print_tree())

    # ──────────────────────────────────────────────────────────
    # Score all branches under the draft's parent (research checkpoint)
    # ──────────────────────────────────────────────────────────
    subheader("Scoring All Draft Branches")

    # The branches + original are all children of the research checkpoint
    parent_id = draft_cp.parent_id
    if parent_id:
        results = score_all_children(store, parent_id)

        print(f"\n  Score Summary:")
        print(f"  {'Branch':<25} {'Score':>8}")
        print(f"  {'─' * 35}")
        for r in results:
            print(f"  {r['branch_label']:<25} {r['score']:>6.1f}/10")

    # ──────────────────────────────────────────────────────────
    # Auto-select the best branch
    # ──────────────────────────────────────────────────────────
    subheader("Intelligent Branch Selection (select_best_branch)")

    if parent_id:
        best = select_best_branch(store, parent_id)
        if best:
            print(f"  ★ Best branch selected: {best.branch_label} ({best.id})")
            print(f"    Score: {best.score:.1f}/10")
            print(f"\n  The workflow can auto-continue from this branch")
            print(f"  to run the Review stage on the best draft.\n")

    # ──────────────────────────────────────────────────────────
    # Final checkpoint tree
    # ──────────────────────────────────────────────────────────
    header("Final Checkpoint Tree")
    print(store.print_tree())

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    header("Demo Complete")
    total_cps = len(store.get_all_checkpoints())
    scored_cps = len([cp for cp in store.get_all_checkpoints() if cp.score is not None])
    print(f"  Total checkpoints created:  {total_cps}")
    print(f"  Checkpoints scored:         {scored_cps}")
    print(f"  Database file:              {db_path}")
    print()
    print("  What was demonstrated:")
    print("  1. ✓ Full 4-stage workflow with automatic checkpointing")
    print("  2. ✓ Rollback to a prior checkpoint and resume")
    print("  3. ✓ Branching with alternative strategies")
    print("  4. ✓ LLM-as-judge scoring (G-Eval inspired)")
    print("  5. ✓ Intelligent branch selection")
    print()
    print("  Run the dashboard to explore interactively:")
    print("    streamlit run dashboard.py")
    print()


if __name__ == "__main__":
    main()
