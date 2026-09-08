"""
checkpoint_store.py — SQLite-backed checkpoint storage for AgentGit++

This module provides the persistence layer for the version-control system.
Each checkpoint stores the full agent state (inputs, outputs, message history)
along with metadata like stage name, parent pointer, branch label, and score.

The checkpoint tree is a DAG (directed acyclic graph) where:
- Each checkpoint has at most one parent (except the root)
- A parent can have multiple children (branches)
- Scores are attached by the LLM judge (the "++" layer)

Schema:
    checkpoints(
        id TEXT PRIMARY KEY,          -- UUID
        parent_id TEXT,               -- FK to parent checkpoint (NULL for root)
        stage_name TEXT,              -- "plan", "research", "draft", "review"
        state_json TEXT,              -- Full serialized agent state
        config_json TEXT,             -- Config/prompt variant used
        score REAL,                   -- LLM judge score (NULL until scored)
        score_reasoning TEXT,         -- Judge's chain-of-thought reasoning
        created_at TEXT,              -- ISO 8601 timestamp
        branch_label TEXT             -- Human-readable label ("main", "branch-1")
    )
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data class representing a single checkpoint
# ---------------------------------------------------------------------------
@dataclass
class Checkpoint:
    """A single checkpoint in the version-control tree.

    Attributes:
        id: Unique identifier (UUID string).
        parent_id: ID of the parent checkpoint, or None for root.
        stage_name: Which workflow stage produced this checkpoint.
        state: The full agent state dictionary (deserialized from JSON).
        config: The configuration used to run this stage.
        score: LLM judge score (None until scored).
        score_reasoning: Judge's chain-of-thought reasoning.
        created_at: ISO 8601 timestamp string.
        branch_label: Human-readable branch label.
    """
    id: str
    parent_id: Optional[str]
    stage_name: str
    state: dict
    config: dict
    score: Optional[float] = None
    score_reasoning: Optional[str] = None
    created_at: str = ""
    branch_label: str = "main"


# ---------------------------------------------------------------------------
# The checkpoint store — all database interactions go through here
# ---------------------------------------------------------------------------
class CheckpointStore:
    """SQLite-backed CRUD store for workflow checkpoints.

    Usage:
        store = CheckpointStore("checkpoints.db")
        cp_id = store.create_checkpoint(
            parent_id=None,
            stage_name="plan",
            state={"topic": "AI in healthcare", "plan": "..."},
            config={"model": "gpt-4o-mini"},
            branch_label="main"
        )
        cp = store.get_checkpoint(cp_id)
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        """Initialize the store and create the table if it doesn't exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the checkpoints table if it doesn't already exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    stage_name TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    score REAL,
                    score_reasoning TEXT,
                    created_at TEXT NOT NULL,
                    branch_label TEXT NOT NULL DEFAULT 'main',
                    FOREIGN KEY (parent_id) REFERENCES checkpoints(id)
                )
            """)
            conn.commit()

    # ----- CREATE -----

    def create_checkpoint(
        self,
        parent_id: Optional[str],
        stage_name: str,
        state: dict,
        config: Optional[dict] = None,
        branch_label: str = "main",
    ) -> str:
        """Create a new checkpoint and store it in the database.

        Args:
            parent_id: ID of the parent checkpoint (None for root).
            stage_name: Name of the workflow stage ("plan", "research", etc.).
            state: Full agent state dictionary to persist.
            config: Configuration/prompt variant used for this stage run.
            branch_label: Human-readable label for this branch.

        Returns:
            The UUID string of the newly created checkpoint.
        """
        checkpoint_id = str(uuid.uuid4())[:8]  # Short IDs for readability
        now = datetime.now(timezone.utc).isoformat()
        config = config or {}

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                    (id, parent_id, stage_name, state_json, config_json,
                     created_at, branch_label)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    parent_id,
                    stage_name,
                    json.dumps(state, default=str),
                    json.dumps(config, default=str),
                    now,
                    branch_label,
                ),
            )
            conn.commit()

        return checkpoint_id

    # ----- READ -----

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieve a single checkpoint by its ID.

        Args:
            checkpoint_id: The UUID of the checkpoint to retrieve.

        Returns:
            A Checkpoint object, or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_checkpoint(row)

    def get_children(self, parent_id: str) -> list[Checkpoint]:
        """Get all child checkpoints (branches) of a given parent.

        This is how we find sibling branches — all children of the same
        parent represent alternative outputs for the same stage.

        Args:
            parent_id: The ID of the parent checkpoint.

        Returns:
            List of Checkpoint objects that are children of the parent.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM checkpoints WHERE parent_id = ? ORDER BY created_at",
                (parent_id,),
            ).fetchall()

        return [self._row_to_checkpoint(row) for row in rows]

    def get_lineage(self, checkpoint_id: str) -> list[Checkpoint]:
        """Walk up the tree from a checkpoint to the root, returning the full lineage.

        Useful for understanding the complete history that led to a checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint to trace back from.

        Returns:
            List of Checkpoints from root to the given checkpoint (inclusive).
        """
        lineage = []
        current_id = checkpoint_id

        while current_id is not None:
            cp = self.get_checkpoint(current_id)
            if cp is None:
                break
            lineage.append(cp)
            current_id = cp.parent_id

        lineage.reverse()  # Root first
        return lineage

    def get_all_checkpoints(self) -> list[Checkpoint]:
        """Retrieve all checkpoints in the database, ordered by creation time.

        Returns:
            List of all Checkpoint objects.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM checkpoints ORDER BY created_at"
            ).fetchall()

        return [self._row_to_checkpoint(row) for row in rows]

    def get_latest_on_branch(self, branch_label: str) -> Optional[Checkpoint]:
        """Get the most recent checkpoint on a given branch.

        Args:
            branch_label: The branch label to search for.

        Returns:
            The latest Checkpoint on that branch, or None if no checkpoints exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE branch_label = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (branch_label,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_checkpoint(row)

    # ----- UPDATE -----

    def update_score(
        self, checkpoint_id: str, score: float, reasoning: str
    ) -> None:
        """Update the LLM judge score for a checkpoint.

        Called by the judge module after evaluating a checkpoint's output.

        Args:
            checkpoint_id: The ID of the checkpoint to score.
            score: The numeric score (0-10 scale).
            reasoning: The judge's chain-of-thought reasoning.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE checkpoints
                SET score = ?, score_reasoning = ?
                WHERE id = ?
                """,
                (score, reasoning, checkpoint_id),
            )
            conn.commit()

    # ----- DELETE -----

    def clear_all(self) -> None:
        """Delete all checkpoints. Useful for resetting between demo runs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM checkpoints")
            conn.commit()

    # ----- HELPERS -----

    def _row_to_checkpoint(self, row: sqlite3.Row) -> Checkpoint:
        """Convert a database row to a Checkpoint dataclass.

        Args:
            row: A sqlite3.Row from the checkpoints table.

        Returns:
            A Checkpoint object with deserialized state and config.
        """
        return Checkpoint(
            id=row["id"],
            parent_id=row["parent_id"],
            stage_name=row["stage_name"],
            state=json.loads(row["state_json"]),
            config=json.loads(row["config_json"]),
            score=row["score"],
            score_reasoning=row["score_reasoning"],
            created_at=row["created_at"],
            branch_label=row["branch_label"],
        )

    def print_tree(self) -> str:
        """Print a visual tree representation of all checkpoints.

        Returns:
            A string containing the formatted tree.
        """
        all_cps = self.get_all_checkpoints()
        if not all_cps:
            return "(empty — no checkpoints yet)"

        # Build adjacency: parent_id -> list of children
        children_map: dict[Optional[str], list[Checkpoint]] = {}
        for cp in all_cps:
            children_map.setdefault(cp.parent_id, []).append(cp)

        # Find root(s) — checkpoints with no parent
        roots = children_map.get(None, [])

        lines = []
        for root in roots:
            self._print_subtree(root, children_map, lines, prefix="", is_last=True)

        return "\n".join(lines)

    def _print_subtree(
        self,
        node: Checkpoint,
        children_map: dict,
        lines: list[str],
        prefix: str,
        is_last: bool,
    ):
        """Recursively build the tree display string.

        Args:
            node: Current checkpoint node.
            children_map: Mapping from parent_id to children list.
            lines: Accumulator list for output lines.
            prefix: Current indentation prefix.
            is_last: Whether this node is the last child of its parent.
        """
        connector = "└── " if is_last else "├── "
        score_str = f" ← score: {node.score:.1f}" if node.score is not None else ""

        # Mark the best-scored sibling with a star
        if node.parent_id and node.score is not None:
            siblings = children_map.get(node.parent_id, [])
            scored_siblings = [s for s in siblings if s.score is not None]
            if scored_siblings and node.score == max(s.score for s in scored_siblings):
                score_str += " ★ best"

        label = f"{node.stage_name} ({node.id}) [{node.branch_label}]{score_str}"
        lines.append(f"{prefix}{connector}{label}")

        # Recurse into children
        children = children_map.get(node.id, [])
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            self._print_subtree(
                child, children_map, lines, new_prefix, is_last=(i == len(children) - 1)
            )
