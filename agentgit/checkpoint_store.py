"""Checkpoint store — prefix-shared tree of LLM-generated artefacts.

Each checkpoint stores the output of one stage for one branch,
along with cost accounting fields (``stage_cost_tokens`` and
``cumulative_cost_tokens``).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Checkpoint:
    """A single node in the exploration tree."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    stage: int = 0
    stage_name: str = ""
    variant: int = 0
    variant_label: str = ""
    content: str = ""
    score: Optional[float] = None
    # --- cost accounting (additive, migration-safe) ---
    stage_cost_tokens: int = 0
    cumulative_cost_tokens: int = 0
    metadata: dict = field(default_factory=dict)


class CheckpointTree:
    """In-memory tree of checkpoints with prefix sharing."""

    def __init__(self) -> None:
        self.nodes: dict[str, Checkpoint] = {}
        self.children: dict[str, list[str]] = {}
        self.root_ids: list[str] = []

    # ---- mutators ----

    def add(self, cp: Checkpoint) -> Checkpoint:
        self.nodes[cp.id] = cp
        if cp.parent_id is None:
            self.root_ids.append(cp.id)
        else:
            self.children.setdefault(cp.parent_id, []).append(cp.id)
        return cp

    # ---- queries ----

    def get(self, node_id: str) -> Checkpoint:
        return self.nodes[node_id]

    def get_children(self, node_id: str) -> list[Checkpoint]:
        return [self.nodes[cid] for cid in self.children.get(node_id, [])]

    def leaves(self) -> list[Checkpoint]:
        parent_set = {pid for pid, kids in self.children.items() if kids}
        return [n for n in self.nodes.values() if n.id not in parent_set]

    def path_to_root(self, node_id: str) -> list[Checkpoint]:
        path: list[Checkpoint] = []
        cur: Optional[str] = node_id
        while cur is not None:
            cp = self.nodes[cur]
            path.append(cp)
            cur = cp.parent_id
        return list(reversed(path))

    def path_content(self, node_id: str) -> str:
        """Concatenate content along the root-to-node path (prefix sharing)."""
        return "\n\n---\n\n".join(cp.content for cp in self.path_to_root(node_id))

    def total_cost(self) -> int:
        return sum(n.stage_cost_tokens for n in self.nodes.values())

    # ---- serialisation ----

    def to_dict(self) -> dict:
        return {
            "nodes": {
                k: {
                    "id": v.id,
                    "parent_id": v.parent_id,
                    "stage": v.stage,
                    "stage_name": v.stage_name,
                    "variant": v.variant,
                    "variant_label": v.variant_label,
                    "content": v.content,
                    "score": v.score,
                    "stage_cost_tokens": v.stage_cost_tokens,
                    "cumulative_cost_tokens": v.cumulative_cost_tokens,
                    "metadata": v.metadata,
                }
                for k, v in self.nodes.items()
            },
            "children": self.children,
            "root_ids": self.root_ids,
        }

