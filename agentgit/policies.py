"""Exploration policies over the checkpoint tree.

All policies implement the common interface:
    run(task, budget_tokens) -> PolicyResult

PolicyResult can also be unpacked as a 3-tuple:
    best_leaf, best_score, tokens_used = policy(task, budget_tokens)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Any, Optional


# ---------------------------------------------------------------------------
# Task protocol
# ---------------------------------------------------------------------------

class ExplorationTask(Protocol):
    num_stages: int
    variants_per_stage: int
    measured_stage_costs: list[float]

    def expand_stage(self, parent_ids: list[str | None], stage_idx: int) -> list[Any]: ...
    def score_nodes(self, node_ids: list[str], stage_idx: int) -> dict[str, float]: ...
    def get_node(self, node_id: str) -> Any: ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    policy_name: str
    best_leaf_id: str
    best_score: float
    tokens_used: int
    leaves_explored: int
    all_leaf_scores: dict[str, float] = field(default_factory=dict)

    def __iter__(self):
        """Allow tuple unpacking: best_leaf, best_score, tokens_used = result."""
        return iter((self.best_leaf_id, self.best_score, self.tokens_used))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _best(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    best_id = max(scores, key=scores.get)
    return best_id, scores[best_id]


# ===================================================================
# 1. Exhaustive — no sharing (every path from scratch)
# ===================================================================

def exhaustive_no_sharing(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Execute all 27 paths with cost charged as if executed from scratch.

    Tokens used = (sum of per-stage costs) * 27.
    """
    x = task.variants_per_stage
    S = task.num_stages

    current_ids: list[str | None] = [None]
    for stage_idx in range(S):
        new_nodes = task.expand_stage(current_ids, stage_idx)
        current_ids = [n.id for n in new_nodes]

    leaf_ids = [cid for cid in current_ids if cid is not None]
    scores = task.score_nodes(leaf_ids, S - 1)
    best_id, best_score = _best(scores)

    # Cost: each of the x^S paths pays for all stages independently
    costs = task.measured_stage_costs
    total_paths = x ** S
    tokens_used = int(sum(costs) * total_paths)

    return PolicyResult(
        policy_name="exhaustive_no_sharing",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=scores,
    )


# ===================================================================
# 2. Exhaustive — shared prefixes (AgentGit baseline)
# ===================================================================

def exhaustive_shared(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Full tree with prefix sharing — each node generated once."""
    S = task.num_stages

    current_ids: list[str | None] = [None]
    total_tokens = 0
    for stage_idx in range(S):
        new_nodes = task.expand_stage(current_ids, stage_idx)
        total_tokens += sum(n.stage_cost_tokens for n in new_nodes)
        current_ids = [n.id for n in new_nodes]

    leaf_ids = [cid for cid in current_ids if cid is not None]
    scores = task.score_nodes(leaf_ids, S - 1)
    best_id, best_score = _best(scores)

    return PolicyResult(
        policy_name="exhaustive_shared",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=total_tokens,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=scores,
    )


# ===================================================================
# 3. Random prune
# ===================================================================

def random_prune(task: ExplorationTask, budget_tokens: int = 10**9, rng_seed: int = 42) -> PolicyResult:
    """Prune to k_i random survivors per stage based on remaining budget."""
    rng = random.Random(rng_seed)
    x = task.variants_per_stage
    S = task.num_stages
    costs = list(task.measured_stage_costs) or [100.0] * S

    current_ids: list[str | None] = [None]
    total_tokens = 0

    for stage_idx in range(S):
        new_nodes = task.expand_stage(current_ids, stage_idx)
        total_tokens += sum(n.stage_cost_tokens for n in new_nodes)
        new_ids = [n.id for n in new_nodes]

        if stage_idx < S - 1:
            remaining_budget = max(0, budget_tokens - total_tokens)
            # Estimate remaining cost per survivor
            rem_cost_per_survivor = sum(costs[j] * (x ** (j - stage_idx)) for j in range(stage_idx + 1, S))
            if rem_cost_per_survivor > 0:
                k = max(1, min(len(new_ids), int(remaining_budget / rem_cost_per_survivor)))
            else:
                k = len(new_ids)
            k = max(1, min(k, len(new_ids)))
            survivors = rng.sample(new_ids, k) if k < len(new_ids) else new_ids
            current_ids = survivors
        else:
            current_ids = new_ids

    leaf_ids = [cid for cid in current_ids if cid is not None]
    scores = task.score_nodes(leaf_ids, S - 1)
    best_id, best_score = _best(scores)

    return PolicyResult(
        policy_name="random_prune",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=total_tokens,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=scores,
    )


# ===================================================================
# 4. Greedy — keep only single best child per stage
# ===================================================================

def greedy(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Keep only the single best child per stage."""
    S = task.num_stages

    current_ids: list[str | None] = [None]
    total_tokens = 0

    for stage_idx in range(S):
        new_nodes = task.expand_stage(current_ids, stage_idx)
        total_tokens += sum(n.stage_cost_tokens for n in new_nodes)
        new_ids = [n.id for n in new_nodes]

        if stage_idx < S - 1:
            scores = task.score_nodes(new_ids, stage_idx)
            best_id = max(scores, key=scores.get)
            current_ids = [best_id]
        else:
            current_ids = new_ids

    leaf_ids = [cid for cid in current_ids if cid is not None]
    scores = task.score_nodes(leaf_ids, S - 1)
    best_id, best_score = _best(scores)

    return PolicyResult(
        policy_name="greedy",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=total_tokens,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=scores,
    )


# ===================================================================
# 5. Beam — keep top-2 per stage
# ===================================================================

def beam(task: ExplorationTask, budget_tokens: int = 10**9, beam_width: int = 2) -> PolicyResult:
    """Keep top-2 nodes by score at each stage."""
    S = task.num_stages

    current_ids: list[str | None] = [None]
    total_tokens = 0

    for stage_idx in range(S):
        new_nodes = task.expand_stage(current_ids, stage_idx)
        total_tokens += sum(n.stage_cost_tokens for n in new_nodes)
        new_ids = [n.id for n in new_nodes]

        if stage_idx < S - 1:
            scores = task.score_nodes(new_ids, stage_idx)
            ranked = sorted(scores, key=scores.get, reverse=True)
            current_ids = ranked[:beam_width]
        else:
            current_ids = new_ids

    leaf_ids = [cid for cid in current_ids if cid is not None]
    scores = task.score_nodes(leaf_ids, S - 1)
    best_id, best_score = _best(scores)

    return PolicyResult(
        policy_name="beam",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=total_tokens,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=scores,
    )


# ===================================================================
# 6a. Cost-aware successive halving (ours)
# ===================================================================

def cost_aware_sh(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Successive halving across stages with cost-aware survivor selection.

    After each stage, score nodes and keep top k_i survivors where k_i is chosen
    so that the projected remaining cost fits the budget:
        remaining_cost = Σ_{future stages} c_j · x_j · survivors
    Choose the largest k_i that fits the budget.
    """
    x = task.variants_per_stage
    S = task.num_stages
    costs = list(task.measured_stage_costs) if task.measured_stage_costs else [100.0] * S

    current_ids: list[str | None] = [None]
    total_tokens = 0

    for stage_idx in range(S):
        new_nodes = task.expand_stage(current_ids, stage_idx)
        stage_tokens = sum(n.stage_cost_tokens for n in new_nodes)
        total_tokens += stage_tokens
        new_ids = [n.id for n in new_nodes]

        # Update measured cost for this stage
        if new_nodes:
            costs[stage_idx] = stage_tokens / len(new_nodes)

        if stage_idx < S - 1:
            scores = task.score_nodes(new_ids, stage_idx)
            ranked = sorted(scores, key=scores.get, reverse=True)

            remaining_budget = max(0, budget_tokens - total_tokens)

            # Choose largest k in [1..len(ranked)] that fits projected budget
            best_k = 1
            for k in range(1, len(ranked) + 1):
                # remaining_cost = Σ_{future stages} c_j · x · survivors
                # At immediate next stage: k * x children.
                # In successive halving, each subsequent stage halves survivor count:
                projected_remaining = 0.0
                curr_survivors = k
                for j in range(stage_idx + 1, S):
                    expanded_children = curr_survivors * x
                    projected_remaining += costs[j] * expanded_children
                    curr_survivors = max(1, expanded_children // 2)

                if projected_remaining <= remaining_budget:
                    best_k = k

            current_ids = ranked[:best_k]
        else:
            current_ids = new_ids

    leaf_ids = [cid for cid in current_ids if cid is not None]
    scores = task.score_nodes(leaf_ids, S - 1)
    best_id, best_score = _best(scores)

    return PolicyResult(
        policy_name="cost_aware_sh",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=total_tokens,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=scores,
    )


# ===================================================================
# 6b. Cost-blind successive halving (ablation)
# ===================================================================

def sh_cost_blind(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Fixed successive halving: k_i = 3, 2, 1."""
    x = task.variants_per_stage
    S = task.num_stages
    keep_schedule = [3, 2, 1]

    current_ids: list[str | None] = [None]
    total_tokens = 0

    for stage_idx in range(S):
        new_nodes = task.expand_stage(current_ids, stage_idx)
        total_tokens += sum(n.stage_cost_tokens for n in new_nodes)
        new_ids = [n.id for n in new_nodes]

        if stage_idx < S - 1:
            scores = task.score_nodes(new_ids, stage_idx)
            ranked = sorted(scores, key=scores.get, reverse=True)
            k = min(keep_schedule[stage_idx], len(ranked))
            current_ids = ranked[:k]
        else:
            current_ids = new_ids

    leaf_ids = [cid for cid in current_ids if cid is not None]
    scores = task.score_nodes(leaf_ids, S - 1)
    best_id, best_score = _best(scores)

    return PolicyResult(
        policy_name="sh_cost_blind",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=total_tokens,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=scores,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POLICIES = {
    "exhaustive_no_sharing": exhaustive_no_sharing,
    "exhaustive_shared": exhaustive_shared,
    "random_prune": random_prune,
    "greedy": greedy,
    "beam": beam,
    "cost_aware_sh": cost_aware_sh,
    "sh_cost_blind": sh_cost_blind,
}
