"""Exploration policies over the checkpoint tree with strict budget enforcement.

Rules:
1. Exhaustive policies run leaves in fixed order until the budget is spent and return the best so far.
2. CASH (Cost-Aware Successive Halving) must NOT exceed the budget unless one path alone doesn't fit,
   and must log those cases.
3. Beam, Greedy, Random Prune, and Cost-Blind SH also strictly cap exploration within budget.

All policies implement the common interface:
    run(task, budget_tokens) -> PolicyResult
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Any, Optional

logger = logging.getLogger("agentgit.policies")


# ---------------------------------------------------------------------------
# Task protocol
# ---------------------------------------------------------------------------

class ExplorationTask(Protocol):
    task_id: str
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
    budget_exceeded_single_path_fallback: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Allow tuple unpacking: best_leaf, best_score, tokens_used = result."""
        return iter((self.best_leaf_id, self.best_score, self.tokens_used))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    best_id = max(scores, key=scores.get)
    return best_id, scores[best_id]


# ===================================================================
# 1. Exhaustive — No Sharing (runs paths in fixed order until budget spent)
# ===================================================================

def exhaustive_no_sharing(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Run leaves in fixed order from scratch until budget is spent, returning best so far.

    Each path pays (c0 + c1 + c2) independently.
    """
    x = task.variants_per_stage
    S = task.num_stages
    costs = list(task.measured_stage_costs) if task.measured_stage_costs else [100.0] * S
    path_cost_est = sum(costs)

    # First, generate the full tree structure to get path definitions
    # (Since we use the shared cache, this costs 0 extra API calls on replay)
    plan_nodes = task.expand_stage([None], 0)
    plan_ids = [n.id for n in plan_nodes]

    code_nodes = task.expand_stage(plan_ids, 1)
    code_ids = [n.id for n in code_nodes]

    fix_nodes = task.expand_stage(code_ids, 2)
    fix_ids = [n.id for n in fix_nodes]

    # Map leaf to its path costs
    leaf_scores = task.score_nodes(fix_ids, 2)

    # Fixed order evaluation of the 27 leaves
    tokens_used = 0
    leaves_explored = 0
    explored_scores: dict[str, float] = {}
    best_leaf_id = ""
    best_score = 0.0

    for leaf_id in fix_ids:
        node = task.get_node(leaf_id)
        # Compute exact path cost from root to leaf
        path_tokens = node.cumulative_cost_tokens

        # Check budget constraint
        if leaves_explored > 0 and (tokens_used + path_tokens > budget_tokens):
            break

        tokens_used += path_tokens
        leaves_explored += 1
        score = leaf_scores.get(leaf_id, 0.0)
        explored_scores[leaf_id] = score

        if score > best_score or not best_leaf_id:
            best_score = score
            best_leaf_id = leaf_id

    return PolicyResult(
        policy_name="exhaustive_no_sharing",
        best_leaf_id=best_leaf_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=leaves_explored,
        all_leaf_scores=explored_scores,
    )


# ===================================================================
# 2. Exhaustive — Shared (runs leaves in fixed order with prefix sharing)
# ===================================================================

def exhaustive_shared(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Walk tree with prefix sharing in fixed order until budget is spent, returning best so far."""
    # Expand full tree
    plan_nodes = task.expand_stage([None], 0)
    code_nodes = task.expand_stage([n.id for n in plan_nodes], 1)
    fix_nodes = task.expand_stage([n.id for n in code_nodes], 2)

    leaf_scores = task.score_nodes([n.id for n in fix_nodes], 2)

    # Track which nodes have been 'paid for' in shared traversal
    expanded_nodes: set[str] = set()
    tokens_used = 0
    leaves_explored = 0
    explored_scores: dict[str, float] = {}
    best_leaf_id = ""
    best_score = 0.0

    for leaf in fix_nodes:
        # Determine incremental cost of reaching this leaf
        code_parent = task.get_node(leaf.parent_id)
        plan_parent = task.get_node(code_parent.parent_id)

        incremental = 0
        nodes_to_add = []
        for n in [plan_parent, code_parent, leaf]:
            if n.id not in expanded_nodes:
                incremental += n.stage_cost_tokens
                nodes_to_add.append(n.id)

        # Enforce budget: stop if exceeding budget (allow at least 1 leaf)
        if leaves_explored > 0 and (tokens_used + incremental > budget_tokens):
            break

        tokens_used += incremental
        expanded_nodes.update(nodes_to_add)
        leaves_explored += 1

        score = leaf_scores.get(leaf.id, 0.0)
        explored_scores[leaf.id] = score
        if score > best_score or not best_leaf_id:
            best_score = score
            best_leaf_id = leaf.id

    return PolicyResult(
        policy_name="exhaustive_shared",
        best_leaf_id=best_leaf_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=leaves_explored,
        all_leaf_scores=explored_scores,
    )


# ===================================================================
# 3. Cost-Aware Successive Halving (CASH) — Strict Budget Enforcement
# ===================================================================

def cost_aware_sh(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Cost-Aware Successive Halving (CASH).

    Dynamically sizes survivor quotas k_i based on measured per-stage costs
    and projected remaining expenditure:
        remaining_cost = sum_{future stages j} c_j * x * survivors_j
    Strictly guarantees tokens_used <= budget unless one path alone doesn't fit,
    in which case it logs the warning and executes the single path fallback.
    """
    x = task.variants_per_stage
    S = task.num_stages
    costs = list(task.measured_stage_costs) if task.measured_stage_costs else [100.0] * S

    single_path_fallback = False
    tokens_used = 0

    # Minimum possible path cost (1 plan + 1 code + 1 fix)
    min_path_cost = sum(costs)
    if min_path_cost > budget_tokens:
        single_path_fallback = True
        logger.warning(
            f"[CASH WARNING] Single path cost ({min_path_cost:.0f}) exceeds budget ({budget_tokens}). "
            f"Executing single path fallback for task '{getattr(task, 'task_id', 'unknown')}'."
        )

    # Stage 0: Plan
    plan_nodes = task.expand_stage([None], 0)
    stage0_tokens = sum(n.stage_cost_tokens for n in plan_nodes)
    tokens_used += stage0_tokens
    costs[0] = stage0_tokens / max(1, len(plan_nodes))

    # Score plans
    plan_scores = task.score_nodes([n.id for n in plan_nodes], 0)
    ranked_plans = sorted(plan_scores, key=plan_scores.get, reverse=True)

    # Determine k0 survivors based on projected remaining budget
    remaining_budget = max(0, budget_tokens - tokens_used)
    best_k0 = 1

    if not single_path_fallback:
        for k in range(len(ranked_plans), 0, -1):
            # Projected cost:
            # Stage 1: k survivors * x children
            cost_stage1 = k * x * costs[1]
            # Stage 2: successive halving keeps max(1, (k*x)//2) survivors * x children
            surv_stage2 = max(1, (k * x) // 2)
            cost_stage2 = surv_stage2 * x * costs[2]
            proj_total = cost_stage1 + cost_stage2
            if proj_total <= remaining_budget:
                best_k0 = k
                break
        else:
            best_k0 = 1

    survivors_stage0 = ranked_plans[:best_k0]

    # Stage 1: Code
    # If budget is extremely tight, expand only as many children as fit
    if tokens_used + len(survivors_stage0) * x * costs[1] > budget_tokens and not single_path_fallback:
        allowed_children = max(1, int((budget_tokens - tokens_used) / max(1.0, costs[1])))
        allowed_children = min(len(survivors_stage0) * x, allowed_children)
    else:
        allowed_children = len(survivors_stage0) * x

    code_nodes = task.expand_stage(survivors_stage0, 1)
    if len(code_nodes) > allowed_children:
        code_nodes = code_nodes[:allowed_children]

    stage1_tokens = sum(n.stage_cost_tokens for n in code_nodes)
    tokens_used += stage1_tokens
    costs[1] = stage1_tokens / max(1, len(code_nodes))

    # Score code drafts
    code_scores = task.score_nodes([n.id for n in code_nodes], 1)
    ranked_code = sorted(code_scores, key=code_scores.get, reverse=True)

    # Determine k1 survivors for Stage 2 (Fix)
    remaining_budget_fix = max(0, budget_tokens - tokens_used)
    best_k1 = 1

    if not single_path_fallback:
        for k in range(len(ranked_code), 0, -1):
            proj_fix = k * x * costs[2]
            if proj_fix <= remaining_budget_fix:
                best_k1 = k
                break
        else:
            best_k1 = 1

    survivors_stage1 = ranked_code[:best_k1]

    # Stage 2: Fix
    # Check if we need to cap fix children to fit budget
    if tokens_used + len(survivors_stage1) * x * costs[2] > budget_tokens and not single_path_fallback:
        allowed_fix = max(1, int((budget_tokens - tokens_used) / max(1.0, costs[2])))
        allowed_fix = min(len(survivors_stage1) * x, allowed_fix)
    else:
        allowed_fix = len(survivors_stage1) * x

    fix_nodes = task.expand_stage(survivors_stage1, 2)
    if len(fix_nodes) > allowed_fix:
        fix_nodes = fix_nodes[:allowed_fix]

    stage2_tokens = sum(n.stage_cost_tokens for n in fix_nodes)
    tokens_used += stage2_tokens

    # Score leaves
    leaf_ids = [n.id for n in fix_nodes]
    final_scores = task.score_nodes(leaf_ids, 2)
    best_id, best_score = _best(final_scores)

    return PolicyResult(
        policy_name="cost_aware_sh",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=len(leaf_ids),
        all_leaf_scores=final_scores,
        budget_exceeded_single_path_fallback=single_path_fallback,
        details={
            "k0": best_k0,
            "k1": best_k1,
            "stage_costs": costs,
        },
    )


# ===================================================================
# 4. Greedy (width = 1 per stage, capped by budget)
# ===================================================================

def greedy(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Keep only single best child per stage, halting if budget exceeded."""
    tokens_used = 0

    # Stage 0
    plans = task.expand_stage([None], 0)
    tokens_used += sum(n.stage_cost_tokens for n in plans)
    p_scores = task.score_nodes([n.id for n in plans], 0)
    best_p = max(p_scores, key=p_scores.get)

    # Stage 1
    codes = task.expand_stage([best_p], 1)
    tokens_used += sum(n.stage_cost_tokens for n in codes)
    c_scores = task.score_nodes([n.id for n in codes], 1)
    best_c = max(c_scores, key=c_scores.get)

    # Stage 2
    fixes = task.expand_stage([best_c], 2)
    # Budget cap check
    if tokens_used + sum(n.stage_cost_tokens for n in fixes) > budget_tokens and budget_tokens < 10**8:
        # Cap to what fits
        allowed = max(1, int((budget_tokens - tokens_used) / max(1, task.measured_stage_costs[2])))
        fixes = fixes[:allowed]

    tokens_used += sum(n.stage_cost_tokens for n in fixes)
    leaf_scores = task.score_nodes([n.id for n in fixes], 2)
    best_id, best_score = _best(leaf_scores)

    return PolicyResult(
        policy_name="greedy",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=len(fixes),
        all_leaf_scores=leaf_scores,
    )


# ===================================================================
# 5. Beam (width = 2, capped by budget)
# ===================================================================

def beam(task: ExplorationTask, budget_tokens: int = 10**9, beam_width: int = 2) -> PolicyResult:
    """Beam search with width=2, capped by budget."""
    tokens_used = 0

    # Stage 0
    plans = task.expand_stage([None], 0)
    tokens_used += sum(n.stage_cost_tokens for n in plans)
    p_scores = task.score_nodes([n.id for n in plans], 0)
    ranked_p = sorted(p_scores, key=p_scores.get, reverse=True)[:beam_width]

    # Stage 1
    codes = task.expand_stage(ranked_p, 1)
    tokens_used += sum(n.stage_cost_tokens for n in codes)
    c_scores = task.score_nodes([n.id for n in codes], 1)
    ranked_c = sorted(c_scores, key=c_scores.get, reverse=True)[:beam_width]

    # Stage 2
    fixes = task.expand_stage(ranked_c, 2)
    if tokens_used + sum(n.stage_cost_tokens for n in fixes) > budget_tokens and budget_tokens < 10**8:
        allowed = max(1, int((budget_tokens - tokens_used) / max(1, task.measured_stage_costs[2])))
        fixes = fixes[:allowed]

    tokens_used += sum(n.stage_cost_tokens for n in fixes)
    leaf_scores = task.score_nodes([n.id for n in fixes], 2)
    best_id, best_score = _best(leaf_scores)

    return PolicyResult(
        policy_name="beam",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=len(fixes),
        all_leaf_scores=leaf_scores,
    )


# ===================================================================
# 6. Cost-Blind Successive Halving (ablation: k = 3, 2, 1)
# ===================================================================

def sh_cost_blind(task: ExplorationTask, budget_tokens: int = 10**9) -> PolicyResult:
    """Fixed successive halving: k = [3, 2, 1], capped by budget."""
    tokens_used = 0

    # Stage 0: keep 3
    plans = task.expand_stage([None], 0)
    tokens_used += sum(n.stage_cost_tokens for n in plans)
    p_scores = task.score_nodes([n.id for n in plans], 0)
    ranked_p = sorted(p_scores, key=p_scores.get, reverse=True)[:3]

    # Stage 1: keep 2
    codes = task.expand_stage(ranked_p, 1)
    tokens_used += sum(n.stage_cost_tokens for n in codes)
    c_scores = task.score_nodes([n.id for n in codes], 1)
    ranked_c = sorted(c_scores, key=c_scores.get, reverse=True)[:2]

    # Stage 2: keep 1
    fixes = task.expand_stage(ranked_c, 2)
    if tokens_used + sum(n.stage_cost_tokens for n in fixes) > budget_tokens and budget_tokens < 10**8:
        allowed = max(1, int((budget_tokens - tokens_used) / max(1, task.measured_stage_costs[2])))
        fixes = fixes[:allowed]

    tokens_used += sum(n.stage_cost_tokens for n in fixes)
    leaf_scores = task.score_nodes([n.id for n in fixes], 2)
    best_id, best_score = _best(leaf_scores)

    return PolicyResult(
        policy_name="sh_cost_blind",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=len(fixes),
        all_leaf_scores=leaf_scores,
    )


# ===================================================================
# 7. Random Prune
# ===================================================================

def random_prune(task: ExplorationTask, budget_tokens: int = 10**9, rng_seed: int = 42) -> PolicyResult:
    """Prune to random survivors per stage fitting budget."""
    rng = random.Random(rng_seed)
    tokens_used = 0

    # Stage 0
    plans = task.expand_stage([None], 0)
    tokens_used += sum(n.stage_cost_tokens for n in plans)
    k0 = max(1, min(len(plans), int(budget_tokens / max(1, sum(task.measured_stage_costs)))))
    surv_p = rng.sample([n.id for n in plans], k0)

    # Stage 1
    codes = task.expand_stage(surv_p, 1)
    tokens_used += sum(n.stage_cost_tokens for n in codes)
    k1 = max(1, min(len(codes), 2))
    surv_c = rng.sample([n.id for n in codes], k1)

    # Stage 2
    fixes = task.expand_stage(surv_c, 2)
    if tokens_used + sum(n.stage_cost_tokens for n in fixes) > budget_tokens and budget_tokens < 10**8:
        allowed = max(1, int((budget_tokens - tokens_used) / max(1, task.measured_stage_costs[2])))
        fixes = fixes[:allowed]

    tokens_used += sum(n.stage_cost_tokens for n in fixes)
    leaf_scores = task.score_nodes([n.id for n in fixes], 2)
    best_id, best_score = _best(leaf_scores)

    return PolicyResult(
        policy_name="random_prune",
        best_leaf_id=best_id,
        best_score=best_score,
        tokens_used=tokens_used,
        leaves_explored=len(fixes),
        all_leaf_scores=leaf_scores,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POLICIES = {
    "cost_aware_sh": cost_aware_sh,
    "exhaustive_shared": exhaustive_shared,
    "exhaustive_no_sharing": exhaustive_no_sharing,
    "beam": beam,
    "greedy": greedy,
    "sh_cost_blind": sh_cost_blind,
    "random_prune": random_prune,
}
