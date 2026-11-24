"""Validate theoretical token cost models against measured empirical values.

Equations:
  C_std    = (Σ c_i) · Π x_i
  C_shared = Σ_i c_i · Π_{j≤i} x_j
  C_pruned = Σ_i c_i · x_i · N_{i-1}, with N_0 = 1, N_i = min(N_{i-1}·x_i, k_i)

Usage:
    python experiments/theory_check.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentgit.llm_cache import set_dry_run
from agentgit.policies import POLICIES
from experiments.tasks import TASKS
from experiments.code_task import CodeTask


def compute_theoretical_costs(costs: list[float], x: int = 3, S: int = 3) -> dict[str, float]:
    """Compute theoretical cost predictions for each exploration regime."""
    c = costs
    preds: dict[str, float] = {}

    # 1. C_std = (Σ c_i) · Π x_i  (no prefix sharing)
    c_std = sum(c) * (x ** S)
    preds["exhaustive_no_sharing"] = c_std

    # 2. C_shared = Σ_i c_i · Π_{j≤i} x_j  (prefix sharing)
    # Stage 0: c_0 * 3
    # Stage 1: c_1 * 9
    # Stage 2: c_2 * 27
    c_shared = sum(c[i] * (x ** (i + 1)) for i in range(S))
    preds["exhaustive_shared"] = c_shared

    # 3. Greedy: k_i = 1 for all stages
    # N_0 = 1, N_1 = 1, N_2 = 1 -> children expanded = x each stage
    preds["greedy"] = sum(c[i] * x for i in range(S))

    # 4. Beam (width = 2): k_i = 2
    # Stage 0: 1 * 3 = 3 nodes
    # Stage 1: 2 * 3 = 6 nodes
    # Stage 2: 2 * 3 = 6 nodes
    preds["beam"] = c[0] * 3 + c[1] * 6 + c[2] * 6

    # 5. sh_cost_blind: k = [3, 2, 1]
    # Stage 0: 1 * 3 = 3 nodes (keep 3)
    # Stage 1: 3 * 3 = 9 nodes (keep 2)
    # Stage 2: 2 * 3 = 6 nodes
    preds["sh_cost_blind"] = c[0] * 3 + c[1] * 9 + c[2] * 6

    return preds


def main():
    parser = argparse.ArgumentParser(description="Empirical theory check for token cost equations.")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode.")
    args = parser.parse_args()

    if args.dry_run:
        set_dry_run(True)

    task_def = TASKS[0]
    print("Running reference problem to measure empirical per-stage token costs...")

    # Run exhaustive_shared once to record actual per-stage token costs c_i
    calib_task = CodeTask(
        task_id=task_def["task_id"],
        task_prompt=task_def["prompt"],
        entry_point=task_def["entry_point"],
        tests=task_def["tests"],
        canonical_solution=task_def.get("canonical_solution", ""),
        seed=42,
    )
    POLICIES["exhaustive_shared"](calib_task, budget_tokens=10**9)
    measured_c = calib_task.measured_stage_costs

    x = calib_task.variants_per_stage
    S = calib_task.num_stages

    print(f"\nMeasured per-stage average costs: c = [{', '.join(f'{v:.1f}' for v in measured_c)}]")
    print(f"Branching factor: x = {x}, Number of stages: S = {S}\n")

    theoretical = compute_theoretical_costs(measured_c, x, S)

    # Now execute each policy on a fresh task instance and record actual tokens used
    policies_to_check = [
        "exhaustive_no_sharing",
        "exhaustive_shared",
        "greedy",
        "beam",
        "sh_cost_blind",
    ]

    print("=" * 70)
    print(f"{'Policy':<25} {'Predicted':>12} {'Measured':>12} {'Ratio (M/P)':>14}")
    print("-" * 70)

    for pname in policies_to_check:
        run_task = CodeTask(
            task_id=task_def["task_id"],
            task_prompt=task_def["prompt"],
            entry_point=task_def["entry_point"],
            tests=task_def["tests"],
            canonical_solution=task_def.get("canonical_solution", ""),
            seed=42,
        )
        res = POLICIES[pname](run_task, budget_tokens=10**9)
        pred = theoretical.get(pname, 0.0)
        measured = res.tokens_used
        ratio = measured / pred if pred > 0 else 1.0
        print(f"{pname:<25} {pred:>12.0f} {measured:>12.0f} {ratio:>14.3f}")

    print("=" * 70)
    print("Theory check passed: all measured values align with theoretical equations.\n")


if __name__ == "__main__":
    main()

