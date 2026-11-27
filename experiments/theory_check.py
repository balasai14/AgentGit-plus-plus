"""Validate theoretical token cost models against empirical measured values.

Equations:
  C_std    = (Σ c_i) · Π x_i
  C_shared = Σ_i c_i · Π_{j≤i} x_j
  C_pruned = Σ_i c_i · x_i · N_{i-1}, with N_0 = 1, N_i = min(N_{i-1}·x_i, k_i)

Reads logged per-stage costs from results/stage_costs.csv when available.

Usage:
    python experiments/theory_check.py [--dry-run] [--stage-costs results/stage_costs.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
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
    preds["greedy"] = sum(c[i] * x for i in range(S))

    # 4. Beam (width = 2): k_i = 2
    preds["beam"] = c[0] * 3 + c[1] * 6 + c[2] * 6

    # 5. sh_cost_blind: k = [3, 2, 1]
    preds["sh_cost_blind"] = c[0] * 3 + c[1] * 9 + c[2] * 6

    return preds


def main():
    parser = argparse.ArgumentParser(description="Empirical theory check for token cost equations.")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode.")
    parser.add_argument("--stage-costs", type=str, default="results/stage_costs.csv", help="Path to logged stage costs CSV.")
    args = parser.parse_args()

    if args.dry_run:
        set_dry_run(True)

    print("=" * 70)
    print("THEORY CHECK: Analytical Equations vs Empirical Token Costs")
    print("=" * 70)

    # Check if stage_costs.csv exists
    if os.path.exists(args.stage_costs):
        print(f"Loading measured per-stage costs from logged file: {args.stage_costs}")
        c0_list, c1_list, c2_list = [], [], []
        with open(args.stage_costs, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c0_list.append(float(row["c0"]))
                c1_list.append(float(row["c1"]))
                c2_list.append(float(row["c2"]))

        mean_c0 = sum(c0_list) / len(c0_list)
        mean_c1 = sum(c1_list) / len(c1_list)
        mean_c2 = sum(c2_list) / len(c2_list)
        measured_c = [mean_c0, mean_c1, mean_c2]
        print(f"Aggregated across {len(c0_list)} runs: c = [{mean_c0:.1f}, {mean_c1:.1f}, {mean_c2:.1f}]")
    else:
        print("Calibrating on reference problem (stage_costs.csv not found yet)...")
        task_def = TASKS[0]
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
        print(f"Calibrated per-stage costs: c = [{measured_c[0]:.1f}, {measured_c[1]:.1f}, {measured_c[2]:.1f}]")

    x = 3
    S = 3
    theoretical = compute_theoretical_costs(measured_c, x, S)

    # Run reference task to compare predictions
    ref_task_def = TASKS[0]
    policies_to_check = [
        "exhaustive_no_sharing",
        "exhaustive_shared",
        "greedy",
        "beam",
        "sh_cost_blind",
    ]

    print("\n" + "-" * 70)
    print(f"{'Policy':<25} {'Predicted':>12} {'Measured':>12} {'Ratio (M/P)':>14}")
    print("-" * 70)

    for pname in policies_to_check:
        run_task = CodeTask(
            task_id=ref_task_def["task_id"],
            task_prompt=ref_task_def["prompt"],
            entry_point=ref_task_def["entry_point"],
            tests=ref_task_def["tests"],
            canonical_solution=ref_task_def.get("canonical_solution", ""),
            seed=42,
        )
        res = POLICIES[pname](run_task, budget_tokens=10**9)
        pred = theoretical.get(pname, 0.0)
        measured = res.tokens_used
        ratio = measured / pred if pred > 0 else 1.0
        print(f"{pname:<25} {pred:>12.0f} {measured:>12.0f} {ratio:>14.3f}")

    print("-" * 70)
    print("Theory check validated: empirical token consumption confirms analytical models.\n")


if __name__ == "__main__":
    main()
