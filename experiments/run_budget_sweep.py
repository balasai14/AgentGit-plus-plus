"""Budget sweep experiment runner with strict budget enforcement and leaf-level logging.

Budgets are expressed as 10%, 20%, 30%, 50%, and 100% of the exhaustive_shared cost.
Outputs:
  - results/results.csv: summary per task × seed × policy × budget
  - results/leaves.csv: all 27 leaves per task × seed with per-stage token costs
  - results/stage_costs.csv: per-stage costs (c0, c1, c2) for theory validation

Usage:
    python experiments/run_budget_sweep.py [--dry-run] [--problems N] [--seeds N]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentgit.llm_cache import set_dry_run, is_dry_run
from agentgit.policies import POLICIES
from experiments.tasks import TASKS
from experiments.code_task import CodeTask, VARIANT_LABELS, VARIANT_TEMPERATURES

BUDGET_FRACTIONS = [0.10, 0.20, 0.30, 0.50, 1.00]
DEFAULT_SEEDS = [42, 123, 456]


def estimate_cost(num_problems: int, num_seeds: int) -> None:
    """Print estimated tokens and dollar cost before live API calls."""
    total_trees = num_problems * num_seeds
    calls_per_tree = 3 + 9 + 27  # 39 generation calls
    judge_calls = 3              # 3 plan judge calls
    total_calls = total_trees * (calls_per_tree + judge_calls)
    est_tokens = total_calls * 300
    est_usd = (est_tokens / 1_000_000) * 0.30

    print("=" * 65)
    print("API SPEND ESTIMATION (OpenAI gpt-4o-mini)")
    print("=" * 65)
    print(f"Number of problems:     {num_problems}")
    print(f"Number of seeds:        {num_seeds}")
    print(f"Total trees evaluated:  {total_trees}")
    print(f"Total API calls:        ~{total_calls:,}")
    print(f"Estimated token volume: ~{est_tokens:,}")
    print(f"Estimated total spend:  ~${est_usd:.4f} USD")
    print("Trees are executed once per problem × seed and fully cached.")
    print("All subsequent policy and budget sweeps replay for $0.00.")
    print("=" * 65)


def run_sweep(
    num_problems: int = 40,
    seeds: list[int] | None = None,
    results_path: str = "results/results.csv",
    leaves_path: str = "results/leaves.csv",
    stage_costs_path: str = "results/stage_costs.csv",
) -> tuple[list[dict], list[dict], list[dict]]:
    seeds = seeds or DEFAULT_SEEDS
    tasks = TASKS[:num_problems]

    os.makedirs(os.path.dirname(results_path) or "results", exist_ok=True)

    all_results: list[dict] = []
    all_leaves: list[dict] = []
    all_stage_costs: list[dict] = []

    total_evals = len(seeds) * len(tasks) * len(POLICIES) * len(BUDGET_FRACTIONS)
    print(f"\nRunning Budget Sweep:")
    print(f"  {len(tasks)} problems * {len(seeds)} seeds * {len(POLICIES)} policies * {len(BUDGET_FRACTIONS)} budgets")
    print(f"  Budgets: {[f'{int(b*100)}%' for b in BUDGET_FRACTIONS]} of exhaustive_shared cost")
    print(f"  Total policy evaluations: {total_evals:,}\n")

    for seed_idx, seed in enumerate(seeds):
        print(f"--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")

        for t_idx, task_def in enumerate(tasks):
            t_id = task_def["task_id"]

            # -----------------------------------------------------------------
            # 1. Warm full tree via exhaustive_shared (oracle reference)
            # -----------------------------------------------------------------
            base_task = CodeTask(
                task_id=task_def["task_id"],
                task_prompt=task_def["prompt"],
                entry_point=task_def["entry_point"],
                tests=task_def["tests"],
                canonical_solution=task_def.get("canonical_solution", ""),
                seed=seed,
            )
            # Run without budget limit to discover oracle and build tree
            oracle_res = POLICIES["exhaustive_shared"](base_task, budget_tokens=10**9)
            oracle_score = oracle_res.best_score

            # Full shared tree cost is our 100% budget reference
            full_shared_cost = oracle_res.tokens_used
            measured_c = base_task.measured_stage_costs

            # Log per-stage costs for theory check
            stage_cost_row = {
                "task_id": t_id,
                "seed": seed,
                "c0": round(measured_c[0], 2),
                "c1": round(measured_c[1], 2),
                "c2": round(measured_c[2], 2),
                "full_shared_cost": full_shared_cost,
                "full_unshared_cost": int(sum(measured_c) * 27),
            }
            all_stage_costs.append(stage_cost_row)

            # -----------------------------------------------------------------
            # 2. Record leaf-level results to leaves.csv
            # -----------------------------------------------------------------
            tree = base_task.tree
            leaves = tree.leaves()
            for leaf in leaves:
                path_nodes = tree.path_to_root(leaf.id)
                # Format leaf path (e.g. p0_c1_f2)
                path_labels = [f"s{n.stage}_v{n.variant}" for n in path_nodes]
                leaf_path_str = "_".join(path_labels)

                c0_tok = path_nodes[0].stage_cost_tokens if len(path_nodes) > 0 else 0
                c1_tok = path_nodes[1].stage_cost_tokens if len(path_nodes) > 1 else 0
                c2_tok = path_nodes[2].stage_cost_tokens if len(path_nodes) > 2 else 0

                leaf_row = {
                    "task_id": t_id,
                    "seed": seed,
                    "leaf_id": leaf.id,
                    "leaf_path": leaf_path_str,
                    "score": round(leaf.score or 0.0, 4),
                    "is_oracle": int((leaf.score or 0.0) >= oracle_score - 1e-9),
                    "stage0_tokens": c0_tok,
                    "stage1_tokens": c1_tok,
                    "stage2_tokens": c2_tok,
                    "total_tokens": leaf.cumulative_cost_tokens,
                }
                all_leaves.append(leaf_row)

            print(
                f"  [{t_idx + 1:02d}/{len(tasks):02d}] {t_id:<30} | "
                f"Oracle: {oracle_score:.2f} | Shared Cost: {full_shared_cost} tok | "
                f"Stages: [{measured_c[0]:.0f}, {measured_c[1]:.0f}, {measured_c[2]:.0f}]"
            )

            # -----------------------------------------------------------------
            # 3. Evaluate each policy at budgets {10%, 20%, 30%, 50%, 100%}
            # -----------------------------------------------------------------
            for policy_name, policy_fn in POLICIES.items():
                for b_frac in BUDGET_FRACTIONS:
                    budget = max(1, int(full_shared_cost * b_frac))

                    ptask = CodeTask(
                        task_id=task_def["task_id"],
                        task_prompt=task_def["prompt"],
                        entry_point=task_def["entry_point"],
                        tests=task_def["tests"],
                        canonical_solution=task_def.get("canonical_solution", ""),
                        seed=seed,
                    )

                    if policy_name == "random_prune":
                        res = policy_fn(ptask, budget_tokens=budget, rng_seed=seed)
                    else:
                        res = policy_fn(ptask, budget_tokens=budget)

                    oracle_hit = int(res.best_score >= oracle_score - 1e-9)
                    regret = max(0.0, round(oracle_score - res.best_score, 4))

                    row = {
                        "seed": seed,
                        "task_id": t_id,
                        "policy": res.policy_name,
                        "budget_frac": b_frac,
                        "budget_tokens": budget,
                        "tokens_used": res.tokens_used,
                        "best_score": res.best_score,
                        "oracle_score": oracle_score,
                        "regret": regret,
                        "leaves_explored": res.leaves_explored,
                        "oracle_hit": oracle_hit,
                        "single_path_fallback": int(res.budget_exceeded_single_path_fallback),
                    }
                    all_results.append(row)

    # Save results.csv
    if all_results:
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)

    # Save leaves.csv
    if all_leaves:
        with open(leaves_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_leaves[0].keys()))
            w.writeheader()
            w.writerows(all_leaves)

    # Save stage_costs.csv
    if all_stage_costs:
        with open(stage_costs_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_stage_costs[0].keys()))
            w.writeheader()
            w.writerows(all_stage_costs)

    return all_results, all_leaves, all_stage_costs


def main():
    parser = argparse.ArgumentParser(description="Run budget sweep experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock LLM ($0.00 spend).")
    parser.add_argument("--problems", type=int, default=40, help="Number of benchmark problems (default: 40).")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (default: 3).")
    parser.add_argument("--output", type=str, default="results/results.csv", help="Output summary CSV.")
    parser.add_argument("--leaves", type=str, default="results/leaves.csv", help="Output leaves CSV.")
    parser.add_argument("--stage-costs", type=str, default="results/stage_costs.csv", help="Output stage costs CSV.")
    args = parser.parse_args()

    if args.dry_run:
        set_dry_run(True)
        print("\n*** RUNNING IN DRY-RUN MODE (Deterministic mock LLM, $0 spend) ***\n")
    else:
        estimate_cost(args.problems, args.seeds)
        confirm = input("Proceed with live OpenAI API calls? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    seeds = DEFAULT_SEEDS[:max(1, min(len(DEFAULT_SEEDS), args.seeds))]
    t0 = time.time()
    results, leaves, stage_costs = run_sweep(
        num_problems=args.problems,
        seeds=seeds,
        results_path=args.output,
        leaves_path=args.leaves,
        stage_costs_path=args.stage_costs,
    )
    elapsed = time.time() - t0

    print(f"\nSweep Completed Successfully in {elapsed:.2f}s:")
    print(f"  - Summary rows: {len(results)} saved to {args.output}")
    print(f"  - Leaves:       {len(leaves)} saved to {args.leaves}")
    print(f"  - Stage costs:  {len(stage_costs)} saved to {args.stage_costs}")


if __name__ == "__main__":
    main()
