"""Budget sweep: for each problem × policy × budget fraction, record exploration metrics.

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
from experiments.code_task import CodeTask

BUDGET_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
DEFAULT_SEEDS = [42, 123, 456]


def estimate_cost(num_problems: int) -> None:
    """Print rough token and dollar estimate before live execution."""
    # Full tree per problem = 3 + 9 + 27 = 39 calls
    # Judge calls for plans = 3 calls
    # Total calls per problem = 42
    total_calls = 42 * num_problems
    avg_tokens = 250
    total_tokens = total_calls * avg_tokens
    # gpt-4o-mini is ~$0.15 / 1M input, $0.60 / 1M output (~$0.30 / 1M blended)
    est_usd = (total_tokens / 1_000_000) * 0.30

    print("=" * 60)
    print("API SPEND ESTIMATION")
    print("=" * 60)
    print(f"Number of problems:     {num_problems}")
    print(f"Total API calls:        ~{total_calls}")
    print(f"Total tokens estimated: ~{total_tokens:,}")
    print(f"Estimated cost:         ~${est_usd:.4f} USD")
    print("Prefix sharing + SQLite caching ensures trees execute once")
    print("and all policy evaluations replay from the cache for free.")
    print("=" * 60)


def run_sweep(num_problems: int = 20, seeds: list[int] | None = None) -> list[dict]:
    seeds = seeds or DEFAULT_SEEDS
    tasks = TASKS[:num_problems]
    all_results: list[dict] = []

    total_configs = len(seeds) * len(tasks) * len(POLICIES) * len(BUDGET_FRACTIONS)
    print(f"\nStarting budget sweep: {len(tasks)} problems * {len(seeds)} seeds * {len(POLICIES)} policies * {len(BUDGET_FRACTIONS)} budgets")
    print(f"Total experiment evaluations: {total_configs}\n")

    for seed_idx, seed in enumerate(seeds):
        print(f"--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")

        for t_idx, task_def in enumerate(tasks):
            t_name = task_def["task_id"]

            # 1. Warm cache and obtain oracle via exhaustive_shared
            oracle_task = CodeTask(
                task_id=task_def["task_id"],
                task_prompt=task_def["prompt"],
                entry_point=task_def["entry_point"],
                tests=task_def["tests"],
                canonical_solution=task_def.get("canonical_solution", ""),
                seed=seed,
            )
            oracle_res = POLICIES["exhaustive_shared"](oracle_task, budget_tokens=10**9)
            oracle_score = oracle_res.best_score
            oracle_leaf = oracle_res.best_leaf_id

            # 2. Compute baseline unshared cost
            ns_task = CodeTask(
                task_id=task_def["task_id"],
                task_prompt=task_def["prompt"],
                entry_point=task_def["entry_point"],
                tests=task_def["tests"],
                canonical_solution=task_def.get("canonical_solution", ""),
                seed=seed,
            )
            ns_res = POLICIES["exhaustive_no_sharing"](ns_task, budget_tokens=10**9)
            baseline_cost = ns_res.tokens_used

            print(f"  [{t_idx + 1:02d}/{len(tasks):02d}] {t_name:<20} | Oracle score: {oracle_score:.2f} | Baseline cost: {baseline_cost} tokens")

            # 3. Evaluate each policy at each budget fraction
            for policy_name, policy_fn in POLICIES.items():
                for b_frac in BUDGET_FRACTIONS:
                    budget = max(1, int(baseline_cost * b_frac))

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
                    regret = max(0.0, oracle_score - res.best_score)

                    row = {
                        "seed": seed,
                        "task_id": t_name,
                        "policy": res.policy_name,
                        "budget_frac": b_frac,
                        "budget_tokens": budget,
                        "tokens_used": res.tokens_used,
                        "best_score": res.best_score,
                        "oracle_score": oracle_score,
                        "regret": regret,
                        "leaves_explored": res.leaves_explored,
                        "oracle_hit": oracle_hit,
                    }
                    all_results.append(row)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run budget-aware exploration sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock LLM without API spend.")
    parser.add_argument("--problems", type=int, default=20, help="Number of coding problems to run (1..20).")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (1..3).")
    parser.add_argument("--output", type=str, default="results/results.csv", help="Output CSV path.")
    args = parser.parse_args()

    if args.dry_run:
        set_dry_run(True)
        print("\n*** RUNNING IN DRY-RUN MODE (Fake deterministic LLM, $0 cost) ***\n")
    else:
        estimate_cost(args.problems)
        confirm = input("Proceed with live LLM calls? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Execution cancelled.")
            sys.exit(0)

    seeds = DEFAULT_SEEDS[:max(1, min(len(DEFAULT_SEEDS), args.seeds))]
    t0 = time.time()
    results = run_sweep(num_problems=args.problems, seeds=seeds)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.output) or "results", exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {args.output} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

