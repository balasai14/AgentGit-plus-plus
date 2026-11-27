# AgentGit++

**Budget-Aware Tree Exploration over Multi-Stage LLM Checkpoints**

AgentGit++ explores branching LLM generation trees (e.g., Plan &rarr; Code &rarr; Fix with temperature and prompt variants) while intelligently managing token budgets via checkpoint prefix sharing and cost-aware successive halving.

---

## Key Features

- **Transparent SQLite LLM Cache (`agentgit/llm_cache.py`)**: Stores responses by `(model, temperature, messages, seed)`. Returns responses with `charged=False` while preserving original token counters for accurate cost accounting on replay.
- **Prefix-Shared Checkpoint Trees (`agentgit/checkpoint_store.py`)**: Stores stage costs and cumulative path costs without duplicating generation of shared prefixes.
- **Strict Budget Enforcement (`agentgit/policies.py`)**:
  - `exhaustive_shared` and `exhaustive_no_sharing` run leaves in fixed order until budget is spent.
  - `cost_aware_sh` (CASH) dynamically sizes survivor quotas based on remaining budget and projected future costs ($\text{remaining\_cost} = \sum c_j \cdot x_j \cdot \text{survivors}$). Strictly bounds expenditure within budget.
  - Baselines: `beam`, `greedy`, `sh_cost_blind`, `random_prune`.
- **40+ Harder Benchmark Suite (`experiments/tasks.py`)**: 45 problems from HumanEval+, MBPP-hard, and LiveCodeBench-lite with pilot pass rates pre-filtered to $[20\%, 80\%]$.
- **Temperature & Prompt Styling (`experiments/code_task.py`)**: 3 variants per stage combining prompt style with distinct temperatures:
  - Variant 0: `concise`, $T = 0.2$
  - Variant 1: `step_by_step`, $T = 0.7$
  - Variant 2: `test_driven`, $T = 1.0$
- **Leaf-Level & Stage Logging**:
  - `results/results.csv`: Task-level policy evaluations across budgets.
  - `results/leaves.csv`: All 27 paths per problem with per-stage token breakdowns.
  - `results/stage_costs.csv`: Empirical $(c_0, c_1, c_2)$ per task for theory validation.
- **Publication-Ready Figures & Tables (`experiments/analyze.py`)**: Figure 1 (Quality vs Budget), Figure 2 (Regret vs Oracle), Markdown & LaTeX tables.
- **Modern Web Dashboard (`web/`)**: shadcn/ui zinc dark-mode dashboard with interactive checkpoint tree visualizer, live policy pruning simulator, and benchmark explorer.

---

## Quick Start (Dry-Run in < 20 seconds)

```bash
# 1. Run budget sweep across 40 tasks and 3 seeds
python experiments/run_budget_sweep.py --dry-run

# 2. Generate analysis figures and LaTeX/Markdown tables
python experiments/analyze.py

# 3. Verify theory equations against logged stage costs
python experiments/theory_check.py --dry-run

# 4. Launch the web dashboard
python web/serve.py
```

See [README_EXPERIMENT.md](README_EXPERIMENT.md) for full experimental setup and live API reproduction.
