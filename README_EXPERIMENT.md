# AgentGit++: Budget-Aware Exploration over Checkpoint Trees

This document outlines the theoretical background, experimental protocol, and instructions to reproduce all figures and tables.

## 1. Research Claim

> In a multi-stage LLM workflow with checkpoint prefix sharing, a cost-aware successive-halving scheduler reaches near-oracle best-branch quality at a fraction of the token cost of exhaustive exploration.

## 2. Theoretical Framework

For a workflow of $S$ stages (e.g. Plan $\to$ Code $\to$ Fix) with branching factor $x$ and average per-stage token costs $c_i$:

1. **Exhaustive Exploration without Prefix Sharing ($C_{\text{std}}$)**:
   $$C_{\text{std}} = \left(\sum_{i=1}^S c_i\right) \cdot \prod_{i=1}^S x_i$$
2. **Exhaustive Exploration with Prefix Sharing ($C_{\text{shared}}$)**:
   $$C_{\text{shared}} = \sum_{i=1}^S \left(c_i \cdot \prod_{j \le i} x_j\right)$$
3. **Pruned / Successive Halving Exploration ($C_{\text{pruned}}$)**:
   $$C_{\text{pruned}} = \sum_{i=1}^S c_i \cdot x_i \cdot N_{i-1}, \quad N_0 = 1, \quad N_i = \min(N_{i-1} \cdot x_i, k_i)$$

Our proposed **Cost-Aware Successive Halving (`cost_aware_sh`)** dynamically solves for the maximum survivor count $k_i$ at each stage such that:
$$\text{Cost}_{\text{used}} + \sum_{j > i} c_j \cdot x \cdot \text{survivors}_j \le B$$

## 3. Quickstart & Dry-Run (< 1 minute)

The repository provides a deterministic mock mode that validates the entire pipeline without spending API tokens:

```bash
# 1. Run the budget sweep across all 20 problems, 3 seeds, and 4 budget fractions
python experiments/run_budget_sweep.py --dry-run

# 2. Analyze results and generate figures + markdown/LaTeX tables
python experiments/analyze.py

# 3. Validate empirical token costs against theoretical formulas
python experiments/theory_check.py --dry-run
```

All figures and tables will be generated under `results/`:
- `results/fig1_score_vs_budget.png`: Figure 1 (Best Score vs Token Budget with 95% CI bands)
- `results/fig2_regret.png`: Figure 2 (Regret vs Oracle across policies)
- `results/table.md`: Summary table in GitHub Markdown
- `results/table.tex`: Summary table in LaTeX format
- `results/results.csv`: Complete raw evaluation logs

## 4. Live API Experiment (gpt-4o-mini)

To run the live experiment against OpenAI:

```bash
# 1. Configure your OpenAI API key in .env
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 2. Run the budget sweep (estimates spend and asks for confirmation first)
python experiments/run_budget_sweep.py --problems 20 --seeds 3

# 3. Generate figures and tables
python experiments/analyze.py

# 4. Verify theory equations
python experiments/theory_check.py
```

## 5. Exploration Policies Evaluated

| Policy | Prefix Sharing | Dynamic Budgeting | Description |
|---|---|---|---|
| `exhaustive_no_sharing` | No | No | Recomputes all 27 paths from scratch (cost upper bound). |
| `exhaustive_shared` | Yes | No | Full tree with prefix caching (AgentGit baseline). |
| `cost_aware_sh` (Ours) | Yes | Yes | Successive halving with survivor count tuned to remaining budget. |
| `sh_cost_blind` | Yes | No | Fixed successive halving ablation ($k_i = 3, 2, 1$). |
| `beam` | Yes | No | Beam search keeping top-2 candidates at each stage. |
| `greedy` | Yes | No | Greedy search keeping single top-1 candidate at each stage. |
| `random_prune` | Yes | Yes | Baseline randomly pruning to budget-constrained survivors. |

