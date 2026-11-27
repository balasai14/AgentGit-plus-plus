# AgentGit++: Budget-Aware Exploration over Checkpoint Trees

Empirical evaluation and reproduction guide for the research claim:

> In a multi-stage LLM workflow with checkpoint prefix sharing, a cost-aware successive-halving scheduler reaches near-oracle best-branch quality at a fraction of the token cost of exhaustive exploration.

---

## 1. Experimental Methodology

### Benchmark Tasks (40+ Harder Problems)
The benchmark suite in `experiments/tasks.py` contains **45 harder coding tasks** drawn from HumanEval+, MBPP-hard, and LiveCodeBench-lite subsets.
- Each problem has rigorous unit test suites (5–8 assertions, including negative values, empty arrays, single elements, and stress corner cases).
- **Pilot Pass-Rate Filter**: Only problems whose 3-sample pilot pass rate falls strictly between **20% and 80%** are retained. This guarantees tasks are neither trivial (100% pass across all leaves) nor intractable (0% pass), creating clear discrimination between search policies.

### Multi-Stage Workflow & Temperature Scaling
Each problem expands a 3-stage tree (**Plan &rarr; Code &rarr; Fix**) with 3 variants per stage, generating $3^3 = 27$ distinct leaves:
- **Variant 0**: `concise` prompt style, **Temperature = 0.2** (deterministic, conservative)
- **Variant 1**: `step_by_step` prompt style, **Temperature = 0.7** (balanced)
- **Variant 2**: `test_driven` prompt style, **Temperature = 1.0** (creative, high variance)

The cache key incorporates `(model, temperature, messages, seed)`, ensuring that distinct seeds produce genuinely different sampling paths.

### Budget Regime (10% to 100% of Shared Cost)
Budgets are defined as **10%, 20%, 30%, 50%, and 100%** of the `exhaustive_shared` tree cost:
- **Exhaustive policies**: Run leaves in fixed order until the budget is depleted, returning the best leaf found so far.
- **CASH (Cost-Aware Successive Halving)**: Dynamically sizes survivor quotas $k_i$ based on measured per-stage costs $c_i$ and projected future expenditure:
  $$\text{remaining\_cost} = \sum_{j > i} c_j \cdot x_j \cdot \text{survivors}_j$$
  CASH **never** exceeds the allocated budget unless one individual path alone does not fit, in which case it logs the warning.
- **Baselines**: `greedy`, `beam`, `sh_cost_blind`, `random_prune` also strictly adhere to the token cap.

---

## 2. Quickstart & Dry-Run (Fast Verification)

Run the full sweep and generate all figures and tables:

```bash
# 1. Run the budget sweep across all 40+ problems and 3 seeds
python experiments/run_budget_sweep.py --dry-run

# 2. Generate publication figures and Markdown/LaTeX tables
python experiments/analyze.py

# 3. Validate theoretical cost models
python experiments/theory_check.py --dry-run
```

---

## 3. Live API Experiment (OpenAI gpt-4o-mini)

```bash
# Configure your API key
echo "OPENAI_API_KEY=your-api-key" > .env

# Run sweep across 40 tasks and 3 seeds
python experiments/run_budget_sweep.py --problems 40 --seeds 3

# Generate analysis artifacts
python experiments/analyze.py

# Check theoretical validation
python experiments/theory_check.py
```

---

## 4. Generated Artifacts

All outputs are saved to `results/`:
- `results/results.csv`: Summary metrics per task &times; seed &times; policy &times; budget.
- `results/leaves.csv`: Leaf-level log recording all 27 paths per problem with per-stage token costs ($c_0, c_1, c_2$), leaf score, and oracle status.
- `results/stage_costs.csv`: Empirical per-stage costs used for theoretical verification.
- `results/fig1_score_vs_budget.png`: Figure 1 (Best Quality vs. Budget Fraction with 95% CI bands).
- `results/fig2_regret.png`: Figure 2 (Regret vs. Oracle).
- `results/table.md`: Markdown summary table.
- `results/table.tex`: Publication-ready LaTeX table.

---

## 5. Web Dashboard

Launch the interactive dashboard to inspect the checkpoint trees and policy simulations:

```bash
python web/serve.py
```
Or open [web/index.html](file:///c:/Users/Sai/Desktop/College/agentgit/web/index.html) in any browser.
