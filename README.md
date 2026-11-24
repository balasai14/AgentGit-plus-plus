# AgentGit++

Budget-Aware Exploration over Checkpoint Trees for Multi-Stage LLM Workflows.

AgentGit++ extends multi-stage LLM generation workflows (e.g., Plan $\to$ Code $\to$ Fix) with:
- **Transparent SQLite LLM Caching** with precise token cost accounting (`agentgit/llm_cache.py`).
- **Checkpoint Trees with Prefix Sharing** (`agentgit/checkpoint_store.py`).
- **Cost-Aware Successive Halving** (`cost_aware_sh` in `agentgit/policies.py`).
- **Reproducible Budget Sweep Experiments** across 20 coding tasks (`experiments/`).

## Quick Start (Dry-Run in < 1 minute)

```bash
# Run the complete sweep and generate analysis figures + tables
python experiments/run_budget_sweep.py --dry-run && python experiments/analyze.py
```

See [README_EXPERIMENT.md](README_EXPERIMENT.md) for detailed experimental protocol, theoretical proofs, and reproduction guides.

