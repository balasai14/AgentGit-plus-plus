# AgentGit++

**Intelligent version control for LLM multi-agent workflows.**

AgentGit++ extends the AgentGit concept — Git-like rollback and branching for agent execution — with an intelligent layer that automatically scores alternative branches using an LLM-as-judge and selects the best path forward. Where AgentGit provides version control for agent execution, **AgentGit++ makes that version control intelligent**.

---

## What This Demonstrates

> **AgentGit** provides version control for agent execution — checkpointing state after each stage, enabling rollback to any prior point, and allowing branches to explore alternative strategies.
>
> **AgentGit++** makes that version control *intelligent* — it uses an LLM judge with a G-Eval-inspired rubric to automatically score branch outputs on relevance, coherence, and completeness, then selects and continues from the best-performing branch without human intervention.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the demo

```bash
python demo.py "The impact of artificial intelligence on modern healthcare"
```

This runs the full end-to-end demo:
- Executes a 4-stage report-writing workflow (Plan → Research → Draft → Review)
- Creates checkpoints at each stage
- Demonstrates rollback and resume
- Creates branches with different writing strategies
- Scores branches using the LLM judge
- Auto-selects the best branch

### 4. Explore the dashboard

```bash
streamlit run dashboard.py
```

The dashboard provides:
- 🌳 Interactive checkpoint tree visualization
- 🔍 State inspector for any checkpoint
- ↩️ Rollback and resume from any point
- 🌿 Branch creation with custom prompts
- ⚖️ LLM judge scoring
- ★ Automatic best-branch selection

---

## File Structure

```
agentgit-plus-plus/
├── agentgit/
│   ├── checkpoint_store.py   # SQLite-backed checkpoint CRUD
│   │                         # (create, get, list children, get lineage)
│   │
│   ├── graph.py              # LangGraph workflow definition
│   │                         # (Plan → Research → Draft → Review nodes)
│   │
│   ├── rollback.py           # rollback(checkpoint_id) → resumes graph from that state
│   │
│   ├── branch.py             # branch(checkpoint_id, new_config) → runs stage variant,
│   │                         # stores sibling checkpoint
│   │
│   ├── judge.py              # LLM-as-judge scoring (G-Eval inspired)
│   │                         # for branch comparison
│   │
│   └── select.py             # select_best_branch(parent_id) → picks the winner
│                             # auto_continue_best() → select + resume
│
├── dashboard.py              # Streamlit app: checkpoint tree, state viewer,
│                             # action triggers, score display
│
├── demo.py                   # CLI script: full end-to-end demo
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
```

---

## Key Concepts

### Checkpoints
Every stage in the workflow automatically saves its full state (inputs, outputs, message history) to SQLite with a unique ID and a pointer to its parent checkpoint. This creates a tree structure that tracks the complete execution history.

### Rollback
Instead of restarting a failed workflow from scratch, you can roll back to any prior checkpoint and resume from there. Only the stages after the rollback point are re-executed.

### Branching
From any checkpoint, you can create branches — re-run the same stage with a different prompt, temperature, or model. Branches are stored as sibling checkpoints under the same parent, enabling A/B comparison of different strategies.

### Scoring (The "++" Layer)
The LLM judge evaluates branch outputs using a structured rubric:
- **Relevance** (0-10): How well does the output address the topic?
- **Coherence** (0-10): How well-structured and logical is it?
- **Completeness** (0-10): How thorough is the coverage?

The judge uses chain-of-thought reasoning before scoring, inspired by the G-Eval methodology.

### Intelligent Selection
`select_best_branch()` compares all branches under a parent, scores any unscored ones, and returns the highest-scoring branch. `auto_continue_best()` takes this further by automatically resuming the workflow from the winning branch.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph (StateGraph) |
| LLM Client | langchain-openai (OpenAI-compatible) |
| Checkpoint Storage | SQLite (via sqlite3) |
| Dashboard | Streamlit + streamlit-flow |
| Config | python-dotenv |

---

## Future Work

These are out of scope for this prototype but represent natural extensions:

- **Delta/compressed checkpoints** — Store only state diffs instead of full snapshots
- **Adaptive pruning** — Automatically discard low-scoring branches to save storage
- **Multi-model BYOK routing** — Route different stages to different LLM providers based on cost/quality tradeoffs
- **Parallel branch execution** — Run multiple branches concurrently instead of sequentially
- **Custom evaluation rubrics** — Allow users to define their own scoring criteria per task type
