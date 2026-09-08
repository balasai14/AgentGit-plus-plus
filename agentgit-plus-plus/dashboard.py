"""
dashboard.py — Streamlit dashboard for AgentGit++

A visual interface for exploring the checkpoint tree, inspecting state,
and triggering operations (rollback, branch, score, select).

Usage:
    streamlit run dashboard.py
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from agentgit.checkpoint_store import CheckpointStore
from agentgit.graph import run_workflow, STAGE_ORDER
from agentgit.branch import create_branch
from agentgit.judge import score_checkpoint, score_all_children
from agentgit.select import select_best_branch
from agentgit.rollback import resume_from


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgentGit++",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
        padding-top: 4.5rem;
        max-width: 95%;
        margin: 0 auto;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d2ff;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
        display: block;
    }
    .sub-title {
        color: #8892b0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .tree-node {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-family: 'Segoe UI', sans-serif;
        margin: 3px 0;
        border: 2px solid;
    }
    .tree-node-plan { background: rgba(0,210,255,0.12); border-color: #00d2ff; color: #00d2ff; }
    .tree-node-research { background: rgba(123,47,247,0.12); border-color: #7b2ff7; color: #7b2ff7; }
    .tree-node-draft { background: rgba(255,107,107,0.12); border-color: #ff6b6b; color: #ff6b6b; }
    .tree-node-review { background: rgba(0,210,120,0.12); border-color: #00d278; color: #00d278; }
    .tree-connector { color: #444; margin-left: 6px; }
    .score-high { color: #00d278; font-weight: 600; }
    .score-mid { color: #ffb400; font-weight: 600; }
    .score-low { color: #ff3c3c; font-weight: 600; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
DB_PATH = "demo_checkpoints.db"

if "store" not in st.session_state:
    st.session_state.store = CheckpointStore(DB_PATH)
if "selected_checkpoint" not in st.session_state:
    st.session_state.selected_checkpoint = None
if "last_action_result" not in st.session_state:
    st.session_state.last_action_result = None

store = st.session_state.store


# ---------------------------------------------------------------------------
# Helper: build HTML tree
# ---------------------------------------------------------------------------
def build_html_tree():
    """Build an HTML representation of the checkpoint tree."""
    all_cps = store.get_all_checkpoints()
    if not all_cps:
        return "<p style='color:#8892b0;'>No checkpoints yet. Run a workflow first.</p>"

    # Build adjacency
    children_map = {}
    for cp in all_cps:
        children_map.setdefault(cp.parent_id, []).append(cp)

    roots = children_map.get(None, [])
    lines = []

    def render_node(cp, depth=0):
        indent = "&nbsp;" * (depth * 6)
        connector = "└── " if depth > 0 else ""
        stage_cls = f"tree-node-{cp.stage_name}"

        score_html = ""
        if cp.score is not None:
            cls = "score-high" if cp.score >= 7 else ("score-mid" if cp.score >= 4 else "score-low")
            score_html = f' <span class="{cls}">★ {cp.score:.1f}</span>'

        label = (
            f'{indent}<span class="tree-connector">{connector}</span>'
            f'<span class="tree-node {stage_cls}">'
            f'{cp.stage_name.upper()} · {cp.branch_label} · <code>{cp.id}</code>'
            f'{score_html}</span>'
        )
        lines.append(label)

        for child in children_map.get(cp.id, []):
            render_node(child, depth + 1)

    for root in roots:
        render_node(root, 0)

    return "<br>".join(lines)


def get_score_class(score):
    if score is None:
        return ""
    if score >= 7:
        return "score-high"
    if score >= 4:
        return "score-mid"
    return "score-low"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🌳 AgentGit++ Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Intelligent version control for LLM agent workflows</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: Actions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 Actions")

    # ---- Clear Data ----
    if st.button("🗑️ Clear All Data", key="clear_btn", use_container_width=True):
        store.clear_all()
        st.session_state.selected_checkpoint = None
        st.session_state.last_action_result = "All data cleared"
        st.rerun()

    # ---- Run New Workflow ----
    with st.expander("▶️ Run New Workflow", expanded=False):
        topic = st.text_input(
            "Report Topic",
            value="The impact of artificial intelligence on modern healthcare",
            key="new_topic",
        )
        if st.button("🚀 Run Workflow", key="run_btn", use_container_width=True):
            with st.spinner("Running 4-stage workflow..."):
                store.clear_all()
                result = run_workflow(topic=topic, store=store, branch_label="main")
                st.session_state.last_action_result = "Workflow completed!"
                st.rerun()

    # ---- Rollback & Resume ----
    all_cps = store.get_all_checkpoints()
    cp_options = {f"{cp.stage_name} ({cp.id}) [{cp.branch_label}]": cp.id for cp in all_cps}

    with st.expander("↩️ Rollback & Resume", expanded=False):
        if cp_options:
            selected_cp_label = st.selectbox(
                "Rollback to:", options=list(cp_options.keys()), key="rollback_select",
            )
            rollback_label = st.text_input("Branch label", value="rollback-run", key="rollback_label")
            if st.button("↩️ Rollback & Resume", key="rollback_btn", use_container_width=True):
                cp_id = cp_options[selected_cp_label]
                with st.spinner(f"Rolling back to {cp_id}..."):
                    resume_from(store, cp_id, branch_label=rollback_label)
                    st.session_state.last_action_result = f"Resumed from {cp_id}"
                    st.rerun()
        else:
            st.info("Run a workflow first.")

    # ---- Create Branch ----
    with st.expander("🌿 Create Branch", expanded=False):
        if cp_options:
            branch_from_label = st.selectbox(
                "Branch from:", options=list(cp_options.keys()), key="branch_select",
            )
            custom_prompt = st.text_area(
                "Custom system prompt:",
                value="Write in a more academic, scholarly style with precise language.",
                key="branch_prompt", height=80,
            )
            branch_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="branch_temp")
            branch_name = st.text_input("Branch label", value="custom-branch", key="branch_name")
            if st.button("🌿 Create Branch", key="branch_btn", use_container_width=True):
                cp_id = cp_options[branch_from_label]
                with st.spinner(f"Creating branch..."):
                    new_id = create_branch(
                        store, cp_id,
                        new_config={"prompt_variant": custom_prompt, "temperature": branch_temp},
                        branch_label=branch_name,
                    )
                    st.session_state.last_action_result = f"Branch created: {new_id}"
                    st.rerun()
        else:
            st.info("Run a workflow first.")

    # ---- Score All ----
    with st.expander("⚖️ Score All Checkpoints", expanded=False):
        if cp_options:
            unscored = [cp for cp in all_cps if cp.score is None]
            st.caption(f"{len(unscored)} unscored checkpoint(s)")
            if st.button("⚖️ Score All", key="score_btn", use_container_width=True):
                with st.spinner(f"Scoring {len(unscored)} checkpoints..."):
                    count = 0
                    for cp in unscored:
                        score_checkpoint(store, cp.id)
                        count += 1
                    st.session_state.last_action_result = f"Scored {count} checkpoints"
                    st.rerun()
        else:
            st.info("Run a workflow first.")

    # ---- Select Best ----
    with st.expander("★ Select Best Branch", expanded=False):
        if cp_options:
            best_parent_label = st.selectbox(
                "Select best child of:", options=list(cp_options.keys()), key="best_select",
            )
            if st.button("★ Select Best", key="best_btn", use_container_width=True):
                cp_id = cp_options[best_parent_label]
                with st.spinner("Selecting..."):
                    best = select_best_branch(store, cp_id)
                    if best:
                        st.session_state.selected_checkpoint = best.id
                        st.session_state.last_action_result = (
                            f"Best: {best.branch_label} ({best.id}) — {best.score:.1f}/10"
                        )
                    st.rerun()
        else:
            st.info("Run a workflow first.")

    if st.session_state.last_action_result:
        st.success(st.session_state.last_action_result)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
all_cps = store.get_all_checkpoints()

if not all_cps:
    st.info("No checkpoints yet. Use the sidebar to **Run New Workflow**.")
else:
    # ---- Metrics ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Checkpoints", len(all_cps))
    with col2:
        st.metric("Branches", len(set(cp.branch_label for cp in all_cps)))
    with col3:
        scored = len([cp for cp in all_cps if cp.score is not None])
        st.metric("Scored", f"{scored}/{len(all_cps)}")
    with col4:
        best_scored = [cp for cp in all_cps if cp.score is not None]
        best_score = max((cp.score for cp in best_scored), default=0)
        st.metric("Best Score", f"{best_score:.1f}/10" if best_scored else "N/A")

    st.divider()

    # ---- Tree + Inspector ----
    tree_col, detail_col = st.columns([1, 1])

    with tree_col:
        st.markdown("### 🌳 Checkpoint Tree")
        st.markdown(build_html_tree(), unsafe_allow_html=True)

        # Text tree fallback
        with st.expander("📋 Text View", expanded=False):
            st.code(store.print_tree(), language=None)

    with detail_col:
        st.markdown("### 🔍 State Inspector")

        cp_labels = {
            f"{cp.stage_name} ({cp.id}) [{cp.branch_label}]": cp.id
            for cp in all_cps
        }

        default_idx = 0
        if st.session_state.selected_checkpoint:
            for i, (label, cp_id) in enumerate(cp_labels.items()):
                if cp_id == st.session_state.selected_checkpoint:
                    default_idx = i
                    break

        selected_label = st.selectbox(
            "Select checkpoint:", options=list(cp_labels.keys()),
            index=default_idx, key="inspect_select",
        )

        if selected_label:
            cp_id = cp_labels[selected_label]
            cp = store.get_checkpoint(cp_id)

            if cp:
                # Metadata
                st.markdown(f"""
                <div class="metric-card">
                    <strong>ID:</strong> <code>{cp.id}</code><br>
                    <strong>Stage:</strong> {cp.stage_name.upper()}<br>
                    <strong>Branch:</strong> {cp.branch_label}<br>
                    <strong>Parent:</strong> <code>{cp.parent_id or 'ROOT'}</code><br>
                    <strong>Created:</strong> {cp.created_at[:19]}
                </div>
                """, unsafe_allow_html=True)

                # Score
                if cp.score is not None:
                    cls = get_score_class(cp.score)
                    st.markdown(f'**Score:** <span class="{cls}">{cp.score:.1f}/10</span>', unsafe_allow_html=True)
                    with st.expander("📝 Judge Reasoning"):
                        st.markdown(cp.score_reasoning or "N/A")
                else:
                    st.caption("Not yet scored")
                    if st.button("⚖️ Score this checkpoint", key=f"score_{cp.id}"):
                        with st.spinner("Scoring..."):
                            score_checkpoint(store, cp.id)
                            st.rerun()

                st.divider()

                # Stage output
                st.markdown("#### Stage Output")
                stage_output = cp.state.get(cp.stage_name, "")
                if stage_output:
                    st.markdown(stage_output)
                else:
                    st.warning("Empty output.")

                # Full state
                with st.expander("🗄️ Full State JSON"):
                    display_state = {
                        k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
                        for k, v in cp.state.items() if k != "messages"
                    }
                    st.json(display_state)

                # Config
                if cp.config and cp.config != {}:
                    with st.expander("⚙️ Config"):
                        st.json(cp.config)

                # Lineage
                with st.expander("🔗 Lineage"):
                    lineage = store.get_lineage(cp.id)
                    for i, a in enumerate(lineage):
                        prefix = "  " * i + ("└── " if i > 0 else "")
                        score_str = f" ({a.score:.1f})" if a.score else ""
                        st.text(f"{prefix}{a.stage_name} ({a.id}) [{a.branch_label}]{score_str}")
