"""
judge.py — LLM-as-judge scoring for AgentGit++ (the "++" layer)

This is the core intelligence that makes AgentGit++ more than basic
version control. It uses a G-Eval-inspired approach to score checkpoint
outputs on three criteria:

    1. Relevance  (0-10): How relevant is the output to the topic/task?
    2. Coherence  (0-10): How well-structured and logically consistent?
    3. Completeness (0-10): How thorough and comprehensive?

The judge uses chain-of-thought reasoning before scoring, which
improves alignment with human judgment (per the G-Eval paper).

Usage:
    from agentgit.judge import score_checkpoint, score_all_children

    score, reasoning = score_checkpoint(store, "abc123")
    results = score_all_children(store, "parent123")
"""

import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agentgit.checkpoint_store import CheckpointStore

load_dotenv()


# ---------------------------------------------------------------------------
# Judge prompt — G-Eval-inspired rubric with chain-of-thought
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI-generated content.
You will be given the output of a specific stage in a report-writing workflow.
Your job is to evaluate it rigorously using the rubric below.

## Evaluation Rubric

Score each criterion on a scale of 1-10:

### 1. Relevance (1-10)
- How directly does this output address the given topic?
- Does it stay on-topic without tangential or irrelevant content?
- 1 = completely off-topic, 10 = perfectly targeted

### 2. Coherence (1-10)
- How well-structured and logically organized is the output?
- Does it flow naturally with clear transitions between ideas?
- Is it free of contradictions or logical inconsistencies?
- 1 = incoherent/disorganized, 10 = perfectly structured

### 3. Completeness (1-10)
- How thorough is the coverage of the topic/task?
- Does it address all key aspects without major omissions?
- Is the depth of analysis appropriate for the stage?
- 1 = severely incomplete, 10 = comprehensively thorough

## Instructions
1. First, provide chain-of-thought reasoning for EACH criterion.
2. Then assign a score for each criterion.
3. Finally, compute the average score.

## Required Output Format (follow this EXACTLY):

REASONING:
[Your detailed chain-of-thought analysis here]

RELEVANCE: [score]/10
COHERENCE: [score]/10
COMPLETENESS: [score]/10
AVERAGE: [average]/10"""


JUDGE_USER_TEMPLATE = """Evaluate the following {stage_name} output for a report on "{topic}".

## Stage Context
This is the "{stage_name}" stage of a 4-stage report-writing workflow
(Plan → Research → Draft → Review).

## Output to Evaluate:
{output}"""


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------
def _extract_scores(response_text: str) -> tuple[float, str]:
    """Parse the judge's response to extract scores and reasoning.

    Looks for the structured format:
        RELEVANCE: X/10
        COHERENCE: X/10
        COMPLETENESS: X/10
        AVERAGE: X/10

    Falls back to regex matching if the format isn't exact.

    Args:
        response_text: The full text response from the judge LLM.

    Returns:
        Tuple of (average_score, full_reasoning_text).
    """
    # Try to extract individual scores
    relevance = _find_score(response_text, "RELEVANCE")
    coherence = _find_score(response_text, "COHERENCE")
    completeness = _find_score(response_text, "COMPLETENESS")
    average = _find_score(response_text, "AVERAGE")

    # If we found individual scores, compute average ourselves for accuracy
    if all(s is not None for s in [relevance, coherence, completeness]):
        computed_avg = round((relevance + coherence + completeness) / 3, 1)
        final_score = computed_avg
    elif average is not None:
        final_score = average
    else:
        # Last resort: find any number that looks like a score
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*10', response_text)
        if numbers:
            final_score = float(numbers[-1])  # Take the last one (usually average)
        else:
            final_score = 5.0  # Default if parsing fails

    return final_score, response_text


def _find_score(text: str, label: str) -> Optional[float]:
    """Extract a numeric score for a specific criterion label.

    Args:
        text: The response text to search in.
        label: The criterion label (e.g., "RELEVANCE").

    Returns:
        The extracted score as a float, or None if not found.
    """
    # Pattern: "LABEL: X/10" or "LABEL: X.X/10"
    pattern = rf'{label}\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------
def score_checkpoint(
    store: CheckpointStore,
    checkpoint_id: str,
    force_rescore: bool = False,
) -> tuple[float, str]:
    """Score a checkpoint's output using the LLM judge.

    This is the core function of the "++" layer. It calls a (typically
    stronger) LLM to evaluate the quality of a stage's output using
    the G-Eval-inspired rubric.

    Args:
        store: The CheckpointStore instance.
        checkpoint_id: ID of the checkpoint to score.
        force_rescore: If True, re-score even if a score already exists.

    Returns:
        Tuple of (score, reasoning) where score is 0-10 float.

    Raises:
        ValueError: If the checkpoint is not found.
    """
    cp = store.get_checkpoint(checkpoint_id)
    if cp is None:
        raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

    # Skip if already scored (unless force_rescore)
    if cp.score is not None and not force_rescore:
        return cp.score, cp.score_reasoning or ""

    # Get the stage output to evaluate
    stage_output = cp.state.get(cp.stage_name, "")
    topic = cp.state.get("topic", "unknown topic")

    if not stage_output or stage_output.strip() == "":
        # Empty output gets a score of 0
        reasoning = "Output is empty — automatic score of 0."
        store.update_score(checkpoint_id, 0.0, reasoning)
        return 0.0, reasoning

    # Build the judge prompt
    user_prompt = JUDGE_USER_TEMPLATE.format(
        stage_name=cp.stage_name,
        topic=topic,
        output=stage_output,
    )

    # Call the judge LLM
    judge_kwargs = dict(
        model=os.getenv("JUDGE_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.1,
        max_tokens=512,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        judge_kwargs["base_url"] = base_url
    judge_llm = ChatOpenAI(**judge_kwargs)

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = judge_llm.invoke(messages)
    response_text = response.content

    # Extract score and reasoning
    score, reasoning = _extract_scores(response_text)

    # Persist the score
    store.update_score(checkpoint_id, score, reasoning)

    print(f"  ⚖ Scored checkpoint {checkpoint_id}: {score:.1f}/10")

    return score, reasoning


def score_all_children(
    store: CheckpointStore,
    parent_id: str,
    force_rescore: bool = False,
) -> list[dict]:
    """Score all child checkpoints (branches) of a given parent.

    Useful for comparing multiple branches before selecting the best one.

    Args:
        store: The CheckpointStore instance.
        parent_id: ID of the parent checkpoint.
        force_rescore: If True, re-score even if scores already exist.

    Returns:
        List of dicts with keys: checkpoint_id, stage_name, branch_label,
        score, reasoning.
    """
    children = store.get_children(parent_id)

    if not children:
        print(f"  No children found for checkpoint {parent_id}")
        return []

    print(f"\n═══ Scoring {len(children)} Branch(es) ═══")

    results = []
    for child in children:
        score, reasoning = score_checkpoint(
            store, child.id, force_rescore=force_rescore
        )
        results.append({
            "checkpoint_id": child.id,
            "stage_name": child.stage_name,
            "branch_label": child.branch_label,
            "score": score,
            "reasoning": reasoning,
        })

    return results
