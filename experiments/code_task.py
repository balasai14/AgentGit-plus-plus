"""3-stage code-generation workflow: Plan -> Code -> Fix.

Each stage has 3 prompt variants (concise / step-by-step / test-driven),
giving 3^3 = 27 total leaf paths per problem.

This module implements the ``ExplorationTask`` protocol for ``agentgit.policies``.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from agentgit.llm_cache import cached_llm_call, LLMResponse, CostAccumulator, is_dry_run
from agentgit.checkpoint_store import Checkpoint, CheckpointTree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEN_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
VARIANTS = 3
STAGES = ["plan", "code", "fix"]
VARIANT_LABELS = ["concise", "step_by_step", "test_driven"]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _plan_prompt(task_prompt: str, variant: str, canonical_solution: str = "") -> str:
    base = f"You are an expert Python engineer. Here is the programming task:\n\n{task_prompt}\n\n"
    if variant == "concise":
        prompt = base + "Write a concise, high-level plan (3-4 bullet points) to solve this."
    elif variant == "step_by_step":
        prompt = base + "Write a detailed, step-by-step algorithmic plan covering edge cases and complexity."
    else:  # test_driven
        prompt = base + "Write a test-driven design plan starting from input/output invariants and boundary tests."

    if is_dry_run() and canonical_solution:
        prompt += f"\n###REF_CODE:\n{canonical_solution}\n###END_REF###"
    return prompt


def _code_prompt(task_prompt: str, plan: str, variant: str, canonical_solution: str = "") -> str:
    base = (
        f"Programming task:\n{task_prompt}\n\n"
        f"Approved architecture plan:\n{plan}\n\n"
        "Implement the complete Python function now.\n"
        "Requirements:\n"
        "- Return only valid Python code.\n"
        "- Enclose your code in ```python ... ``` blocks.\n"
    )
    if variant == "concise":
        prompt = base + "- Keep implementation minimal and direct."
    elif variant == "step_by_step":
        prompt = base + "- Include clear inline comments explaining each step of the plan."
    else:  # test_driven
        prompt = base + "- Defensively handle edge cases, empty inputs, and type bounds."

    if is_dry_run() and canonical_solution:
        prompt += f"\n###REF_CODE:\n{canonical_solution}\n###END_REF###"
    return prompt


def _fix_prompt(task_prompt: str, code: str, test_feedback: str, variant: str, canonical_solution: str = "") -> str:
    base = (
        f"Programming task:\n{task_prompt}\n\n"
        f"Current implementation:\n```python\n{code}\n```\n\n"
        f"Automated unit test feedback:\n{test_feedback}\n\n"
        "Diagnose any failures and output the corrected Python function.\n"
        "Return ONLY the fixed code enclosed in ```python ... ``` blocks.\n"
    )
    if variant == "concise":
        prompt = base + "- Make minimal, targeted bug fixes."
    elif variant == "step_by_step":
        prompt = base + "- Carefully analyze why tests failed before correcting the code."
    else:  # test_driven
        prompt = base + "- Rewrite or refactor the solution to ensure 100% test pass rate."

    if is_dry_run() and canonical_solution:
        prompt += f"\n###REF_CODE:\n{canonical_solution}\n###END_REF###"
    return prompt


# ---------------------------------------------------------------------------
# Code extraction & Unit test execution
# ---------------------------------------------------------------------------

_TEST_RESULT_CACHE: dict[str, tuple[float, str]] = {}


def extract_code(text: str) -> str:
    """Extract python source code from LLM completion text."""
    match = re.search(r"```python\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_unit_tests(code: str, entry_point: str, tests: list[str], timeout: float = 5.0) -> tuple[float, str]:
    """Execute unit tests in a subprocess with timeout and memoization.

    Returns:
        (score, details_string) where score in [0.0, 1.0] is fraction of passed tests.
    """
    clean_code = extract_code(code)
    cache_key = f"{entry_point}::{hash(clean_code)}::{len(tests)}"
    if cache_key in _TEST_RESULT_CACHE:
        return _TEST_RESULT_CACHE[cache_key]

    # In dry-run mode, perform instant fast evaluation without spawning subprocesses
    if is_dry_run():
        if not clean_code or "def " not in clean_code:
            res = (0.0, "Dry-run: stub solution failed all tests.")
        else:
            # Check how close it is to reference
            h = abs(hash(clean_code)) % 100
            if "###REF_CODE" in clean_code or h < 80:
                res = (1.0, f"Dry-run: PASSED {len(tests)}/{len(tests)}")
            elif h < 90:
                passed = max(1, len(tests) - 1)
                res = (passed / len(tests), f"Dry-run: PASSED {passed}/{len(tests)}")
            else:
                passed = max(1, len(tests) // 2)
                res = (passed / len(tests), f"Dry-run: PASSED {passed}/{len(tests)}")
        _TEST_RESULT_CACHE[cache_key] = res
        return res

    test_suite = repr(tests)
    runner_script = f"""
import sys

{clean_code}

tests = {test_suite}
passed = 0
failures = []

for idx, t in enumerate(tests):
    try:
        exec(t, globals())
        passed += 1
    except AssertionError as e:
        failures.append(f"Test {{idx + 1}} failed assertion: {{t}}")
    except Exception as e:
        failures.append(f"Test {{idx + 1}} error: {{type(e).__name__}}: {{e}}")

total = len(tests)
print(f"PASSED: {{passed}}/{{total}}")
if failures:
    print("DETAILS: " + " | ".join(failures[:3]))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        match = re.search(r"PASSED:\s*(\d+)/(\d+)", proc.stdout)
        if match:
            p = int(match.group(1))
            tot = int(match.group(2))
            frac = p / max(1, tot)
            res = (frac, proc.stdout.strip())
        else:
            err = proc.stderr.strip()[:300] if proc.stderr else "Runtime execution error"
            res = (0.0, f"Execution failed: {err}")
    except subprocess.TimeoutExpired:
        res = (0.0, "Execution timed out (5s limit exceeded)")
    except Exception as ex:
        res = (0.0, f"Subprocess runner error: {ex}")

    _TEST_RESULT_CACHE[cache_key] = res
    return res


def judge_plan_quality(plan: str, task_prompt: str, seed: int = 42) -> float:
    """LLM-as-judge: rates a plan on a scale of 0 to 10."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert software engineer reviewing algorithm plans. "
                "Rate the provided plan for correctness, completeness, and clarity on a scale of 0 to 10. "
                "Output ONLY a single integer score between 0 and 10."
            ),
        },
        {
            "role": "user",
            "content": f"Task specification:\n{task_prompt}\n\nProposed Plan:\n{plan}",
        },
    ]

    resp = cached_llm_call(JUDGE_MODEL, messages, temperature=0.0, seed=seed)
    match = re.search(r"\b(10|[0-9])\b", resp.content)
    if match:
        score_10 = int(match.group(1))
    else:
        score_10 = 5
    return score_10 / 10.0


# ---------------------------------------------------------------------------
# CodeTask implementation
# ---------------------------------------------------------------------------

@dataclass
class CodeTask:
    """A coding task conforming to the ExplorationTask protocol."""
    task_id: str
    task_prompt: str
    entry_point: str
    tests: list[str]
    canonical_solution: str = ""
    seed: int = 42

    # Protocol fields
    num_stages: int = 3
    variants_per_stage: int = VARIANTS

    # Internal tree and cost tracking
    tree: CheckpointTree = field(default_factory=CheckpointTree)
    cost_accumulator: CostAccumulator = field(default_factory=CostAccumulator)
    _stage_costs: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    _stage_counts: list[int] = field(default_factory=lambda: [0, 0, 0])

    @property
    def measured_stage_costs(self) -> list[float]:
        """Empirical average token cost per stage."""
        costs = []
        for i in range(self.num_stages):
            cnt = self._stage_counts[i]
            if cnt > 0:
                costs.append(self._stage_costs[i] / cnt)
            else:
                costs.append(100.0)  # default estimate before measurement
        return costs

    def expand_stage(self, parent_ids: list[str | None], stage_idx: int) -> list[Checkpoint]:
        """Generate variant children for parents at this stage."""
        stage_name = STAGES[stage_idx]
        new_nodes: list[Checkpoint] = []

        for pid in parent_ids:
            parent = self.tree.get(pid) if pid else None
            parent_cumulative = parent.cumulative_cost_tokens if parent else 0

            for v_idx, v_label in enumerate(VARIANT_LABELS):
                call_seed = self.seed + stage_idx * 100 + v_idx * 10

                if stage_name == "plan":
                    prompt = _plan_prompt(self.task_prompt, v_label, self.canonical_solution)
                elif stage_name == "code":
                    plan_content = parent.content if parent else ""
                    prompt = _code_prompt(self.task_prompt, plan_content, v_label, self.canonical_solution)
                else:  # fix
                    code_content = parent.content if parent else ""
                    # Test parent code to get error feedback for prompt
                    _, feedback = run_unit_tests(code_content, self.entry_point, self.tests)
                    prompt = _fix_prompt(self.task_prompt, code_content, feedback, v_label, self.canonical_solution)

                messages = [{"role": "user", "content": prompt}]
                resp = cached_llm_call(
                    GEN_MODEL,
                    messages,
                    temperature=TEMPERATURE,
                    seed=call_seed,
                )
                self.cost_accumulator.add(resp)

                node = Checkpoint(
                    parent_id=pid,
                    stage=stage_idx,
                    stage_name=stage_name,
                    variant=v_idx,
                    variant_label=v_label,
                    content=resp.content,
                    stage_cost_tokens=resp.total_tokens,
                    cumulative_cost_tokens=parent_cumulative + resp.total_tokens,
                )
                self.tree.add(node)
                new_nodes.append(node)

                self._stage_costs[stage_idx] += resp.total_tokens
                self._stage_counts[stage_idx] += 1

        return new_nodes

    def score_nodes(self, node_ids: list[str], stage_idx: int) -> dict[str, float]:
        """Score nodes at this stage (LLM judge for plans, unit tests for code/fix)."""
        scores: dict[str, float] = {}
        stage_name = STAGES[stage_idx]

        for nid in node_ids:
            node = self.tree.get(nid)
            if stage_name == "plan":
                score = judge_plan_quality(node.content, self.task_prompt, seed=self.seed)
            else:
                score, _ = run_unit_tests(node.content, self.entry_point, self.tests)

            node.score = score
            scores[nid] = score

        return scores

    def get_node(self, node_id: str) -> Checkpoint:
        return self.tree.get(node_id)

