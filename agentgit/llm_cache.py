"""LLM call cache with SQLite backend and token-cost tracking.

Every LLM call goes through ``cached_llm_call()``. On first call the
response is stored; on replay the stored response is returned with
``charged=False`` so cost accounting stays correct while no money is spent.

The cache key incorporates model, temperature, messages, AND seed so that
different seeds produce distinct samples.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Container returned by every (cached or live) LLM call."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    charged: bool  # True = hit API; False = cache replay
    model: str = ""
    cache_key: str = ""
    seed: int | None = None
    temperature: float = 0.7


@dataclass
class CostAccumulator:
    """Running tally of token expenditure for a task / tree."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0

    def add(self, resp: LLMResponse) -> None:
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.total_tokens += resp.total_tokens
        self.calls += 1
        if not resp.charged:
            self.cache_hits += 1


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

_DEFAULT_DB = "cache.db"


def _cache_key(model: str, temperature: float, messages: list[dict], seed: int | None = None) -> str:
    """Generate SHA256 key including model, temperature, messages, and seed."""
    blob = json.dumps(
        {
            "model": model,
            "temperature": round(temperature, 4),
            "messages": messages,
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key           TEXT PRIMARY KEY,
            model               TEXT,
            temperature         REAL,
            seed                INTEGER,
            messages            TEXT,
            response            TEXT,
            prompt_tokens       INTEGER,
            completion_tokens   INTEGER,
            total_tokens        INTEGER,
            latency_s           REAL,
            created_at          REAL
        )
    """)
    conn.commit()


_conn_cache: dict[str, sqlite3.Connection] = {}


def get_connection(db_path: str = _DEFAULT_DB) -> sqlite3.Connection:
    if db_path not in _conn_cache:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _ensure_table(conn)
        _conn_cache[db_path] = conn
    return _conn_cache[db_path]


# ---------------------------------------------------------------------------
# Deterministic mock LLM for dry-run
# ---------------------------------------------------------------------------

_FAKE_COUNTER = 0


def _fake_llm_call(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    seed: int | None = None,
) -> LLMResponse:
    """Deterministic fake LLM for --dry-run testing.

    Varies behavior across prompt style, temperature (0.2, 0.7, 1.0), and seed.
    Higher temperatures produce more stochastic variation and potential bugs,
    allowing different leaves to achieve varied pass rates (20% - 80%).
    """
    global _FAKE_COUNTER
    _FAKE_COUNTER += 1
    key = _cache_key(model, temperature, messages, seed=seed)
    h_int = int(key[:8], 16) + (seed or 0) * 101

    last_msg = messages[-1]["content"] if messages else ""
    lower = last_msg.lower()

    # Extract reference code if present
    ref_match = re.search(r"###REF_CODE:\s*([\s\S]*?)###END_REF###", last_msg)
    ref_code = ref_match.group(1).strip() if ref_match else ""

    # Parse task entry point from prompt if available
    ep_match = re.search(r"`([a-zA-Z_][a-zA-Z0-9_]*)\(", last_msg)
    entry_point = ep_match.group(1) if ep_match else "solution"

    if "judge" in lower or "rate" in lower or "score" in lower:
        # LLM-as-judge for plans: rating 3 to 9
        score = 4 + (h_int % 6)
        content = str(score)
        ptok = max(30, len(last_msg.split()))
        ctok = 3

    elif "plan" in lower or "brief plan" in lower:
        # Plan stage
        if temperature <= 0.3:
            content = (
                f"1. Concise analysis of {entry_point} boundary inputs.\n"
                "2. Direct deterministic algorithmic approach.\n"
                "3. Return final result without side-effects."
            )
        elif temperature <= 0.8:
            content = (
                f"1. Decompose {entry_point} into pre-processing, core loop, and post-conditions.\n"
                "2. Explicit invariant checks and memory management.\n"
                "3. Handle zero-length and edge-value assertions."
            )
        else:  # High temperature (1.0)
            content = (
                f"1. Test-driven decomposition for {entry_point}.\n"
                "2. Adversarial exploration of edge cases, non-standard types, and performance bounds.\n"
                "3. Refactored single-pass implementation."
            )
        ptok = max(40, len(last_msg.split()))
        ctok = max(25, len(content.split()))

    elif "fix" in lower or "correct" in lower or "diagnose" in lower:
        # Fix stage: takes code and test feedback, attempts to fix
        ptok = max(70, len(last_msg.split()))
        # Quality varies with temperature and seed
        # temp 0.2: high fix fidelity if bug was simple
        # temp 0.7: standard fix
        # temp 1.0: can introduce regression or completely fix
        rand_val = (h_int % 100)
        if ref_code:
            if temperature <= 0.3:
                # 70% fix rate
                content = ref_code if rand_val < 70 else _inject_subtle_bug(ref_code, h_int)
            elif temperature <= 0.8:
                # 60% fix rate
                content = ref_code if rand_val < 60 else _inject_subtle_bug(ref_code, h_int)
            else:
                # 50% fix rate, higher variance
                content = ref_code if rand_val < 50 else _inject_subtle_bug(ref_code, h_int)
        else:
            content = f"def {entry_point}(*args, **kwargs):\n    return None\n"
        ctok = max(30, len(content.split()))

    else:
        # Code generation stage
        ptok = max(50, len(last_msg.split()))
        rand_val = (h_int % 100)
        if ref_code:
            # Vary quality:
            # temp 0.2: conservative, 45% perfect, 55% partial
            # temp 0.7: balanced, 55% perfect, 45% partial
            # temp 1.0: high variance, 40% perfect, 60% partial
            if temperature <= 0.3:
                content = ref_code if rand_val < 45 else _inject_subtle_bug(ref_code, h_int)
            elif temperature <= 0.8:
                content = ref_code if rand_val < 55 else _inject_subtle_bug(ref_code, h_int)
            else:
                content = ref_code if rand_val < 40 else _inject_subtle_bug(ref_code, h_int)
        else:
            content = f"def {entry_point}(*args, **kwargs):\n    return None\n"
        ctok = max(30, len(content.split()))

    # Wrap code in markdown blocks if it's code
    if "def " in content and "```" not in content:
        content = f"```python\n{content}\n```"

    total = ptok + ctok
    return LLMResponse(
        content=content,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        total_tokens=total,
        latency_s=0.005,
        charged=False,
        model=model,
        cache_key=key,
        seed=seed,
        temperature=temperature,
    )


def _inject_subtle_bug(code: str, salt: int) -> str:
    """Inject a realistic subtle edge-case bug without affecting loop bounds or causing infinite loops."""
    mod = salt % 3
    if mod == 0 and "return " in code:
        # Off-by-one or edge case in return
        return code.replace("return ", "return -1 if False else ", 1)
    elif mod == 1 and " + 1" in code:
        return code.replace(" + 1", " + 0", 1)
    elif mod == 2 and " - 1" in code:
        return code.replace(" - 1", " - 0", 1)
    return code + "\n# edge case variant\n"


# ---------------------------------------------------------------------------
# Global dry-run flag
# ---------------------------------------------------------------------------

_DRY_RUN = False


def set_dry_run(flag: bool = True) -> None:
    global _DRY_RUN
    _DRY_RUN = flag


def is_dry_run() -> bool:
    return _DRY_RUN


# ---------------------------------------------------------------------------
# Core call function
# ---------------------------------------------------------------------------

def cached_llm_call(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    db_path: str = _DEFAULT_DB,
    seed: int | None = None,
) -> LLMResponse:
    """Call an LLM with transparent SQLite caching.

    On cache hit the stored response is returned with ``charged=False``.
    Token counts are always reported so cost accounting stays correct
    even on replay.
    The cache key includes ``seed`` so different seeds produce distinct cache hits.
    """
    key = _cache_key(model, temperature, messages, seed=seed)

    # --- dry-run shortcut ---
    if _DRY_RUN:
        return _fake_llm_call(model, messages, temperature=temperature, seed=seed)

    # --- check cache ---
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT response, prompt_tokens, completion_tokens, total_tokens, latency_s "
        "FROM llm_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()

    if row is not None:
        return LLMResponse(
            content=row[0],
            prompt_tokens=row[1],
            completion_tokens=row[2],
            total_tokens=row[3],
            latency_s=row[4],
            charged=False,
            model=model,
            cache_key=key,
            seed=seed,
            temperature=temperature,
        )

    # --- live API call ---
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()

    client = OpenAI()
    t0 = time.time()
    extra: dict[str, Any] = {}
    if seed is not None:
        extra["seed"] = seed
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )
    latency = time.time() - t0
    choice = completion.choices[0].message.content or ""
    usage = completion.usage

    prompt_toks = usage.prompt_tokens if usage else 100
    comp_toks = usage.completion_tokens if usage else 50
    tot_toks = usage.total_tokens if usage else (prompt_toks + comp_toks)

    resp = LLMResponse(
        content=choice,
        prompt_tokens=prompt_toks,
        completion_tokens=comp_toks,
        total_tokens=tot_toks,
        latency_s=latency,
        charged=True,
        model=model,
        cache_key=key,
        seed=seed,
        temperature=temperature,
    )

    # store in cache
    conn.execute(
        "INSERT OR REPLACE INTO llm_cache "
        "(cache_key, model, temperature, seed, messages, response, "
        " prompt_tokens, completion_tokens, total_tokens, latency_s, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            key, model, temperature, seed, json.dumps(messages), choice,
            resp.prompt_tokens, resp.completion_tokens, resp.total_tokens,
            latency, time.time(),
        ),
    )
    conn.commit()
    return resp


def cost_of(resp: LLMResponse) -> int:
    """Return total token cost of a single call (always reported, even on replay)."""
    return resp.total_tokens
