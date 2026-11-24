"""LLM call cache with SQLite backend and token-cost tracking.

Every LLM call goes through ``cached_llm_call()``.  On first call the
response is stored; on replay the stored response is returned with
``charged=False`` so cost accounting stays correct while no money is spent.
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
    charged: bool  # True = actually hit the API; False = cache replay
    model: str = ""
    cache_key: str = ""


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


def _cache_key(model: str, temperature: float, messages: list[dict]) -> str:
    blob = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key           TEXT PRIMARY KEY,
            model               TEXT,
            temperature         REAL,
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
# Fake / dry-run LLM
# ---------------------------------------------------------------------------

_FAKE_COUNTER = 0


def _fake_llm_call(model: str, messages: list[dict], seed: int | None = None, **kw) -> LLMResponse:
    """Deterministic fake for --dry-run testing.

    Produces realistic responses for plan, judge, code, and fix stages.
    """
    global _FAKE_COUNTER
    _FAKE_COUNTER += 1
    seed_str = _cache_key(model, kw.get("temperature", 0.0), messages)
    hash_int = int(seed_str[:8], 16) + (seed or 0)

    last_msg = messages[-1]["content"] if messages else ""
    lower = last_msg.lower()

    # Check for embedded dry-run reference code in message
    ref_match = re.search(r"###REF_CODE:\s*([\s\S]*?)###END_REF###", last_msg)
    ref_code = ref_match.group(1).strip() if ref_match else ""

    if "judge" in lower or "rate" in lower or "score" in lower:
        # LLM-as-judge — deterministic score 4..10
        score = 4 + (hash_int % 7)
        content = str(score)
        ptok = max(30, len(last_msg.split()))
        ctok = 3
    elif "plan" in lower or "brief plan" in lower:
        # Plan stage
        content = (
            "1. Validate input types and boundary conditions.\n"
            "2. Initialize auxiliary data structures.\n"
            "3. Execute primary transformation loop.\n"
            "4. Return finalized result satisfying assert specs."
        )
        ptok = max(40, len(last_msg.split()))
        ctok = 45
    elif "fix" in lower or "fixing" in lower:
        # Fix stage — takes previous code and improves it
        if ref_code:
            # 85% chance to output perfect code, otherwise slightly flawed
            if hash_int % 100 < 85:
                content = ref_code
            else:
                content = ref_code.replace("==", "!=") if "==" in ref_code else ref_code
        else:
            content = "def solution(*args, **kwargs):\n    return None\n"
        ptok = max(80, len(last_msg.split()))
        ctok = max(30, len(content.split()))
    else:
        # Code generation stage
        if ref_code:
            # Variant 0 -> 40% chance perfect, Variant 1 -> 70% chance perfect, Variant 2 -> 90% perfect
            mod = hash_int % 10
            if mod >= 3:
                content = ref_code
            else:
                # Buggy version that passes only some tests
                content = ref_code.replace("range(2,", "range(3,") if "range(2," in ref_code else (
                    ref_code.replace("a + b", "a + b if a > 0 else 0") if "a + b" in ref_code else ref_code
                )
        else:
            content = "def solution(*args, **kwargs):\n    return None\n"
        ptok = max(60, len(last_msg.split()))
        ctok = max(30, len(content.split()))

    total = ptok + ctok
    return LLMResponse(
        content=content,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        total_tokens=total,
        latency_s=0.005,
        charged=False,
        model=model,
        cache_key=seed_str,
    )


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
    """
    key = _cache_key(model, temperature, messages)

    # --- dry-run shortcut ---
    if _DRY_RUN:
        return _fake_llm_call(model, messages, seed=seed, temperature=temperature)

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
    )

    # store in cache
    conn.execute(
        "INSERT OR REPLACE INTO llm_cache "
        "(cache_key, model, temperature, messages, response, "
        " prompt_tokens, completion_tokens, total_tokens, latency_s, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            key, model, temperature, json.dumps(messages), choice,
            resp.prompt_tokens, resp.completion_tokens, resp.total_tokens,
            latency, time.time(),
        ),
    )
    conn.commit()
    return resp


def cost_of(resp: LLMResponse) -> int:
    """Return total token cost of a single call (always reported, even on replay)."""
    return resp.total_tokens
