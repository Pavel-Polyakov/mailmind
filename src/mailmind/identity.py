"""Derived run identity.

Both values are computed from what actually shapes a run's output. Nothing here
is hand-maintained: a version you must remember to bump is hidden state, and
reproducibility claims stop holding the first time someone forgets.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _digest(payload: str, length: int) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def prompt_version(prompt_text: str, schema: dict[str, Any]) -> str:
    """Identity of a prompt: its text plus the schema it must produce."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return _digest(prompt_text + "\x00" + canonical, 12)


def run_fingerprint(
    *,
    model: str,
    base_url: str | None,
    prompt_version: str,
    schema_version: str,
    query: str | None,
    limit: int | None,
    max_body_chars: int,
    reuse_labels: bool,
) -> str:
    """Identity of a scan: same fingerprint means same intended work.

    A rerun with a matching fingerprint is a continuation of the interrupted
    run; anything else is a new run that must not overwrite the old results.
    """
    payload = json.dumps(
        {
            "model": model,
            "base_url": base_url,
            "prompt_version": prompt_version,
            "schema_version": schema_version,
            "query": query,
            "limit": limit,
            "max_body_chars": max_body_chars,
            "reuse_labels": reuse_labels,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _digest(payload, 16)
