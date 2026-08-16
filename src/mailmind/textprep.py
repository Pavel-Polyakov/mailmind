"""Turning a raw mail body into the text the LLM actually sees.

The truncation cap is recorded on the run because it is the biggest single
driver of both cost and comparability: two runs are only comparable if their
caps match.
"""

from __future__ import annotations

import html
import re

TRUNCATION_MARKER = "\n[... truncated ...]"

_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAKS = re.compile(r"<(br|/p|/div|/tr|/li|/h[1-6])\s*/?>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")

# An attribution line introducing a quoted reply; everything after it is quoted.
_ATTRIBUTION = re.compile(
    r"""^\s*(
          On\ .{0,200}?\ wrote:
        | .{0,120}\ wrote:$
        | -{2,5}\s*Original\ Message\s*-{2,5}
        | _{5,}
        | From:\s.+
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_SIGNATURE = re.compile(r"^--\s?$")

# Tracking pixels, unsubscribe furniture, and other boilerplate lines.
_BOILERPLATE = re.compile(
    r"""^\s*(
          view\ (this\ email\ )?in\ (your\ )?browser
        | unsubscribe(\ from\ (this|these).*)?
        | (you\ are\ )?receiving\ this\ (email|message)\ because.*
        | sent\ (from|via)\ my\ \w+
        | \[image:.*\]
        | https?://\S+
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_BLANKS = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def html_to_text(raw: str) -> str:
    """A deliberately crude HTML fallback for mail with no text/plain part."""
    text = _SCRIPT_STYLE.sub(" ", raw)
    text = _BREAKS.sub("\n", text)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return text


def strip_quoted(text: str) -> str:
    """Drop quoted reply chains, signatures, and obvious boilerplate."""
    kept: list[str] = []
    for line in text.splitlines():
        if _SIGNATURE.match(line) or _ATTRIBUTION.match(line):
            break
        if line.lstrip().startswith(">"):
            continue
        if _BOILERPLATE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def truncate(text: str, max_chars: int) -> str:
    """Cut to the cap, preferring a line then a word boundary."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Only honour a boundary that is reasonably close to the cap, otherwise a
    # body with one enormous line would lose almost everything.
    floor = int(max_chars * 0.8)
    cut = window.rfind("\n")
    if cut < floor:
        cut = window.rfind(" ")
    if cut < floor:
        cut = max_chars
    return window[:cut].rstrip() + TRUNCATION_MARKER


def prepare_body(
    plain: str | None, html_body: str | None, max_chars: int
) -> str:
    """Full pipeline: pick a part, de-quote, normalise whitespace, truncate."""
    text = plain if plain and plain.strip() else html_to_text(html_body or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​‌‍﻿]", "", text)
    text = strip_quoted(text)
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANKS.sub("\n\n", text).strip()
    return truncate(text, max_chars)
