"""Marker-delimited blocks that Headroom manages inside instruction files.

``headroom learn`` and the memory exporters rewrite a section of files the
model reads as instructions on every turn — ``CLAUDE.md``/``CLAUDE.local.md``,
``AGENTS.md``, ``GEMINI.md``, ``GROK.md``, ``.cursor/rules/*.mdc``. The section
is delimited by two HTML-comment markers and rebuilt from content that was
derived from session transcripts: tool output, error text, user messages,
extracted memories. That content is attacker-influenced — a tool result can
carry anything — so before it goes between the markers it must not be able to

* **close or open an HTML comment.** Our markers *are* HTML comments. Content
  containing the end marker would terminate the block early and leave whatever
  follows it outside, where the next run's non-greedy match never looks again:
  a persistent, invisible-to-the-parser instruction in a team-shared file. Any
  other comment would hide text from the human reading the file while the
  model still reads it.
* **hide characters from the human.** Zero-width, bidi-override and other
  control characters render as nothing (or reorder what is shown) while the
  model sees them.
* **forge the block's own structure.** The learn block is parsed back by
  splitting on ``\\n### `` headings; a content line starting with ``### `` would
  become a fake section on the next run.

:func:`sanitize_block_text` neutralises all three. It is visible-by-design:
``<!--`` becomes ``&lt;!--`` rather than being silently deleted, so a reviewer
diffing the file sees that something tried to open a comment.

:func:`block_pattern` matches the block from the first start marker to the
**last** end marker. A well-formed file has exactly one of each, so this is
the same match as a non-greedy pattern; a file poisoned by an earlier version
(a nested end marker followed by escaped text) is re-absorbed whole and, once
its content has been through :func:`sanitize_block_text`, comes out well-formed
again — the fix heals files written before it.
"""

from __future__ import annotations

import re

# Characters that render as nothing or reorder what is rendered: C0/C1
# controls except TAB and LF, DEL, zero-width space/joiners/marks, line and
# paragraph separators, bidi embeddings and overrides, word joiner and friends,
# bidi isolates, and the BOM.
_INVISIBLE_RANGES = (
    (0x00, 0x08),  # C0 controls before TAB
    (0x0B, 0x0C),  # VT, FF
    (0x0E, 0x1F),  # C0 controls after CR
    (0x7F, 0x9F),  # DEL and C1 controls
    (0x200B, 0x200F),  # zero-width space/joiners, LRM/RLM
    (0x2028, 0x2029),  # line/paragraph separator
    (0x202A, 0x202E),  # bidi embeddings and overrides
    (0x2060, 0x2064),  # word joiner and invisible operators
    (0x2066, 0x2069),  # bidi isolates
    (0xFEFF, 0xFEFF),  # BOM / zero-width no-break space
)
_INVISIBLE = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _INVISIBLE_RANGES) + "]")

_HEADING_LINE = re.compile(r"^(#{3}) (?=\S)", re.MULTILINE)


def sanitize_block_text(text: str, *, escape_headings: bool = False) -> str:
    """Make *text* safe to place between Headroom's block markers.

    Removes invisible/control characters (keeping ``\\n`` and ``\\t``),
    neutralises HTML comment delimiters so the text can neither close our
    markers nor hide itself, and — when ``escape_headings`` is set — escapes
    lines that would parse as a ``### `` section heading. Carriage returns are
    normalised to ``\\n`` first so a ``\\r`` cannot split a delimiter.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = text.replace("<!--", "&lt;!--").replace("-->", "--&gt;")
    if escape_headings:
        text = _HEADING_LINE.sub(r"\\\1 ", text)
    return text


def block_pattern(start: str, end: str) -> re.Pattern[str]:
    """Pattern for the managed block: first *start* marker to the **last** *end*.

    See the module docstring for why the outermost pair is the right match.
    """
    return re.compile(re.escape(start) + r".*" + re.escape(end), re.DOTALL)


__all__ = ["block_pattern", "sanitize_block_text"]
