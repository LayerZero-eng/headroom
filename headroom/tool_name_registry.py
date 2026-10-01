"""Remember ``tool_use_id -> tool name`` from streamed model responses.

Claude Code's server-stored Threads send a continue turn as only the new
``tool_result`` blocks; the assistant ``tool_use`` that names the tool lives
upstream and is never resent. ContentRouter matches tool exclusions (Read,
Edit, Grep, ...) by name from the request's own ``tool_use`` blocks, so on
continue turns every result was nameless and fresh source code was lossily
compressed. Names seen in responses passing through the proxy fill that gap.
Process-global, thread-safe LRU.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

# ponytail: bounded dict, LRU/TTL if memory matters
_MAX = 20_000
_lock = threading.Lock()
_names: OrderedDict[str, str] = OrderedDict()
# message id -> {"system"/"tools": sha256} as FORWARDED on that Thread turn (a key is
# present only if the forwarded body had it). The stored thread holds these exactly,
# not what the client sent, and the API 400s on any difference. The values are
# canonical JSON bytes interned by hash: system/tools are large and identical across
# the turns of a thread, so each distinct blob is held once and freed when no
# recorded turn references it.
_THREAD_MAX = 2_000  # ponytail: bounded dict, LRU/TTL if memory matters
_thread_pinned: OrderedDict[str, dict[str, str]] = OrderedDict()
_blobs: dict[str, bytes] = {}
_blob_refs: dict[str, int] = {}
ThreadPinned = dict[str, bytes]


def thread_pinned_of(body: dict[str, Any]) -> ThreadPinned | None:
    """Canonical JSON of a Thread request body's system/tools, else None.

    Order-preserving (the bytes must round-trip to the same wire order); bytes are
    immutable, so later mutation of ``body`` cannot change what is recorded.
    """
    if not isinstance(body.get("thread"), dict):
        return None
    return {
        k: json.dumps(body[k], separators=(",", ":"), ensure_ascii=False).encode()
        for k in ("system", "tools")
        if k in body
    }


def _release(refs: dict[str, str]) -> None:
    for sha in refs.values():
        _blob_refs[sha] -= 1
        if _blob_refs[sha] <= 0:
            del _blob_refs[sha], _blobs[sha]


def record_thread_pinned(message_id: str, forwarded: ThreadPinned) -> None:
    refs: dict[str, str] = {}
    with _lock:
        for k, blob in forwarded.items():
            sha = hashlib.sha256(blob).hexdigest()
            _blobs.setdefault(sha, blob)
            _blob_refs[sha] = _blob_refs.get(sha, 0) + 1
            refs[k] = sha
        old = _thread_pinned.pop(message_id, None)
        if old is not None:
            _release(old)
        _thread_pinned[message_id] = refs
        while len(_thread_pinned) > _THREAD_MAX:
            _release(_thread_pinned.popitem(last=False)[1])


def lookup_thread_pinned(message_id: object) -> dict[str, Any] | None:
    """The system/tools forwarded on that turn, parsed fresh (callers may mutate)."""
    if not isinstance(message_id, str):
        return None
    with _lock:
        refs = _thread_pinned.get(message_id)
        blobs = {k: _blobs[sha] for k, sha in refs.items()} if refs is not None else None
    return None if blobs is None else {k: json.loads(v) for k, v in blobs.items()}


def record(tool_use_id: str, name: str) -> None:
    with _lock:
        _names[tool_use_id] = name
        _names.move_to_end(tool_use_id)
        while len(_names) > _MAX:
            _names.popitem(last=False)


def _record_blocks(blocks: object) -> None:
    for b in blocks if isinstance(blocks, list) else ():
        if (
            isinstance(b, dict)
            and b.get("type") in ("tool_use", "server_tool_use")
            and isinstance(b.get("id"), str)
            and isinstance(b.get("name"), str)
        ):
            record(b["id"], b["name"])


def record_from_json(resp: object) -> None:
    """Record tool_use blocks from a non-streaming Messages response body."""
    if isinstance(resp, dict):
        _record_blocks(resp.get("content"))


def record_from_sse(data: bytes, thread_pinned: ThreadPinned | None = None) -> bytes:
    """Record tool_use starts from SSE bytes; return the trailing partial line.

    Parses each complete ``data:`` line as JSON (field-order independent). The
    caller prepends the returned remainder to the next chunk.
    """
    *lines, rest = data.split(b"\n")
    for line in lines:
        if line.startswith(b"data:"):
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "content_block_start":
                _record_blocks([ev.get("content_block")])
            elif (
                thread_pinned is not None
                and isinstance(ev, dict)
                and ev.get("type") == "message_start"
            ):
                msg = ev.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("id"), str):
                    record_thread_pinned(msg["id"], thread_pinned)
    return rest[-65536:]


def lookup(tool_use_id: str) -> str | None:
    with _lock:
        return _names.get(tool_use_id)
