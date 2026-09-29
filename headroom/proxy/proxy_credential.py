"""Keep the operator's proxy credential inside the proxy.

``HEADROOM_PROXY_TOKEN`` authenticates a client *to Headroom*. It must never
reach a model provider. Clients send it in one of several places — the explicit
``x-headroom-proxy-token`` header, or ``Authorization: Bearer <token>``, which is
what most SDKs do when they are pointed at a proxy — and before this module the
``Authorization`` form was forwarded upstream verbatim, handing the operator's
credential to the provider (and to any ``x-headroom-base-url`` upstream) as if it
were an API key.

The fix is deliberately *value-based*, not *path-based*:

* A header is removed when its credential value **is** the proxy token (exactly,
  or as the credential part of ``<scheme> <token>``), whatever the header is
  called. That covers ``authorization``, ``x-api-key``, ``api-key``,
  ``x-goog-api-key`` and anything a future provider adapter invents.
* It does not depend on *how* the security gate decided to let the request in.
  Loopback callers and trusted-gateway CIDRs skip the token check, but a loopback
  client that sends the token anyway must not leak it either.

Two layers apply the rule:

1. :class:`ProxyCredentialScrubMiddleware` runs **innermost** — after the
   security gate and after proxy extensions (oauth2 needs to see the credential
   to decide whether a request authenticated), immediately before routing. Every
   handler, backend adapter (LiteLLM, any-llm), WebSocket relay and gateway turn
   therefore receives a request that no longer carries the token. The names of
   the removed headers are recorded on ``scope["state"]`` under
   :data:`SCRUBBED_STATE_KEY` for diagnostics.
2. :func:`install_upstream_credential_guard` adds an httpx request hook to the
   shared upstream clients as defence in depth, so a header assembled from
   anywhere other than the inbound request (configured extras, stored state, a
   future code path) still cannot carry the token out.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PROXY_TOKEN_ENV = "HEADROOM_PROXY_TOKEN"
SCRUBBED_STATE_KEY = "headroom.proxy_credential_scrubbed"

# A header value longer than this cannot be a bare token or ``Bearer <token>``;
# skipping it keeps the per-request cost flat for large cookies and the like.
_MAX_CREDENTIAL_VALUE_BYTES = 4096


def resolve_proxy_token(config: Any = None) -> str | None:
    """Return the configured proxy token, or ``None`` when auth is off.

    One resolution rule for every consumer (security gate, WebSocket gate, the
    scrub middleware and the upstream client hook) so they cannot disagree about
    which token is in force.
    """
    configured = getattr(config, "proxy_token", None) if config is not None else None
    return configured or os.environ.get(PROXY_TOKEN_ENV) or None


def carries_proxy_credential(value: bytes | str, token: bytes) -> bool:
    """True when ``value`` is the proxy token, bare or as ``<scheme> <token>``.

    Constant-time on the comparison. Never matches when ``token`` is empty.
    """
    if not token:
        return False
    raw = value.encode("latin-1", "replace") if isinstance(value, str) else value
    if len(raw) > _MAX_CREDENTIAL_VALUE_BYTES:
        return False
    candidate = raw.strip()
    if hmac.compare_digest(candidate, token):
        return True
    _scheme, sep, rest = candidate.partition(b" ")
    return bool(sep) and hmac.compare_digest(rest.strip(), token)


def scrub_header_pairs(
    headers: list[tuple[bytes, bytes]], token: bytes
) -> tuple[list[tuple[bytes, bytes]], list[str]]:
    """Return ``headers`` without entries carrying the proxy token, plus their names."""
    if not token:
        return headers, []
    kept: list[tuple[bytes, bytes]] = []
    removed: list[str] = []
    for name, value in headers:
        if carries_proxy_credential(value, token):
            removed.append(name.decode("latin-1").lower())
        else:
            kept.append((name, value))
    return kept, removed


class ProxyCredentialScrubMiddleware:
    """Remove the proxy credential from inbound HTTP and WebSocket requests.

    Register it **first** in ``create_app`` (Starlette prepends middleware, so the
    first registration is innermost): the security gate and extensions must still
    see the credential; nothing after routing may.
    """

    def __init__(self, app: Any, *, proxy_token: str | None = None) -> None:
        self.app = app
        self.token = proxy_token.encode("utf-8") if proxy_token else b""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if self.token and scope.get("type") in {"http", "websocket"}:
            kept, removed = scrub_header_pairs(list(scope.get("headers") or []), self.token)
            if removed:
                scope = dict(scope)
                scope["headers"] = kept
                state = dict(scope.get("state") or {})
                state[SCRUBBED_STATE_KEY] = removed
                scope["state"] = state
                logger.debug(
                    "event=proxy_credential_scrubbed path=%s headers=%s",
                    scope.get("path"),
                    ",".join(removed),
                )
        await self.app(scope, receive, send)


def install_upstream_credential_guard(client: Any, proxy_token: str | None) -> Any:
    """Add an httpx request hook that drops any header carrying the proxy token.

    Returns ``client``. A no-op when no token is configured. Logged at WARNING
    when it fires, because reaching this layer means some code path built
    upstream headers without going through the inbound scrub.
    """
    if not proxy_token:
        return client
    token = proxy_token.encode("utf-8")

    async def _drop_proxy_credential(request: Any) -> None:
        leaking = [
            name for name, value in request.headers.raw if carries_proxy_credential(value, token)
        ]
        for name in leaking:
            key = name.decode("latin-1")
            if key in request.headers:
                del request.headers[key]
        if leaking:
            logger.warning(
                "event=proxy_credential_blocked_upstream host=%s headers=%s",
                request.url.host,
                ",".join(n.decode("latin-1").lower() for n in leaking),
            )

    client.event_hooks.setdefault("request", []).append(_drop_proxy_credential)
    return client
