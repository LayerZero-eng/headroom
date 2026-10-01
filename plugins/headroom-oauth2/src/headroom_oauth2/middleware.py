"""ASGI middleware that injects a refreshed OAuth2 bearer on each upstream request.

Headroom's litellm backend forwards the request's `Authorization` bearer to the
upstream as the API key, so setting it here makes the minted token reach the
backend with no core changes.

Two things this layer deliberately does *not* do:

* It does not touch requests the proxy answers itself. Health probes,
  ``/stats``, ``/metrics``, the compress/retrieve/telemetry routes and any
  extension's own ``/ext/...`` routes never go upstream, so there is no bearer
  to inject and no reason to mint one for them. Before this rule existed,
  ``/health`` returned 502 whenever the IdP was unreachable, and every
  management request cost a token mint when the cache was cold.
* It does not mint for a request that has not authenticated to the proxy.
  Current cores register extension middleware *inside* the
  ``HEADROOM_PROXY_TOKEN`` gate, so such a request never reaches this layer;
  on an older core, where extension middleware ran outermost, this check keeps
  an unauthenticated caller from spending a mint or having its ``Authorization``
  rewritten before the gate compares it (which turned the proxy token into a
  401 for every remote client). When the proxy token *is* the bearer the client
  sent, it is verified here and then replaced with the upstream token, which is
  the intended flow, and the verified proxy credential is re-attached as
  ``x-headroom-proxy-token`` so a gate running inside this layer still sees it
  (the core strips ``x-headroom-*`` before the upstream hop). Clients may also
  send that header themselves when they need ``Authorization`` for something
  else.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Iterable

from .provider import OAuth2Error

log = logging.getLogger("headroom_oauth2")

# Exact paths the proxy answers itself and that must stay reachable when the
# IdP is down (orchestrator probes).
LOCAL_ROUTE_PATHS: frozenset[str] = frozenset(
    {"/health", "/healthz", "/livez", "/readyz", "/favicon.ico", "/subscription-window"}
)

# Prefixes of routes the proxy (or a co-installed extension) serves locally,
# never forwarding to a provider. Everything NOT matched here is treated as an
# upstream request, including the provider passthrough catch-all, so a new
# provider route needs no change here; a new *local* route does.
LOCAL_ROUTE_PREFIXES: tuple[str, ...] = (
    "/stats",  # /stats, /stats-history, /stats-lifetime
    "/metrics",
    "/quota",
    "/settings",
    "/dashboard",
    "/admin/",
    "/debug/",
    "/transformations/",
    "/v1/compress",  # incl. /v1/compress/response
    "/v1/usage",
    "/v1/retrieve",
    "/v1/telemetry",
    "/v1/toin",
    "/v1/feedback",
    "/ext/",  # convention for routes registered by other extensions
)

_PROJECT_PREFIX = "/p/"


def _strip_project_prefix(path: str) -> str:
    """Return ``path`` without a ``/p/<project>`` base-URL prefix.

    Uses the core's own splitter when available so the two cannot drift; the
    fallback handles the same ``/p/<name>/rest`` shape.
    """
    try:
        from headroom.proxy.project_policy import split_project_path

        return split_project_path(path)[1]
    except Exception:  # noqa: BLE001 — older core or a different layout
        if not path.startswith(_PROJECT_PREFIX):
            return path
        remainder = path[len(_PROJECT_PREFIX) :]
        segment, sep, rest = remainder.partition("/")
        if not segment:
            return path
        return ("/" + rest) if sep else "/"


def is_local_route(path: str, extra_prefixes: Iterable[str] = ()) -> bool:
    """True when ``path`` is served by the proxy itself and never goes upstream."""
    path = _strip_project_prefix(path or "/")
    if path in LOCAL_ROUTE_PATHS:
        return True
    if path.startswith(LOCAL_ROUTE_PREFIXES):
        return True
    return any(p and path.startswith(p) for p in extra_prefixes)


def _is_loopback(host: str | None) -> bool:
    try:
        from headroom.proxy.loopback_guard import is_loopback_host

        return bool(is_loopback_host(host))
    except Exception:  # noqa: BLE001 — older core; mirror its rule
        if host is None:
            return True
        return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


def _read_proxy_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    """Same rule as the core gate: explicit header first, else the Bearer."""
    try:
        from headroom.proxy.server import read_proxy_token
        from starlette.datastructures import Headers

        return read_proxy_token(Headers(raw=headers))
    except Exception:  # noqa: BLE001 — older core; mirror its rule
        explicit = None
        bearer = None
        for k, v in headers:
            key = k.lower()
            if key == b"x-headroom-proxy-token" and explicit is None:
                explicit = v.decode("latin-1")
            elif key == b"authorization" and bearer is None:
                raw = v.decode("latin-1")
                if raw.lower().startswith("bearer "):
                    bearer = raw[7:].strip()
        if explicit is not None:
            return explicit or None
        return bearer or None


class OAuth2Middleware:
    """ASGI middleware that replaces the request Authorization with a minted bearer."""

    def __init__(
        self,
        app,
        provider,
        *,
        proxy_token: str | None = None,
        local_path_prefixes: Iterable[str] = (),
    ):
        self.app = app
        self.provider = provider
        self.proxy_token = proxy_token or None
        self._proxy_token_bytes = self.proxy_token.encode("utf-8") if self.proxy_token else b""
        self.local_path_prefixes = tuple(p for p in local_path_prefixes if p)

    def _authenticated_to_proxy(self, scope) -> bool:
        """True unless a proxy token is configured and this caller has not presented it."""
        if not self.proxy_token:
            return True
        client = scope.get("client")
        if _is_loopback(client[0] if client else None):
            return True
        provided = _read_proxy_token(list(scope.get("headers") or []))
        return provided is not None and hmac.compare_digest(
            provided.encode("utf-8", "replace"), self._proxy_token_bytes
        )

    def _carry_proxy_credential(self, headers: list[tuple[bytes, bytes]]) -> None:
        """Keep the proxy credential visible to the core gate after Authorization is replaced.

        When the client authenticated with ``Authorization: Bearer <proxy token>`` the
        replacement below removes the only copy of it. On a core whose gate runs *inside*
        this layer that would turn a valid request into a 401, so the verified credential
        is re-attached under the explicit header the gate prefers. The core strips every
        ``x-headroom-*`` header before the upstream hop, so it never leaves the proxy;
        on a core whose gate has already run it is simply unused.
        """
        if not self.proxy_token:
            return
        if any(k.lower() == b"x-headroom-proxy-token" for k, _ in headers):
            return
        headers.append((b"x-headroom-proxy-token", self._proxy_token_bytes))

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if is_local_route(scope.get("path") or "/", self.local_path_prefixes):
            await self.app(scope, receive, send)
            return
        if not self._authenticated_to_proxy(scope):
            # Leave the request exactly as it arrived; the proxy's own gate
            # answers it. Minting here would spend an IdP round-trip on a
            # caller that is about to be refused.
            await self.app(scope, receive, send)
            return
        # Hot path: a cached, still-valid token needs no thread hop. Only mint (blocking
        # urllib) off the event loop when the cache is empty/expired.
        token = self.provider.cached()
        if token is None:
            try:
                loop = asyncio.get_running_loop()
                token = await loop.run_in_executor(None, self.provider.token)
            except OAuth2Error as e:
                log.warning("oauth2: token mint failed: %s", e)
                await self._error(
                    send, 502, "upstream_auth_error", "could not obtain upstream credentials"
                )
                return
        headers = [(k, v) for (k, v) in scope.get("headers", []) if k.lower() != b"authorization"]
        self._carry_proxy_credential(headers)
        headers.append((b"authorization", b"Bearer " + token.encode()))
        await self.app(dict(scope, headers=headers), receive, send)

    @staticmethod
    async def _error(send, status, etype, message):
        body = json.dumps({"type": "error", "error": {"type": etype, "message": message}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
