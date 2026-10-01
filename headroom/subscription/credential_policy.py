"""Which credential a background usage poller may use (VAPT 01-F16).

Usage pollers (the Claude subscription tracker, the Codex ``/wham/usage``
refresher) call a provider on the proxy's behalf and show the result on the
operator's dashboard. The account they poll must be the **operator's**:

1. An operator-configured credential always wins — ``CLAUDE_CODE_OAUTH_TOKEN``
   or the proxy user's own Claude Code credentials file.
2. A credential *learned from traffic* is adopted only from the **local
   operator**: a loopback peer whose request was not forwarded by another hop.
   On a shared proxy (several principals behind one Headroom), a network
   caller's ``Authorization`` bearer must never become the polled account —
   that would spend the caller's credential on a request they never made and
   publish their usage on someone else's dashboard.
3. ``HEADROOM_SUBSCRIPTION_TRAFFIC_TOKEN=off`` disables learning from traffic
   entirely (operator-configured credentials only).

The default (``local``) keeps the single-developer experience: Claude Code on
the same machine, where the OAuth token often lives in the OS keychain rather
than a file, still gets its subscription window without extra setup.
"""

from __future__ import annotations

import os
from typing import Any

from headroom.proxy.loopback_guard import is_loopback_host

TRAFFIC_TOKEN_ENV = "HEADROOM_SUBSCRIPTION_TRAFFIC_TOKEN"
_OFF_VALUES = frozenset({"off", "0", "false", "no", "none", "disabled"})

# Any of these means the request crossed another hop (a gateway or reverse
# proxy on the same host), so the loopback peer is not the end caller.
_FORWARDING_HEADERS = ("forwarded", "x-forwarded-for", "x-real-ip")


def traffic_token_adoption_enabled() -> bool:
    """Whether pollers may learn a credential from local-operator traffic."""
    value = os.environ.get(TRAFFIC_TOKEN_ENV, "local").strip().lower()
    return value not in _OFF_VALUES


def is_local_operator_connection(conn: Any) -> bool:
    """True only for a direct loopback caller (HTTP request or WebSocket).

    Fails closed: a missing peer address, a non-loopback peer, or any
    forwarding header returns ``False``. Trusted-gateway CIDRs deliberately do
    not count — a gateway fronts other principals by definition.
    """
    client = getattr(conn, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    if not isinstance(host, str) or not is_loopback_host(host):
        return False
    headers = getattr(conn, "headers", None)
    if headers is not None:
        for name in _FORWARDING_HEADERS:
            try:
                if headers.get(name):
                    return False
            except Exception:
                return False
    return True


def may_adopt_caller_credential(conn: Any) -> bool:
    """Whether this caller's own credential may drive a background poll."""
    return traffic_token_adoption_enabled() and is_local_operator_connection(conn)
