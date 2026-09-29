"""The proxy credential (HEADROOM_PROXY_TOKEN) must never reach a provider.

VAPT finding 01-F14: a client that authenticated to the proxy with
``Authorization: Bearer <proxy token>`` — the form most SDKs use — had that
header forwarded upstream verbatim, so the operator's proxy credential arrived at
the model provider as an API key.

Every test here drives a real ``create_app`` with a proxy token configured and a
capturing upstream, and asserts that no upstream request on any route contains
the token in any header, for every way a client can present it.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from headroom.proxy.loopback_guard import require_loopback
from headroom.proxy.proxy_credential import (
    SCRUBBED_STATE_KEY,
    ProxyCredentialScrubMiddleware,
    carries_proxy_credential,
    install_upstream_credential_guard,
    scrub_header_pairs,
)
from headroom.proxy.server import ProxyConfig, create_app

TOKEN = "hrpt_7Qm4vN2xK9sL1pR8wT5yU3zA6bC0dE"
PROVIDER_KEY = "sk-provider-real-key-0123456789"

ANTHROPIC = "https://api.anthropic.com"
OPENAI = "https://api.openai.com"


def _app(**kwargs: Any):
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        proxy_token=TOKEN,
        **kwargs,
    )
    app = create_app(config)
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _capture(router: respx.MockRouter) -> list[dict[str, str]]:
    seen: list[dict[str, str]] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append({k.lower(): v for k, v in request.headers.items()})
        if request.url.host == "api.anthropic.com":
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    router.route(url__startswith=ANTHROPIC).mock(side_effect=_record)
    router.route(url__startswith=OPENAI).mock(side_effect=_record)
    return seen


def _assert_no_token(seen: list[dict[str, str]]) -> None:
    assert seen, "the request never reached the upstream; the test proves nothing"
    for headers in seen:
        for name, value in headers.items():
            assert TOKEN not in value, f"proxy token leaked upstream in header {name!r}"


ANTHROPIC_BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}
OPENAI_BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

# (label, headers the client sends)
AUTH_FORMS = [
    ("authorization-bearer", {"authorization": f"Bearer {TOKEN}"}),
    ("authorization-lowercase-scheme", {"authorization": f"bearer {TOKEN}"}),
    ("explicit-header", {"x-headroom-proxy-token": TOKEN}),
    ("x-api-key", {"x-headroom-proxy-token": TOKEN, "x-api-key": TOKEN}),
    ("api-key", {"x-headroom-proxy-token": TOKEN, "api-key": TOKEN}),
]

ROUTES = [
    ("anthropic-messages", "POST", "/v1/messages", ANTHROPIC_BODY),
    ("openai-chat", "POST", "/v1/chat/completions", OPENAI_BODY),
    ("project-prefixed", "POST", "/p/demo/v1/chat/completions", OPENAI_BODY),
    ("passthrough-get", "GET", "/v1/models", None),
]


@pytest.mark.parametrize("client_host", ["203.0.113.7", "127.0.0.1"], ids=["remote", "loopback"])
@pytest.mark.parametrize("label,auth", AUTH_FORMS, ids=[a[0] for a in AUTH_FORMS])
@pytest.mark.parametrize("route", ROUTES, ids=[r[0] for r in ROUTES])
def test_proxy_token_never_reaches_the_provider(route, label, auth, client_host) -> None:
    _name, method, path, body = route
    app = _app()
    with respx.mock(assert_all_called=False) as router:
        seen = _capture(router)
        with TestClient(app, client=(client_host, 50000)) as client:
            resp = client.request(
                method,
                path,
                headers={**auth, "anthropic-version": "2023-06-01"},
                content=json.dumps(body) if body is not None else None,
            )
    assert resp.status_code != 401, resp.text
    _assert_no_token(seen)


def test_provider_key_is_still_forwarded_alongside_the_explicit_proxy_header() -> None:
    """The documented way to send both: proxy token in x-headroom-proxy-token,
    provider key in Authorization. Only the proxy credential is removed."""
    app = _app()
    with respx.mock(assert_all_called=False) as router:
        seen = _capture(router)
        with TestClient(app, client=("203.0.113.7", 50000)) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={
                    "x-headroom-proxy-token": TOKEN,
                    "authorization": f"Bearer {PROVIDER_KEY}",
                },
                json=OPENAI_BODY,
            )
    assert resp.status_code == 200, resp.text
    _assert_no_token(seen)
    assert seen[-1]["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_bearer_proxy_token_without_provider_key_goes_upstream_without_credentials() -> None:
    """Authenticating with Authorization: Bearer <proxy token> and no provider key
    now sends no Authorization upstream (the provider 401s) rather than leaking."""
    app = _app()
    with respx.mock(assert_all_called=False) as router:
        seen = _capture(router)
        with TestClient(app, client=("203.0.113.7", 50000)) as client:
            client.post(
                "/v1/chat/completions",
                headers={"authorization": f"Bearer {TOKEN}"},
                json=OPENAI_BODY,
            )
    _assert_no_token(seen)
    assert "authorization" not in seen[-1]


def test_bearer_proxy_token_is_not_harvested_by_background_pollers() -> None:
    """On main the subscription usage poller captured the caller's
    ``Authorization`` bearer — here, the proxy token — and replayed it to
    ``api.anthropic.com/api/oauth/usage`` on later, unrelated requests. The
    inbound scrub means the token never reaches anything that could store it."""
    everything: list[tuple[str, dict[str, str]]] = []

    def _record(request: httpx.Request) -> httpx.Response:
        everything.append((str(request.url), {k.lower(): v for k, v in request.headers.items()}))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    with respx.mock(assert_all_called=False) as router:
        router.route(url__startswith=ANTHROPIC).mock(side_effect=_record)
        router.route().mock(return_value=httpx.Response(200, json={}))
        for headers in ({"authorization": f"Bearer {TOKEN}"}, {"x-headroom-proxy-token": TOKEN}):
            with TestClient(_app(), client=("203.0.113.7", 50000)) as client:
                client.post(
                    "/v1/messages",
                    headers={**headers, "anthropic-version": "2023-06-01"},
                    json=ANTHROPIC_BODY,
                )
    assert everything
    for url, headers in everything:
        for name, value in headers.items():
            assert TOKEN not in value, f"proxy token replayed to {url} in {name!r}"


def test_gate_still_rejects_a_wrong_token() -> None:
    """The scrub runs inside the gate; it must not weaken authentication."""
    app = _app()
    with TestClient(app, client=("203.0.113.7", 50000)) as client:
        resp = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer not-the-token"},
            json=OPENAI_BODY,
        )
    assert resp.status_code == 401


# ── unit level ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        (TOKEN, True),
        (f"Bearer {TOKEN}", True),
        (f"  Token   {TOKEN}  ", True),
        (f"Bearer {TOKEN}x", False),
        (TOKEN[:-1], False),
        (f"Bearer {PROVIDER_KEY}", False),
        ("", False),
    ],
)
def test_carries_proxy_credential(value: str, expected: bool) -> None:
    assert carries_proxy_credential(value, TOKEN.encode()) is expected


def test_empty_token_never_matches() -> None:
    assert carries_proxy_credential("", b"") is False
    assert scrub_header_pairs([(b"authorization", b"")], b"") == ([(b"authorization", b"")], [])


@pytest.mark.asyncio
async def test_websocket_handshake_is_scrubbed_and_recorded() -> None:
    seen: dict[str, Any] = {}

    async def inner(scope, receive, send):
        seen["headers"] = scope["headers"]
        seen["state"] = scope.get("state", {})

    mw = ProxyCredentialScrubMiddleware(inner, proxy_token=TOKEN)
    await mw(
        {
            "type": "websocket",
            "path": "/v1/responses",
            "headers": [
                (b"authorization", f"Bearer {TOKEN}".encode()),
                (b"x-api-key", PROVIDER_KEY.encode()),
            ],
        },
        None,
        None,
    )
    assert seen["headers"] == [(b"x-api-key", PROVIDER_KEY.encode())]
    assert seen["state"][SCRUBBED_STATE_KEY] == ["authorization"]


@pytest.mark.asyncio
async def test_upstream_client_guard_drops_token_from_any_header() -> None:
    """Defence in depth: headers built from anywhere but the inbound request."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    client = install_upstream_credential_guard(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), TOKEN
    )
    async with client:
        await client.get(
            "https://upstream.example/v1/x",
            headers={
                "authorization": f"Bearer {TOKEN}",
                "x-goog-api-key": TOKEN,
                "x-keep": "value",
            },
        )
    headers = {k.lower(): v for k, v in sent[0].headers.items()}
    assert "authorization" not in headers
    assert "x-goog-api-key" not in headers
    assert headers["x-keep"] == "value"
