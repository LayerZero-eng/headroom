"""The oauth2 layer only acts on upstream requests that have authenticated to the proxy.

Regression tests for two defects (VAPT tracker 03-F4): the middleware rewrote
``Authorization`` on *every* HTTP request, so (a) remote clients presenting the
proxy token as their bearer were refused once oauth2 was enabled, and (b) local
routes such as ``/health`` minted a token and returned 502 whenever the IdP was
unreachable.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from headroom_oauth2 import OAuth2Middleware, install
from headroom_oauth2.middleware import is_local_route

try:  # only the end-to-end tests need FastAPI; they importorskip the core too
    from fastapi import Request
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment,misc]

NONLOOPBACK = ("203.0.113.5", 44444)
LOOPBACK = ("127.0.0.1", 12345)


class _Recording:
    def __init__(self):
        self.called = False
        self.scope = None

    async def __call__(self, scope, receive, send):
        self.called = True
        self.scope = scope


class _NeverMint:
    """Provider double that fails the test if the middleware tries to use it."""

    def cached(self):
        raise AssertionError("provider consulted for a request that must not be injected")

    def token(self):
        raise AssertionError("provider minted for a request that must not be injected")


class _Cached:
    def __init__(self, tok="MINTED"):
        self.tok = tok
        self.reads = 0

    def cached(self):
        self.reads += 1
        return self.tok

    def token(self):
        return self.tok


async def _recv():
    return {"type": "http.request"}


async def _ignore(_msg):
    pass


def _scope(path="/v1/chat/completions", headers=(), client=NONLOOPBACK):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": list(headers),
        "client": client,
    }


def _run(mw, scope):
    asyncio.run(mw(scope, _recv, _ignore))


def _auth_of(app) -> bytes | None:
    return dict(app.scope["headers"]).get(b"authorization")


# --- local routes are never touched -------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/healthz",
        "/livez",
        "/readyz",
        "/stats",
        "/stats-history",
        "/metrics",
        "/quota",
        "/settings",
        "/dashboard/",
        "/admin/upstream",
        "/debug/tasks",
        "/v1/compress",
        "/v1/compress/response",
        "/v1/usage",
        "/v1/retrieve/abc",
        "/v1/telemetry/export",
        "/v1/toin/stats",
        "/v1/feedback/tool",
        "/ext/lossless-guard/record",
        "/p/myproject/stats",  # project-prefixed local route
    ],
)
def test_local_routes_are_passed_through_without_minting(path):
    app = _Recording()
    mw = OAuth2Middleware(app, _NeverMint())
    _run(mw, _scope(path, headers=[(b"authorization", b"Bearer CLIENT")]))
    assert app.called
    assert _auth_of(app) == b"Bearer CLIENT"  # untouched


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/models",
        "/p/myproject/v1/messages",  # project-prefixed upstream route
        "/some/vendor/passthrough",  # provider catch-all
        "/",
    ],
)
def test_upstream_routes_are_injected(path):
    app = _Recording()
    mw = OAuth2Middleware(app, _Cached("TOK"))
    _run(mw, _scope(path, headers=[(b"authorization", b"Bearer CLIENT")]))
    assert _auth_of(app) == b"Bearer TOK"


def test_is_local_route_honours_extra_prefixes():
    assert not is_local_route("/vendor-local/ping")
    assert is_local_route("/vendor-local/ping", ["/vendor-local/"])
    assert is_local_route("/p/proj/vendor-local/ping", ["/vendor-local/"])


# --- proxy-token awareness -------------------------------------------------------


def test_no_proxy_token_configured_injects_for_everyone():
    app = _Recording()
    mw = OAuth2Middleware(app, _Cached("TOK"), proxy_token=None)
    _run(mw, _scope(headers=[]))
    assert _auth_of(app) == b"Bearer TOK"


def test_unauthenticated_remote_caller_is_left_untouched_and_costs_no_mint():
    app = _Recording()
    mw = OAuth2Middleware(app, _NeverMint(), proxy_token="s3cr3t")
    scope = _scope(headers=[(b"x-keep", b"1")])
    _run(mw, scope)
    assert app.called  # the proxy's own gate answers it; we do not
    assert app.scope["headers"] == [(b"x-keep", b"1")]


def test_wrong_proxy_token_is_left_untouched_and_costs_no_mint():
    app = _Recording()
    mw = OAuth2Middleware(app, _NeverMint(), proxy_token="s3cr3t")
    _run(mw, _scope(headers=[(b"authorization", b"Bearer nope")]))
    assert _auth_of(app) == b"Bearer nope"


def test_proxy_token_as_bearer_is_accepted_and_replaced_with_upstream_token():
    """The flow that used to 401: client authenticates with the proxy token in Authorization."""
    app = _Recording()
    mw = OAuth2Middleware(app, _Cached("TOK"), proxy_token="s3cr3t")
    _run(mw, _scope(headers=[(b"authorization", b"Bearer s3cr3t")]))
    hdrs = dict(app.scope["headers"])
    assert hdrs[b"authorization"] == b"Bearer TOK"
    # The proxy credential travels on to the core gate under its explicit header
    # (which the core strips before the upstream hop), so a gate that runs inside
    # this layer still authenticates the request.
    assert hdrs[b"x-headroom-proxy-token"] == b"s3cr3t"


def test_loopback_caller_gets_credential_carried_only_if_it_had_one():
    app = _Recording()
    mw = OAuth2Middleware(app, _Cached("TOK"), proxy_token="s3cr3t")
    _run(mw, _scope(headers=[], client=LOOPBACK))
    hdrs = dict(app.scope["headers"])
    assert hdrs[b"authorization"] == b"Bearer TOK"
    # Loopback is exempt at the gate too; the credential is still attached so the
    # request looks the same to the gate regardless of where this layer sits.
    assert hdrs[b"x-headroom-proxy-token"] == b"s3cr3t"


def test_no_proxy_token_configured_adds_no_header():
    app = _Recording()
    mw = OAuth2Middleware(app, _Cached("TOK"), proxy_token=None)
    _run(mw, _scope(headers=[]))
    assert b"x-headroom-proxy-token" not in dict(app.scope["headers"])


def test_explicit_proxy_token_header_is_accepted_and_preserved():
    app = _Recording()
    mw = OAuth2Middleware(app, _Cached("TOK"), proxy_token="s3cr3t")
    _run(
        mw,
        _scope(
            headers=[
                (b"x-headroom-proxy-token", b"s3cr3t"),
                (b"authorization", b"Bearer something-else"),
            ]
        ),
    )
    hdrs = dict(app.scope["headers"])
    assert hdrs[b"authorization"] == b"Bearer TOK"
    assert hdrs[b"x-headroom-proxy-token"] == b"s3cr3t"  # the core strips it before upstream


def test_loopback_caller_is_exempt_like_the_core_gate():
    app = _Recording()
    mw = OAuth2Middleware(app, _Cached("TOK"), proxy_token="s3cr3t")
    _run(mw, _scope(headers=[], client=LOOPBACK))
    assert _auth_of(app) == b"Bearer TOK"


def test_non_ascii_bearer_is_refused_not_crashed():
    app = _Recording()
    mw = OAuth2Middleware(app, _NeverMint(), proxy_token="s3cr3t")
    _run(mw, _scope(headers=[(b"authorization", "Bearer s3cr3té".encode("latin-1"))]))
    assert app.called


# --- install() wiring -------------------------------------------------------------


def _capture_install(monkeypatch, *, proxy_token_env=None, cfg_token=None, local_paths=None):
    monkeypatch.setenv("HEADROOM_OAUTH2_TOKEN_URL", "https://idp.example.com/token?client=acme")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_ID", "c")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_SECRET", "s")
    monkeypatch.delenv("HEADROOM_OAUTH2_HEADERS", raising=False)
    if proxy_token_env is None:
        monkeypatch.delenv("HEADROOM_PROXY_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HEADROOM_PROXY_TOKEN", proxy_token_env)
    if local_paths is None:
        monkeypatch.delenv("HEADROOM_OAUTH2_LOCAL_PATHS", raising=False)
    else:
        monkeypatch.setenv("HEADROOM_OAUTH2_LOCAL_PATHS", local_paths)
    captured = {}

    class App:
        def add_middleware(self, cls, **kw):
            captured["cls"] = cls
            captured.update(kw)

    cfg = type("Cfg", (), {"backend": "litellm-openai", "proxy_token": cfg_token})()
    install(App(), cfg)
    return captured


def test_install_passes_config_proxy_token(monkeypatch):
    cap = _capture_install(monkeypatch, cfg_token="from-config", proxy_token_env="from-env")
    assert cap["cls"] is OAuth2Middleware
    assert cap["proxy_token"] == "from-config"  # config wins, same precedence as the core


def test_install_falls_back_to_env_proxy_token(monkeypatch):
    cap = _capture_install(monkeypatch, proxy_token_env="from-env")
    assert cap["proxy_token"] == "from-env"


def test_install_passes_extra_local_paths(monkeypatch):
    cap = _capture_install(monkeypatch, local_paths="/vendor-local/, /other")
    assert cap["local_path_prefixes"] == ["/vendor-local/", "/other"]


def test_install_log_line_does_not_carry_the_token_url_query(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="headroom_oauth2")
    _capture_install(monkeypatch)
    [record] = [r for r in caplog.records if "auth installed" in r.getMessage()]
    # The logged token_url is exactly scheme://host -- no path, no query.
    assert record.args[0] == "https://idp.example.com"
    assert "client=acme" not in caplog.text
    assert "/token" not in caplog.text


# --- end to end through the real proxy app --------------------------------------


def test_remote_client_with_proxy_token_bearer_is_not_refused_when_oauth2_is_enabled(
    monkeypatch,
):
    """Through create_app: the 401-on-every-remote-request regression, plus /health with the IdP down."""
    server = pytest.importorskip("headroom.proxy.server")
    testclient = pytest.importorskip("fastapi.testclient")
    from headroom.proxy import extensions

    monkeypatch.setenv("HEADROOM_OAUTH2_TOKEN_URL", "https://idp.invalid/token")  # unreachable
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_ID", "c")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_SECRET", "s")
    monkeypatch.setenv("HEADROOM_OAUTH2_TIMEOUT", "1")
    monkeypatch.setattr(extensions, "discover", lambda: iter([("oauth2", install)]))

    config = server.ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        proxy_token="s3cr3t",
        proxy_extensions=["oauth2"],
    )
    app = server.create_app(config)
    with testclient.TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
        # Local routes: no mint, so the unreachable IdP is irrelevant.
        assert c.get("/health").status_code == 200
        assert c.get("/stats", headers={"Authorization": "Bearer s3cr3t"}).status_code == 200
        # Still gated: no credential, no access.
        assert c.get("/stats").status_code == 401
        # Upstream route with the proxy token: authenticates, then tries to mint -> 502
        # from oauth2 (IdP unreachable), NOT 401 from the gate.
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": []},
            headers={"Authorization": "Bearer s3cr3t"},
        )
        assert r.status_code == 502
        assert r.json()["error"]["type"] == "upstream_auth_error"


def test_bearer_proxy_token_survives_replacement_when_gate_runs_inside(monkeypatch):
    """IdP reachable: the request must clear the core gate even when this layer ran first.

    Registers a probe route under a non-local path so what reaches the handler is
    exactly what an upstream would have received, without any upstream.
    """
    server = pytest.importorskip("headroom.proxy.server")
    testclient = pytest.importorskip("fastapi.testclient")
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from headroom.proxy import extensions

    mints = {"n": 0}

    class IdP(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0) or 0))
            mints["n"] += 1
            body = json.dumps({"access_token": "MINTED", "expires_in": 3600}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), IdP)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv(
            "HEADROOM_OAUTH2_TOKEN_URL", f"http://127.0.0.1:{srv.server_address[1]}/token"
        )
        monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_ID", "c")
        monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_SECRET", "s")
        seen = {}

        def spy_install(app, config):
            install(app, config)

            @app.post("/probe/upstream-like")
            async def probe(request: Request):
                seen["authorization"] = request.headers.get("authorization")
                seen["proxy_header"] = request.headers.get("x-headroom-proxy-token")
                return {"ok": True}

        monkeypatch.setattr(extensions, "discover", lambda: iter([("oauth2", spy_install)]))
        config = server.ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            proxy_token="s3cr3t",
            proxy_extensions=["oauth2"],
        )
        app = server.create_app(config)
        with testclient.TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            r = c.post("/probe/upstream-like", headers={"Authorization": "Bearer s3cr3t"})
            assert r.status_code == 200, r.text
            assert seen["authorization"] == "Bearer MINTED"  # upstream bearer injected
            assert seen["proxy_header"] == "s3cr3t"  # proxy credential carried to the gate
            assert mints["n"] == 1

            seen.clear()
            r = c.post("/probe/upstream-like")  # no credential at all
            assert r.status_code == 401
            assert seen == {}  # never reached the handler
            assert mints["n"] == 1  # and cost no mint
    finally:
        srv.shutdown()
