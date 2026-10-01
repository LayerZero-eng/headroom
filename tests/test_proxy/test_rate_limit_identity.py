"""Rate-limit identity and eviction-resistant bucket bookkeeping (VAPT 01-F3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.proxy.rate_limit_identity import RateLimitIdentity, rate_limit_identity
from headroom.proxy.rate_limiter import (
    MAX_BUCKETS_PER_OWNER,
    MAX_RATE_LIMITER_BUCKETS,
    TokenBucketRateLimiter,
)


def _request(peer: str, *, authenticated: bool = False, headers=None):
    return SimpleNamespace(
        client=SimpleNamespace(host=peer),
        headers=headers or {},
        state=SimpleNamespace(proxy_authenticated=authenticated),
    )


# ── identity ────────────────────────────────────────────────────────────────


def test_untrusted_identity_ignores_the_credential() -> None:
    a = rate_limit_identity(_request("203.0.113.9", headers={"authorization": "Bearer k1"}))
    b = rate_limit_identity(_request("203.0.113.9", headers={"x-api-key": "k2"}))
    c = rate_limit_identity(_request("203.0.113.9"))
    assert a == b == c
    assert a.pool == "untrusted"


def test_trusted_identity_is_per_credential_and_owned_by_the_peer() -> None:
    a = rate_limit_identity(
        _request("203.0.113.9", authenticated=True, headers={"authorization": "Bearer k1"})
    )
    b = rate_limit_identity(
        _request("203.0.113.9", authenticated=True, headers={"authorization": "Bearer k2"})
    )
    assert a.bucket != b.bucket
    assert a.owner == b.owner == "peer:203.0.113.9"
    assert a.pool == b.pool == "trusted"


@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": "Bearer sk-ant-api03-SAME-PREFIX-A"},
        {"x-api-key": "sk-ant-api03-SAME-PREFIX-A"},
        {"x-goog-api-key": "AIzaSyD-SAME-PREFIX-A"},
    ],
)
def test_trusted_identity_uses_the_whole_credential_not_a_prefix(headers) -> None:
    other = {k: v[:-1] + "B" for k, v in headers.items()}
    a = rate_limit_identity(_request("127.0.0.1", headers=headers))
    b = rate_limit_identity(_request("127.0.0.1", headers=other))
    assert a.bucket != b.bucket


def test_bucket_names_carry_no_credential_material() -> None:
    ident = rate_limit_identity(
        _request("127.0.0.1", headers={"authorization": "Bearer sk-live-secret-value"})
    )
    assert "sk-live" not in ident.bucket
    assert "secret" not in ident.bucket


def test_loopback_is_trusted_without_a_token() -> None:
    assert rate_limit_identity(_request("127.0.0.1")).pool == "trusted"


def test_trusted_gateway_peer_is_trusted(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "10.20.0.0/16")
    assert rate_limit_identity(_request("10.20.3.4")).pool == "trusted"
    assert rate_limit_identity(_request("10.21.3.4")).pool == "untrusted"


def test_ipv6_peers_are_grouped_by_slash_64() -> None:
    a = rate_limit_identity(_request("2001:db8:1:2::1"))
    b = rate_limit_identity(_request("2001:db8:1:2:ffff:ffff:ffff:ffff"))
    c = rate_limit_identity(_request("2001:db8:1:3::1"))
    assert a == b
    assert a != c


def test_ipv4_mapped_ipv6_is_the_ipv4_peer() -> None:
    assert rate_limit_identity(_request("::ffff:203.0.113.9")) == rate_limit_identity(
        _request("203.0.113.9")
    )


# ── limiter bookkeeping ─────────────────────────────────────────────────────


def _ident(owner: str, n: int | None = None, pool: str = "trusted") -> RateLimitIdentity:
    bucket = owner if n is None else f"{owner}|cred:{n}"
    return RateLimitIdentity(bucket=bucket, owner=owner, pool=pool)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_untrusted_flood_cannot_evict_a_trusted_bucket() -> None:
    limiter = TokenBucketRateLimiter(requests_per_minute=1)
    victim = _ident("peer:127.0.0.1", 1)
    assert (await limiter.check_request(victim))[0] is True
    for i in range(MAX_RATE_LIMITER_BUCKETS * 2):
        await limiter.check_request(_ident(f"peer:198.51.{i // 256}.{i % 256}", pool="untrusted"))
    # Still limited: the victim's spent bucket survived the flood.
    assert (await limiter.check_request(victim))[0] is False


@pytest.mark.asyncio
async def test_one_owner_rotating_credentials_cannot_evict_other_owners() -> None:
    limiter = TokenBucketRateLimiter(requests_per_minute=1)
    victim = _ident("peer:203.0.113.1", 1)
    assert (await limiter.check_request(victim))[0] is True
    attacker = "peer:203.0.113.66"
    for n in range(MAX_RATE_LIMITER_BUCKETS * 2):
        await limiter.check_request(_ident(attacker, n))
    assert (await limiter.check_request(victim))[0] is False
    assert limiter._owner_buckets[f"trusted:{attacker}"] <= MAX_BUCKETS_PER_OWNER + 1


@pytest.mark.asyncio
async def test_owner_past_the_cap_shares_one_overflow_bucket() -> None:
    limiter = TokenBucketRateLimiter(requests_per_minute=1)
    owner = "peer:203.0.113.66"
    for n in range(MAX_BUCKETS_PER_OWNER):
        assert (await limiter.check_request(_ident(owner, n)))[0] is True
    # First identity past the cap lands in the overflow bucket and spends it...
    assert (await limiter.check_request(_ident(owner, 10_000)))[0] is True
    # ...so the next fresh identity is limited, not granted a new bucket.
    assert (await limiter.check_request(_ident(owner, 10_001)))[0] is False


@pytest.mark.asyncio
async def test_plain_string_keys_still_work() -> None:
    limiter = TokenBucketRateLimiter(requests_per_minute=1)
    assert (await limiter.check_request("client"))[0] is True
    assert (await limiter.check_request("client"))[0] is False
