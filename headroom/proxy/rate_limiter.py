"""Token bucket rate limiter for the Headroom proxy.

Rate limits requests and token usage per identity. Handlers pass a
:class:`~headroom.proxy.rate_limit_identity.RateLimitIdentity` (see that module
for who a request is charged to); a plain string is still accepted and treated
as a trusted identity that owns itself.

Bucket bookkeeping is built so an attacker cannot flush other callers' state:

* trusted and untrusted identities live in **separate** bounded LRU pools, so a
  flood of unauthenticated peers cannot evict an authenticated caller's bucket;
* one owner (a peer) can hold at most ``MAX_BUCKETS_PER_OWNER`` buckets in a
  pool; past that its new identities share one overflow bucket instead of
  evicting anyone else's.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

from headroom.proxy.models import RateLimitState
from headroom.proxy.rate_limit_identity import RateLimitIdentity
from headroom.proxy.rate_limit_policy import consume_from_bucket, refilled_tokens

# Maximum rate limiter buckets per pool (prevents memory DoS via spoofed keys).
MAX_RATE_LIMITER_BUCKETS = 1000
# Maximum distinct buckets one owner (peer) may hold in a pool before its new
# identities fold into a single overflow bucket.
MAX_BUCKETS_PER_OWNER = 32

_UNTRUSTED_PREFIX = "untrusted/"


class TokenBucketRateLimiter:
    """Token bucket rate limiter for requests and tokens."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int | None = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Request and token state share one bounded lifecycle per bucket, and no
        # hot-path operation scans all identities. ``_bucket_lru`` is the trusted
        # pool; untrusted buckets are namespaced and live in their own LRU.
        self._request_buckets: dict[str, RateLimitState] = {}
        self._token_buckets: dict[str, RateLimitState] = {}
        self._bucket_lru: OrderedDict[str, None] = OrderedDict()
        self._untrusted_lru: OrderedDict[str, None] = OrderedDict()
        self._bucket_owner: dict[str, str] = {}
        self._owner_buckets: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def _touch_bucket(self, identity: str | RateLimitIdentity) -> str:
        """Resolve ``identity`` to a bucket name, mark it active, and return it.

        Evicts the least-recently-used bucket of the same pool at capacity, and
        folds an owner's identities into one overflow bucket once it holds
        ``MAX_BUCKETS_PER_OWNER``.
        """
        if isinstance(identity, str):
            identity = RateLimitIdentity(bucket=identity, owner=identity, pool="trusted")
        untrusted = identity.pool == "untrusted"
        lru = self._untrusted_lru if untrusted else self._bucket_lru
        prefix = _UNTRUSTED_PREFIX if untrusted else ""
        owner_key = f"{identity.pool}:{identity.owner}"

        name = prefix + identity.bucket
        if name in lru:
            lru.move_to_end(name)
            return name
        if self._owner_buckets.get(owner_key, 0) >= MAX_BUCKETS_PER_OWNER:
            name = f"{prefix}{identity.owner}#overflow"
            if name in lru:
                lru.move_to_end(name)
                return name

        if len(lru) >= MAX_RATE_LIMITER_BUCKETS:
            evicted, _ = lru.popitem(last=False)
            self._request_buckets.pop(evicted, None)
            self._token_buckets.pop(evicted, None)
            evicted_owner = self._bucket_owner.pop(evicted, None)
            if evicted_owner is not None:
                remaining = self._owner_buckets.get(evicted_owner, 1) - 1
                if remaining > 0:
                    self._owner_buckets[evicted_owner] = remaining
                else:
                    self._owner_buckets.pop(evicted_owner, None)
        lru[name] = None
        self._bucket_owner[name] = owner_key
        self._owner_buckets[owner_key] = self._owner_buckets.get(owner_key, 0) + 1
        return name

    def _request_bucket(self, key: str) -> RateLimitState:
        state = self._request_buckets.get(key)
        if state is None:
            state = RateLimitState(tokens=self.requests_per_minute, last_update=time.time())
            self._request_buckets[key] = state
        return state

    def _token_bucket(self, key: str, capacity: int) -> RateLimitState:
        state = self._token_buckets.get(key)
        if state is None:
            state = RateLimitState(tokens=capacity, last_update=time.time())
            self._token_buckets[key] = state
        return state

    def _refill(self, state: RateLimitState, rate_per_minute: float) -> float:
        """Refill bucket based on elapsed time."""
        now = time.time()
        state.tokens = refilled_tokens(
            current_tokens=state.tokens,
            last_update=state.last_update,
            now=now,
            rate_per_minute=rate_per_minute,
        )
        state.last_update = now
        return state.tokens

    async def check_request(self, key: str | RateLimitIdentity = "default") -> tuple[bool, float]:
        """Check if request is allowed. Returns (allowed, wait_seconds)."""
        async with self._lock:
            bucket = self._touch_bucket(key)
            state = self._request_bucket(bucket)
            available = self._refill(state, self.requests_per_minute)

            allowed, state.tokens, wait_seconds = consume_from_bucket(
                available_tokens=available,
                requested_tokens=1,
                rate_per_minute=self.requests_per_minute,
            )
            return allowed, wait_seconds

    async def check_tokens(
        self, key: str | RateLimitIdentity, token_count: int
    ) -> tuple[bool, float]:
        """Check if token usage is allowed. ``tokens_per_minute=None`` never limits."""
        tpm = self.tokens_per_minute
        if tpm is None:
            return True, 0.0
        async with self._lock:
            bucket = self._touch_bucket(key)
            state = self._token_bucket(bucket, tpm)
            available = self._refill(state, tpm)

            allowed, state.tokens, wait_seconds = consume_from_bucket(
                available_tokens=available,
                requested_tokens=token_count,
                rate_per_minute=tpm,
            )
            return allowed, wait_seconds

    async def stats(self) -> dict:
        """Get rate limiter statistics."""
        async with self._lock:
            return {
                "requests_per_minute": self.requests_per_minute,
                "tokens_per_minute": self.tokens_per_minute,
                "active_keys": len(self._bucket_lru) + len(self._untrusted_lru),
            }
