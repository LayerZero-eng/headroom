# headroom-oauth2

Generic **OAuth2 client-credentials** upstream-auth extension for the
[Headroom](https://github.com/headroomlabs-ai/headroom) proxy.

When Headroom routes to an OpenAI-compatible backend that is protected by an
OAuth2 client-credentials flow (enterprise AI gateways, Azure AD / Entra, Okta,
Auth0, Keycloak, Cognito, …), this extension mints a bearer token from a
configurable token endpoint, caches + refreshes it (single-flight), and injects
`Authorization: Bearer <token>` on each upstream request. Optional static upstream
headers are sent via litellm. **Fully vendor-neutral — no provider is hard-coded.**

It plugs into Headroom's public `headroom.proxy_extension` entry-point seam, so it
is fully out-of-tree and opt-in.

## Install & enable
```bash
pip install headroom-oauth2
headroom proxy --backend litellm-openai --proxy-extension oauth2
```

## Configure (env; no-op unless HEADROOM_OAUTH2_TOKEN_URL is set)
| Env | Meaning |
|-----|---------|
| `HEADROOM_OAUTH2_TOKEN_URL` | token endpoint (client_credentials grant) |
| `HEADROOM_OAUTH2_CLIENT_ID` / `_CLIENT_SECRET` | credentials (secrets) |
| `HEADROOM_OAUTH2_SCOPES` | space/comma-separated scopes |
| `HEADROOM_OAUTH2_AUDIENCE` | optional audience |
| `HEADROOM_OAUTH2_GRANT_TYPE` | default `client_credentials` |
| `HEADROOM_OAUTH2_AUTH_STYLE` | `post` (form creds) or `basic` (HTTP Basic) |
| `HEADROOM_OAUTH2_HEADERS` | static upstream headers, `K=V,K2=V2` |
| `HEADROOM_OAUTH2_LOCAL_PATHS` | extra path prefixes the proxy serves locally (never injected), `,`-separated |

Tokens are minted with the standard library (`urllib`, system cert store), which
works behind corporate SSL-inspection where bundled-root TLS stacks fail.


**Effective backends:** the injected bearer reaches the upstream only for OpenAI-compatible /
passthrough litellm providers. `bedrock` / `vertex` / `sagemaker` authenticate from env and
ignore it, so this extension is a no-op there (it logs a warning at startup).

**Transport:** `token_url` must be `https` (loopback `http` is allowed for tests; set
`HEADROOM_OAUTH2_ALLOW_INSECURE=1` to override). Tokens are minted with the standard library
(`urllib`, system cert store), so a corporate-injected CA is trusted without bundling roots.

## What gets the upstream bearer, and when

The minted token is injected only on requests that **go upstream**. Routes the
proxy answers itself — `/health`, `/livez`, `/readyz`, `/stats*`, `/metrics`,
`/quota`, `/settings*`, `/dashboard*`, `/admin/*`, `/debug/*`, `/v1/compress*`,
`/v1/usage`, `/v1/retrieve*`, `/v1/telemetry*`, `/v1/toin*`, `/v1/feedback*`,
and any extension route under `/ext/` — are passed through untouched, so an
unreachable IdP never turns a health probe into a 502 and management calls never
cost a token mint. A `/p/<project>/` base-URL prefix is stripped before the
check. Add your own local prefixes with `HEADROOM_OAUTH2_LOCAL_PATHS`.

**With `HEADROOM_PROXY_TOKEN` set:** a remote client authenticates to the proxy
either with `Authorization: Bearer <proxy token>` or with the explicit
`x-headroom-proxy-token: <proxy token>` header (use the latter when the client
needs `Authorization` for something else). The credential is verified here, the
upstream bearer replaces `Authorization`, and the proxy credential is carried on
to the core's gate as `x-headroom-proxy-token` — which the core strips before the
upstream hop, so it never leaves the host. A request that has not authenticated
is left exactly as it arrived for the proxy's own gate to refuse; no token is
minted for it. Loopback callers are exempt, matching the core.

Requires `headroom-ai` ≥ 0.40 for extension middleware to run inside the proxy's
inbound gate; on older cores this extension applies the same token check itself.
