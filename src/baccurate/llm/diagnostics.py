"""Trace and measure LLM calls, accumulating totals over a single run."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)
trace_logger = logging.getLogger("baccurate.llm_trace")

_SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
}
_FILE_ONLY_TRACE_EXTRA = {"llm_file_only": True, "llm_trace_unbounded": True}
_UNBOUNDED_TRACE_EXTRA = {"llm_trace_unbounded": True}


@dataclass(slots=True)
class _TraceContext:
    accession: str
    target: str
    model: str
    attempt: int = 0


class LLMFailureCategory(StrEnum):
    HTTP_CLIENT_ERROR = "http_client_error"
    HTTP_SERVER_ERROR = "http_server_error"
    HTTP_OTHER_ERROR = "http_other_error"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    TRANSPORT = "transport"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    UNEXPECTED = "unexpected"


@dataclass(slots=True)
class _MetricBucket:
    attempts: int = 0
    successes: int = 0
    failures: Counter[LLMFailureCategory] = field(default_factory=Counter)
    retries: int = 0
    duration_seconds: float = 0.0
    reported_usage_calls: int = 0
    unavailable_usage_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    pending_successful_responses: int = 0


class _LLMMetrics:
    """LLM call metrics for one run, grouped by (target, model).

    Tracks:
    - attempts
    - retries
    - durations
    - token usage
    - failures
    """

    def __init__(self, model_identifiers: Mapping[str, str | None]) -> None:
        self._default_models = dict(model_identifiers)
        self._buckets: dict[tuple[str, str | None], _MetricBucket] = {
            (target, model): _MetricBucket() for target, model in model_identifiers.items()
        }

    def record_attempt(
        self,
        *,
        target: str,
        model: str,
        attempt: int,
        duration_seconds: float,
        response: httpx.Response | None,
        exception: Exception | None,
    ) -> None:
        bucket = self._buckets.setdefault((target, model or None), _MetricBucket())
        bucket.attempts += 1
        bucket.retries += int(attempt > 1)
        bucket.duration_seconds += duration_seconds

        if response is None:
            bucket.failures[_exception_category(exception)] += 1
            return
        if not 200 <= response.status_code < 300:
            bucket.failures[_status_category(response.status_code)] += 1
            return

        bucket.pending_successful_responses += 1
        usage = _reported_usage(response)
        if usage is None:
            bucket.unavailable_usage_calls += 1
            return
        bucket.reported_usage_calls += 1
        bucket.prompt_tokens += usage.get("prompt_tokens", 0)
        bucket.completion_tokens += usage.get("completion_tokens", 0)
        bucket.total_tokens += usage.get("total_tokens", 0)

    def record_success(self, *, target: str, model: str) -> None:
        """Record one model result accepted by its caller."""
        bucket = self._buckets.setdefault((target, model or None), _MetricBucket())
        invalid_retries = max(0, bucket.pending_successful_responses - 1)
        if invalid_retries:
            bucket.failures[LLMFailureCategory.INVALID_MODEL_RESPONSE] += invalid_retries
        bucket.pending_successful_responses = 0
        bucket.successes += 1

    def record_failure(self, *, target: str, model: str, category: LLMFailureCategory) -> None:
        """Record a failure that the transport counted as a successful call.

        The HTTP request itself succeeded, but the result was unusable — an
        invalid model response, or the SDK giving up after its retries. Any
        successful responses still pending on this (target, model) bucket are
        reclassified under `category` (at least one, even if none are pending).
        """
        bucket = self._buckets.setdefault((target, model or None), _MetricBucket())
        failure_count = max(1, bucket.pending_successful_responses)
        bucket.pending_successful_responses = 0
        bucket.failures[category] += failure_count

    def record_pending_failures(
        self, *, target: str, model: str, category: LLMFailureCategory
    ) -> None:
        """Charge any still-pending successful responses to a terminal failure.

        When a call ultimately fails, the HTTP requests that came back 200 but
        were rejected along the way are counted under `category`. Unlike
        record_failure, this adds only the pending count, with no minimum of one.
        """
        bucket = self._buckets.setdefault((target, model or None), _MetricBucket())
        bucket.failures[category] += bucket.pending_successful_responses
        bucket.pending_successful_responses = 0

    def snapshot(self, cache_hits: Mapping[str, int] | None = None) -> dict[str, object]:
        """Return stable JSON values, adding build-report cache counts by target."""
        cache_hits = cache_hits or {}
        entries = []
        aggregate = _MetricBucket()
        for (target, model), bucket in sorted(
            self._buckets.items(), key=lambda item: (item[0][0], item[0][1] or "")
        ):
            target_cache_hits = (
                int(cache_hits.get(target, 0)) if model == self._default_models.get(target) else 0
            )
            entries.append(
                {
                    "target": target,
                    "model": model,
                    **_metric_document(bucket, cache_hits=target_cache_hits),
                }
            )
            _merge_metric_bucket(aggregate, bucket)
        return {
            "aggregate": _metric_document(
                aggregate,
                cache_hits=sum(int(value) for value in cache_hits.values()),
            ),
            "by_target_and_model": entries,
        }


_trace_context: ContextVar[_TraceContext | None] = ContextVar("llm_trace_context", default=None)
_observability_context: ContextVar[LLMObservability | None] = ContextVar(
    "llm_observability",
    default=None,
)


class LLMObservability:
    """Hold the measurements and call context for one run."""

    def __init__(
        self,
        model_identifiers: Mapping[str, str | None],
        *,
        trace_enabled: bool = False,
    ) -> None:
        self._metrics = _LLMMetrics(model_identifiers)
        self._trace_enabled = trace_enabled
        self._seen_system_prompts: set[str] = set()
        self._token: Token[LLMObservability | None] | None = None

    def start(self) -> None:
        if self._token is not None:
            raise RuntimeError("LLM observability is already active")
        self._token = _observability_context.set(self)

    def snapshot(self, cache_hits: Mapping[str, int] | None = None) -> dict[str, object]:
        return self._metrics.snapshot(cache_hits)

    def close(self) -> None:
        if self._token is not None:
            _observability_context.reset(self._token)
            self._token = None

    def _record_attempt(
        self,
        *,
        target: str,
        model: str,
        attempt: int,
        duration_seconds: float,
        response: httpx.Response | None,
        exception: Exception | None,
    ) -> None:
        self._metrics.record_attempt(
            target=target,
            model=model,
            attempt=attempt,
            duration_seconds=duration_seconds,
            response=response,
            exception=exception,
        )

    def _traces_calls(self) -> bool:
        return self._trace_enabled

    def _claim_system_prompt(self, prompt_id: str) -> bool:
        is_new = prompt_id not in self._seen_system_prompts
        self._seen_system_prompts.add(prompt_id)
        return is_new


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One model call whose semantic outcome is classified by its caller."""

    target: str
    model: str
    _observability: LLMObservability | None

    def accepted(self) -> None:
        """Record that the caller accepted the model response."""
        if self._observability is not None:
            self._observability._metrics.record_success(target=self.target, model=self.model)

    def failed(self, category: LLMFailureCategory) -> None:
        """Classify a semantic failure not visible from HTTP status alone."""
        if self._observability is not None:
            self._observability._metrics.record_failure(
                target=self.target,
                model=self.model,
                category=category,
            )

    def validation_retries_exhausted(self) -> None:
        """Classify successful responses rejected by model validation."""
        if self._observability is not None:
            self._observability._metrics.record_pending_failures(
                target=self.target,
                model=self.model,
                category=LLMFailureCategory.INVALID_MODEL_RESPONSE,
            )


@contextmanager
def observe_llm_call(*, accession: str, target: str, model: str) -> Iterator[LLMCall]:
    """Observe transport attempts and let the caller classify their outcome."""
    call = LLMCall(target, model, _observability_context.get())
    token = _trace_context.set(_TraceContext(accession, target, model))
    try:
        yield call
    finally:
        _trace_context.reset(token)


class _Sanitizer:
    """Keep known credentials and sensitive headers out of the trace log."""

    def __init__(self, secrets: set[str]) -> None:
        self.secrets = sorted((secret for secret in secrets if secret), key=len, reverse=True)

    def text(self, value: str) -> str:
        for secret in self.secrets:
            value = value.replace(secret, "<redacted>")
        return value

    def value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): self.value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value

    def headers(self, headers: httpx.Headers) -> dict[str, str]:
        return {
            key.lower(): "<redacted>" if key.lower() in _SENSITIVE_HEADERS else self.text(value)
            for key, value in headers.multi_items()
        }

    def url(self, url: httpx.URL) -> tuple[str, dict[str, str]]:
        split = urlsplit(str(url))
        host = split.hostname or ""
        if split.port is not None:
            host = f"{host}:{split.port}"
        if split.username is not None or split.password is not None:
            host = f"<redacted>:<redacted>@{host}"
        query = {
            key: self.text(value) for key, value in parse_qsl(split.query, keep_blank_values=True)
        }
        sanitized_url = urlunsplit(
            (split.scheme, host, split.path, urlencode(query), split.fragment)
        )
        return self.text(sanitized_url), query


class ObservedLLMTransport(httpx.BaseTransport):
    """Observe actual synchronous HTTP attempts without enabling SDK logging."""

    def __init__(
        self,
        transport: httpx.BaseTransport,
        *,
        configured_secrets: set[str] | None = None,
    ) -> None:
        self._transport = transport
        self._sanitizer = _Sanitizer(configured_secrets or set())

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        context = _trace_context.get()
        observability = _observability_context.get()
        if context is None or observability is None:
            return self._transport.handle_request(request)

        context.attempt += 1
        started = monotonic()
        try:
            response = self._transport.handle_request(request)
            response.read()
        except Exception as error:
            duration = monotonic() - started
            observability._record_attempt(
                target=context.target,
                model=context.model,
                attempt=context.attempt,
                duration_seconds=duration,
                response=None,
                exception=error,
            )
            if observability._traces_calls():
                self._log(request, context, observability, duration, exception=error)
            raise
        duration = monotonic() - started
        observability._record_attempt(
            target=context.target,
            model=context.model,
            attempt=context.attempt,
            duration_seconds=duration,
            response=response,
            exception=None,
        )
        if observability._traces_calls():
            self._log(request, context, observability, duration, response=response)
        return response

    def close(self) -> None:
        self._transport.close()

    def _body(self, content: bytes) -> Any:
        text = content.decode("utf-8", errors="replace")
        try:
            return self._sanitizer.value(json.loads(text))
        except json.JSONDecodeError:
            return self._sanitizer.text(text)

    def _request_content(
        self,
        request: httpx.Request,
        observability: LLMObservability,
    ) -> Any:
        content = self._body(request.content)
        if not isinstance(content, dict):
            return content
        messages = content.get("messages")
        if not isinstance(messages, list):
            return content
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            prompt = message.get("content")
            if not isinstance(prompt, str):
                continue
            prompt_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if observability._claim_system_prompt(prompt_id):
                trace_logger.debug(
                    "LLM system prompt id=%s content=%s",
                    prompt_id,
                    self._sanitizer.text(prompt),
                    extra=_FILE_ONLY_TRACE_EXTRA,
                )
            message["content"] = {"system_prompt_id": prompt_id}
        return content

    def _log(
        self,
        request: httpx.Request,
        context: _TraceContext,
        observability: LLMObservability,
        duration: float,
        *,
        response: httpx.Response | None = None,
        exception: Exception | None = None,
    ) -> None:
        endpoint, query = self._sanitizer.url(request.url)
        request_content = self._request_content(request, observability)
        parameters = (
            {key: value for key, value in request_content.items() if key != "messages"}
            if isinstance(request_content, dict)
            else {}
        )
        payload: dict[str, Any] = {
            "accession": self._sanitizer.text(context.accession),
            "target": context.target,
            "model": context.model,
            "attempt": context.attempt,
            "duration_seconds": round(duration, 6),
            "method": request.method,
            "endpoint": endpoint,
            "query_parameters": query,
            "parameters": parameters,
            "request_headers": self._sanitizer.headers(request.headers),
            "request_content": request_content,
        }
        status: str | int
        if response is not None:
            status = response.status_code
            payload.update(
                {
                    "status": response.status_code,
                    "response_headers": self._sanitizer.headers(response.headers),
                    "response_content": self._body(response.content),
                }
            )
        else:
            status = type(exception).__name__
            payload["exception"] = {
                "type": type(exception).__name__,
                "details": self._sanitizer.text(str(exception)),
            }
        trace_logger.debug(
            "LLM trace %s",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            extra=_FILE_ONLY_TRACE_EXTRA,
        )
        trace_logger.debug(
            "LLM call target=%s model=%s attempt=%d status=%s duration=%.3fs",
            context.target,
            context.model,
            context.attempt,
            status,
            duration,
            extra=_UNBOUNDED_TRACE_EXTRA,
        )


def _status_category(status_code: int) -> LLMFailureCategory:
    if 400 <= status_code < 500:
        return LLMFailureCategory.HTTP_CLIENT_ERROR
    if 500 <= status_code < 600:
        return LLMFailureCategory.HTTP_SERVER_ERROR
    return LLMFailureCategory.HTTP_OTHER_ERROR


def _exception_category(exception: Exception | None) -> LLMFailureCategory:
    if isinstance(exception, httpx.TimeoutException):
        return LLMFailureCategory.TIMEOUT
    if isinstance(exception, httpx.NetworkError):
        return LLMFailureCategory.CONNECTION
    if isinstance(exception, httpx.HTTPError):
        return LLMFailureCategory.TRANSPORT
    return LLMFailureCategory.UNEXPECTED


def _reported_usage(response: httpx.Response) -> dict[str, int] | None:
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(body, Mapping) or not isinstance(body.get("usage"), Mapping):
        return None
    usage = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = body["usage"].get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            usage[name] = value
    return usage or None


def _merge_metric_bucket(total: _MetricBucket, bucket: _MetricBucket) -> None:
    total.attempts += bucket.attempts
    total.successes += bucket.successes
    total.failures.update(bucket.failures)
    total.retries += bucket.retries
    total.duration_seconds += bucket.duration_seconds
    total.reported_usage_calls += bucket.reported_usage_calls
    total.unavailable_usage_calls += bucket.unavailable_usage_calls
    total.prompt_tokens += bucket.prompt_tokens
    total.completion_tokens += bucket.completion_tokens
    total.total_tokens += bucket.total_tokens


def _metric_document(bucket: _MetricBucket, *, cache_hits: int) -> dict[str, object]:
    return {
        "attempts": bucket.attempts,
        "successes": bucket.successes,
        "failures": dict(sorted(bucket.failures.items())),
        "cache_hits": cache_hits,
        "retries": bucket.retries,
        "duration_seconds": round(bucket.duration_seconds, 6),
        "token_usage": {
            "reported_calls": bucket.reported_usage_calls,
            "unavailable_calls": bucket.unavailable_usage_calls,
            "prompt_tokens": bucket.prompt_tokens,
            "completion_tokens": bucket.completion_tokens,
            "total_tokens": bucket.total_tokens,
        },
    }
