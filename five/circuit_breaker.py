"""
Fault-tolerance primitives for external data calls.

Adapted from the Tattva system's ``data/circuit_breaker.py``.

Two independent mechanisms, deliberately layered:

  * **Retry with backoff** handles *transient* failures — a DNS hiccup, a
    momentary 5xx. It retries the same call a couple of times with growing
    delays.

  * **The circuit breaker** handles *sustained* failures. Once a service has
    failed repeatedly there is no value in continuing to wait 30 seconds per
    request for it: the circuit opens and subsequent calls fail immediately,
    letting the caller fall back to cached data. After a recovery timeout one
    probe call is allowed through; if it succeeds the circuit closes again.

State machine::

    CLOSED    → OPEN:       failure_count >= failure_threshold
    OPEN      → HALF_OPEN:  recovery_timeout elapsed since the last failure
    HALF_OPEN → CLOSED:     the probe call succeeded
    HALF_OPEN → OPEN:       the probe call also failed
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable

log = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when a call is blocked because the circuit is OPEN."""


class CircuitBreaker:
    """Per-service circuit breaker. Thread-safe."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 1, name: str = "default") -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            if self.state is CircuitState.OPEN:
                if self.last_failure_time is None:
                    raise CircuitBreakerError(f"Circuit '{self.name}' is OPEN")
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is OPEN — retry in "
                        f"{self.recovery_timeout - elapsed:.0f}s")

            if self.state is CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls > self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' HALF_OPEN — probe already in flight")

        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def _on_success(self) -> None:
        with self._lock:
            previous = self.state
            self.success_count += 1
            self.last_success_time = time.time()
            if self.state is CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.half_open_calls = 0
            # The threshold counts *consecutive* failures, so any success
            # clears the tally. Without this, isolated failures accumulate
            # across an otherwise healthy session — five bad tickers spread
            # over an hour would trip the breaker on a provider that is fine.
            self.failure_count = 0
        if previous is CircuitState.HALF_OPEN:
            log.info("Circuit '%s' CLOSED — service recovered", self.name)

    def _on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state is CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                log.warning("Circuit '%s' probe failed — back to OPEN", self.name)
            elif (self.state is CircuitState.CLOSED
                  and self.failure_count >= self.failure_threshold):
                self.state = CircuitState.OPEN
                log.warning("Circuit '%s' OPEN after %d consecutive failures",
                            self.name, self.failure_count)

    def reset(self) -> None:
        """Force the circuit closed — used by the UI's manual refresh action."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name, "state": self.state.value,
                "failures": self.failure_count, "successes": self.success_count,
                "last_failure": self.last_failure_time,
                "last_success": self.last_success_time,
            }


class RetryWithBackoff:
    """Decorator retrying a call with exponentially growing delays."""

    def __init__(self, max_retries: int = 2, initial_delay: float = 1.5,
                 backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = self.initial_delay
            last: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 — any transport failure
                    last = exc
                    if attempt < self.max_retries:
                        log.debug("%s failed (attempt %d/%d): %s — retrying in %.1fs",
                                  func.__name__, attempt + 1, self.max_retries + 1,
                                  exc, delay)
                        time.sleep(delay)
                        delay *= self.backoff_factor
            raise last  # type: ignore[misc]
        return wrapper


# One breaker per external service, shared process-wide.
yfinance_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=90.0,
                                  name="yfinance")
