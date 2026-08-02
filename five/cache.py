"""
Two-tier (memory + disk) TTL cache with a last-good-snapshot fallback.

Adapted from the Tattva system's ``data/cache.py``.

The important property is the *stale fallback*. A naive cache has two states —
fresh or gone — so when the provider is down the UI shows nothing. Here an
expired entry is kept on disk and can still be served explicitly via
:meth:`Cache.get_stale`, so an outage degrades to "yesterday's data, clearly
labelled" instead of an error page.

The force-refresh window is scoped **per Streamlit session**. Streamlit serves
every concurrent session from one process, so a single global flag would mean
one user pressing "Refresh" forces every other session to bypass its cache too
— amplifying rate-limit exposure for people who never asked for it.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fve"
DEFAULT_TTL_SECONDS = 6 * 3600
SNAPSHOT_RETENTION_DAYS = 10

# session id -> unix time until which that session bypasses the fresh cache
_FORCE_UNTIL: dict[str, float] = {}
_MAX_TRACKED_SESSIONS = 50


NO_SESSION = "_no_session_"


def current_session_key() -> str:
    """Streamlit session id, or a shared key outside a script run.

    If ``streamlit`` is not already in ``sys.modules`` we are by definition not
    inside a Streamlit script run, so there is no session to look up. Checking
    that first matters: importing Streamlit costs the better part of a second,
    and this runs on every cache read — a headless caller (research script,
    notebook, test) would otherwise pay that import just to be told there is no
    session. Inside the app Streamlit is already imported, so the check is free.
    """
    if "streamlit" not in sys.modules:
        return NO_SESSION
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            ctx = get_script_run_ctx(suppress_warning=True)
        except TypeError:
            ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
    except Exception:  # noqa: BLE001 — running headless is normal
        pass
    return NO_SESSION


def begin_force_refresh(window_seconds: float = 120.0) -> None:
    """Make this session bypass the fresh cache for a short window.

    The disk snapshot is deliberately preserved, so a forced refresh that then
    fails degrades to stale data rather than to nothing.
    """
    _FORCE_UNTIL[current_session_key()] = time.time() + window_seconds
    while len(_FORCE_UNTIL) > _MAX_TRACKED_SESSIONS:
        _FORCE_UNTIL.pop(next(iter(_FORCE_UNTIL)))


def force_refresh_active() -> bool:
    return time.time() < _FORCE_UNTIL.get(current_session_key(), 0.0)


class Cache:
    """TTL cache with a memory tier, a disk tier and a stale fallback.

    Keys are an MD5 of ``(version, *args)``; bumping ``version`` invalidates
    the whole namespace atomically.
    """

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS, disk_dir: Path | None = None,
                 version: str = "v1", namespace: str = "") -> None:
        self.ttl = ttl
        self.version = version
        self._memory: dict[str, tuple[Any, float]] = {}
        base = Path(disk_dir) if disk_dir else DEFAULT_CACHE_DIR
        self._disk_dir = base / namespace if namespace else base
        try:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 — cache is an optimisation
            log.warning("Cache directory unavailable (%s); memory tier only", exc)
        self._lock = threading.Lock()

        self.hits = 0
        self.misses = 0
        self.stale_hits = 0
        self.writes = 0
        self.last_write_time: float | None = None

    # -- keys ---------------------------------------------------------------
    def _key(self, *args: Any) -> str:
        raw = f"{self.version}|" + "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._disk_dir / f"{key}.pkl"

    # -- read ---------------------------------------------------------------
    def get(self, *args: Any) -> Any | None:
        """Return the value if it exists and is within TTL, else ``None``."""
        if force_refresh_active():
            with self._lock:
                self.misses += 1
            return None

        key = self._key(*args)
        now = time.time()

        with self._lock:
            entry = self._memory.get(key)
            if entry is not None and (now - entry[1]) < self.ttl:
                self.hits += 1
                return entry[0]

        path = self._path(key)
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    value, written = pickle.load(fh)
                if (now - written) < self.ttl:
                    with self._lock:
                        self._memory[key] = (value, written)
                        self.hits += 1
                    return value
            except Exception as exc:  # noqa: BLE001 — corrupt entry is a miss
                log.debug("Discarding unreadable cache entry %s: %s", key, exc)

        with self._lock:
            self.misses += 1
        return None

    def get_stale(self, *args: Any) -> Any | None:
        """Return the last-good value **ignoring TTL**, for failure fallback."""
        key = self._key(*args)
        with self._lock:
            entry = self._memory.get(key)
        if entry is not None:
            with self._lock:
                self.stale_hits += 1
            return entry[0]

        path = self._path(key)
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    value, _ = pickle.load(fh)
                with self._lock:
                    self.stale_hits += 1
                return value
            except Exception:  # noqa: BLE001
                return None
        return None

    def age_seconds(self, *args: Any) -> float | None:
        """Seconds since the entry was written, or ``None`` if absent."""
        key = self._key(*args)
        with self._lock:
            entry = self._memory.get(key)
        if entry is not None:
            return time.time() - entry[1]
        path = self._path(key)
        if path.exists():
            try:
                return time.time() - path.stat().st_mtime
            except Exception:  # noqa: BLE001
                return None
        return None

    # -- write --------------------------------------------------------------
    def put(self, *args: Any, value: Any) -> None:
        key = self._key(*args)
        now = time.time()
        with self._lock:
            self._memory[key] = (value, now)
            self.writes += 1
            self.last_write_time = now
        try:
            with open(self._path(key), "wb") as fh:
                pickle.dump((value, now), fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist cache entry: %s", exc)
        self.prune()

    def prune(self, retention_days: int = SNAPSHOT_RETENTION_DAYS) -> None:
        """Delete disk snapshots older than the retention window."""
        cutoff = time.time() - retention_days * 86400
        try:
            for p in self._disk_dir.glob("*.pkl"):
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def snapshots_newest_first(self) -> list[Any]:
        """Every persisted value in this namespace, newest first.

        Used to backfill instruments a partial fetch dropped.
        """
        out: list[Any] = []
        try:
            paths = sorted(self._disk_dir.glob("*.pkl"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:  # noqa: BLE001
            return out
        for p in paths:
            try:
                with open(p, "rb") as fh:
                    value, _ = pickle.load(fh)
                out.append(value)
            except Exception:  # noqa: BLE001
                continue
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits, "misses": self.misses,
                "stale_hits": self.stale_hits, "writes": self.writes,
                "hit_rate": (self.hits / total) if total else 0.0,
                "entries": len(self._memory),
            }


# Namespaced caches. Bump ``version`` to invalidate a namespace wholesale.
panel_cache = Cache(ttl=6 * 3600, version="v1", namespace="panel")
target_cache = Cache(ttl=6 * 3600, version="v1", namespace="target")
# A resolved symbol is a near-permanent fact; a failed one is not — a listing
# that does not exist today may exist next week, so failures expire quickly.
symbol_cache = Cache(ttl=30 * 86400, version="v1", namespace="symbols")
symbol_fail_cache = Cache(ttl=600, version="v1", namespace="symbol_fail")
