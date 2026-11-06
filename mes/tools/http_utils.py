"""
HTTP helpers for MES evidence sources.

* one ``requests`` session per thread, with retries on transient errors;
* per-host minimum interval between requests (NCBI allows 3 requests/s without a key);
* a sqlite response cache (``.cache/http.sqlite``) so repeated questions do not hit the APIs again.
"""

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mes import config

USER_AGENT = "MES/2 (evidence synthesis research prototype)"

# Minimum seconds between two requests to the same host
_MIN_INTERVAL = {
    "eutils.ncbi.nlm.nih.gov": 0.11 if config.NCBI_API_KEY else 0.34,
    "www.ebi.ac.uk": 0.1,
    "clinicaltrials.gov": 0.1,
    "api.unpaywall.org": 0.1,
}

_local = threading.local()
_host_locks = {}
_host_last = {}
_host_guard = threading.Lock()


@dataclass
class HttpResult:
    status: int
    content: bytes
    content_type: str = ""
    from_cache: bool = False

    @property
    def ok(self):
        return 200 <= self.status < 300

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content)


def _session():
    if not hasattr(_local, "session"):
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
        session.headers["User-Agent"] = USER_AGENT
        _local.session = session
    return _local.session


def _throttle(host):
    interval = _MIN_INTERVAL.get(host)
    if not interval:
        return
    with _host_guard:
        lock = _host_locks.setdefault(host, threading.Lock())
    with lock:
        wait = _host_last.get(host, 0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _host_last[host] = time.monotonic()


class _ResponseCache:
    """
    Thread-safe sqlite cache of successful GET responses.  The cache folder may be synced between machines
    (a cloud-synced folder), so a locked or damaged file disables the cache for this process instead of failing requests.
    """

    def __init__(self, path):
        self._lock = threading.Lock()
        self._conn = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
            conn.execute("CREATE TABLE IF NOT EXISTS responses "
                         "(key TEXT PRIMARY KEY, ts REAL, status INTEGER, ctype TEXT, body BLOB)")
            # responses that can no longer be used are removed once per process (the file would grow forever)
            keep_hours = max(7 * 24, 2 * config.HTTP_CACHE_TTL_HOURS)
            conn.execute("DELETE FROM responses WHERE ts < ?", (time.time() - keep_hours * 3600,))
            conn.commit()
            self._conn = conn
        except (sqlite3.Error, OSError) as e:
            self._disable(e)

    def _disable(self, error):
        # get/put return early once the connection is gone, so this prints once per process
        print(f"[MES] HTTP cache disabled ({type(error).__name__}: {error})")
        self._conn = None

    def get(self, key, ttl_hours):
        if self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute("SELECT ts, status, ctype, body FROM responses WHERE key = ?",
                                         (key,)).fetchone()
        except sqlite3.Error as e:
            self._disable(e)
            return None
        if row and time.time() - row[0] < ttl_hours * 3600:
            return HttpResult(row[1], row[3], row[2], from_cache=True)
        return None

    def put(self, key, result):
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                                   (key, time.time(), result.status, result.content_type, result.content))
                self._conn.commit()
        except sqlite3.Error as e:
            self._disable(e)


_cache = None
_cache_guard = threading.Lock()


def _get_cache():
    global _cache
    with _cache_guard:
        if _cache is None:
            _cache = _ResponseCache(config.CACHE_DIR / "http.sqlite")
    return _cache


def http_get(url, params=None, timeout=20, cache=True, ttl_hours=None, headers=None):
    """
    GET ``url`` and return an ``HttpResult``.  Successful responses are cached
    for ``ttl_hours`` (default ``MES_HTTP_CACHE_TTL_HOURS``).  Network errors raise
    ``requests.RequestException``; HTTP errors are returned with their status.
    """
    params = dict(params or {})
    ttl_hours = config.HTTP_CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
    key = hashlib.sha256((url + "?" + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()
    if cache and ttl_hours > 0:
        hit = _get_cache().get(key, ttl_hours)
        if hit is not None:
            return hit

    _throttle(urlparse(url).netloc)
    response = _session().get(url, params=params, timeout=timeout, headers=headers)
    result = HttpResult(response.status_code, response.content, response.headers.get("content-type", ""))
    if cache and ttl_hours > 0 and result.ok:
        _get_cache().put(key, result)
    return result


def get_json(url, params=None, **kwargs):
    """GET JSON; returns ``None`` on HTTP errors."""
    result = http_get(url, params, **kwargs)
    return result.json() if result.ok else None
