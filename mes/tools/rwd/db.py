"""
Read-only access to the MES real-world-data store (MIMIC-IV as Parquet, built by scripts/build_mimic_db.py).

The views are created when the connection opens, from paths resolved on this machine, so the store folder holds
no machine-specific paths (it can be moved or copied). Schemas: ``mimiciv_hosp`` and ``mimiciv_icu`` (raw tables), ``mimiciv_derived``
(mimic-code concepts) and ``mes`` (MES analysis layer). After the views exist, file access is limited to the
store folder and the configuration is locked, so queries cannot read or write anything else.
"""

import json
import os
import threading
from pathlib import Path

import duckdb

from mes import config

SCHEMA_DIRS = {"mimiciv_hosp": "hosp", "mimiciv_icu": "icu", "mimiciv_derived": "derived", "mes": "mes"}


def _quote(path):
    return str(path).replace("'", "''")


def default_memory():
    """DuckDB memory limit: about a fifth of the machine's memory (1–6 GB), leaving room for the LLM server."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        total = 16
    return f"{max(1, min(6, int(total * 0.2)))}GB"


class Store:
    def __init__(self, root=None):
        self.root = Path(root or config.MIMIC_DIR).expanduser()
        self._con = None
        self._lock = threading.Lock()
        self._columns = {}

    @property
    def available(self):
        return (self.root / "catalog.json").exists() and (self.root / "mes" / "icu_stays.parquet").exists()

    def catalog(self):
        path = self.root / "catalog.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def tables(self, schema):
        return sorted(f.stem for f in (self.root / SCHEMA_DIRS[schema]).glob("*.parquet"))

    def _open(self, lock=True):
        con = duckdb.connect(":memory:")
        con.execute(f"SET threads = {int(os.environ.get('MES_RWD_THREADS', 8))}")
        con.execute(f"SET memory_limit = '{os.environ.get('MES_RWD_MEMORY') or default_memory()}'")
        for schema, folder in SCHEMA_DIRS.items():
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            for f in sorted((self.root / folder).glob("*.parquet")):
                con.execute(f"CREATE VIEW {schema}.{f.stem} AS SELECT * FROM read_parquet('{_quote(f)}')")
        if lock:
            con.execute(f"SET allowed_directories = ['{_quote(self.root.resolve())}/']")
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
        return con

    def cursor(self):
        """A cursor on the shared read-only connection (one per thread / query)."""
        with self._lock:
            if self._con is None:
                self._con = self._open()
        return self._con.cursor()

    def columns(self, table):
        """The column names of a table or view, e.g. ``mes.icu_stays`` (cached)."""
        if table not in self._columns:
            try:
                self._columns[table] = set(self.query(f"SELECT * FROM {table} LIMIT 0").columns)
            except Exception:
                return set()
        return self._columns[table]

    def query(self, sql, params=None, timeout=60):
        """Run ``sql`` and return a DataFrame; the query is interrupted after ``timeout`` seconds."""
        cur = self.cursor()
        timer = threading.Timer(timeout, cur.interrupt)
        timer.start()
        try:
            return cur.execute(sql, params or []).df()
        except duckdb.InterruptException as e:
            raise TimeoutError(f"query stopped after {timeout} s") from e
        finally:
            timer.cancel()
            cur.close()


_store = None
_store_lock = threading.Lock()


def get_store():
    global _store
    with _store_lock:
        if _store is None:
            _store = Store()
    return _store
