"""
Build the MES real-world-data store from the MIMIC-IV CSV release (one-time; resumable).

    python -m mes.scripts.build_mimic_db                  # all steps
    python -m mes.scripts.build_mimic_db --steps mes,dictionary,catalog  # rebuild the fast steps only

Steps (each writes Parquet files with relative paths only, so the folder works on every machine that syncs it):
    raw         hosp/*.csv.gz, icu/*.csv.gz → <MIMIC_DIR>/hosp|icu/<table>.parquet (zstd; column types from the
                mimic-code schema, so ICD codes such as "0389" keep their leading zero)
    concepts    the mimic-code DuckDB concepts (SOFA, Sepsis-3, KDIGO, ventilation, vasoactive agents, Charlson,
                first-day measurements, ...) → <MIMIC_DIR>/derived/<concept>.parquet
    mes         MES analysis layer (cohorts with outcomes, diagnoses, drug and procedure exposures)
                → <MIMIC_DIR>/mes/<table>.parquet
    dictionary  concept dictionary with patient counts + embeddings → <MIMIC_DIR>/dictionary.parquet, .npy
    catalog     row counts, coverage, validation → <MIMIC_DIR>/catalog.json

Every file is written to ``<name>.tmp`` and renamed when complete (a synced folder never sees half a file). A table whose
Parquet file exists is skipped; ``--force`` rebuilds. The temporary DuckDB used while building lives outside the
synced folder and is deleted at the end.
"""

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import duckdb

from mes import config

logger = logging.getLogger("build_mimic_db")
VENDOR = Path(__file__).resolve().parents[1] / "tools" / "rwd" / "vendor" / "mimic_code"
SCHEMAS = {"hosp": "mimiciv_hosp", "icu": "mimiciv_icu"}
# Row counts published for MIMIC-IV v2.2 (CHANGELOG); the build fails if the core tables differ
EXPECTED_V22 = {"patients": 299712, "admissions": 431231, "icustays": 73181}
STEPS = ("raw", "concepts", "mes", "dictionary", "catalog")


# ============================================================================ helpers
def _say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _sql_path(path):
    return str(path).replace("'", "''")


def _write_parquet(con, select_sql, target, force=False):
    """COPY (select) to ``target`` atomically; returns the row count (skips existing files unless forced)."""
    target = Path(target)
    if target.exists() and not force:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    tmp.unlink(missing_ok=True)
    con.execute(f"COPY ({select_sql}) TO '{_sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880)")
    os.replace(tmp, target)
    return con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(target)}')").fetchone()[0]


def schema_types():
    """{table: [(column, duckdb type)]} from the mimic-code create.sql."""
    sql = (VENDOR / "create.sql").read_text()
    tables = {}
    for m in re.finditer(r"CREATE TABLE mimiciv_\w+\.(\w+)\s*\((.*?)\);", sql, re.S):
        cols = []
        for line in m.group(2).splitlines():
            line = line.strip().rstrip(",")
            if not line or line.startswith("--"):
                continue
            name, kind = line.split()[0], " ".join(line.split()[1:]).upper()
            kind = kind.replace("NOT NULL", "").strip()
            if kind.startswith(("VARCHAR", "CHAR", "TEXT")):
                kind = "VARCHAR"
            elif kind.startswith("TIMESTAMP"):
                kind = "TIMESTAMP"
            elif kind.startswith(("DOUBLE", "NUMERIC", "DECIMAL", "REAL")):
                kind = "DOUBLE"
            elif kind.startswith("FLOAT"):
                kind = "FLOAT"
            cols.append((name, kind))
        tables[m.group(1)] = cols
    return tables


def connect(db_path=None):
    con = duckdb.connect(str(db_path) if db_path else ":memory:")
    tmp = Path(tempfile.gettempdir()) / "mes_mimic_spill"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = '{_sql_path(tmp)}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{os.environ.get('MES_BUILD_MEMORY', '10GB')}'")
    con.execute(f"SET threads = {min(os.cpu_count() or 4, 8)}")
    return con


def raw_views(con, out):
    """Schemas mimiciv_hosp / mimiciv_icu as views over the raw Parquet files (absolute paths: build DB only)."""
    for folder, schema in SCHEMAS.items():
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for f in sorted((out / folder).glob("*.parquet")):
            con.execute(f"CREATE OR REPLACE VIEW {schema}.{f.stem} AS SELECT * FROM read_parquet('{_sql_path(f)}')")


# ============================================================================ step 1: raw tables
def step_raw(source, out, force):
    types = schema_types()
    files = sorted(source.glob("hosp/*.csv.gz")) + sorted(source.glob("icu/*.csv.gz"))
    if not files:
        raise SystemExit(f"No MIMIC-IV csv.gz files under {source} (expected hosp/ and icu/)")
    # Largest first, a few at a time: gzip decompression is single-threaded per file
    files.sort(key=lambda f: f.stat().st_size, reverse=True)
    con = connect()
    counts = {}

    def convert(f):
        table, folder = f.name[:-7], f.parent.name
        target = out / folder / f"{table}.parquet"
        if target.exists() and not force:
            return table, "kept"
        cols = types.get(table)
        if not cols:
            return table, "not in the mimic-code schema, skipped"
        spec = "{" + ", ".join(f"'{c}': '{t}'" for c, t in cols) + "}"
        t = time.time()
        cur = con.cursor()
        n = _write_parquet(cur, f"SELECT * FROM read_csv('{_sql_path(f)}', header=true, delim=',', quote='\"', "
                                f"escape='\"', columns={spec})", target, force=True)
        _say(f"raw {folder}/{table}: {n:,} rows in {time.time() - t:.0f}s")
        return table, n

    with ThreadPoolExecutor(max_workers=4) as pool:
        for table, n in pool.map(convert, files):
            counts[table] = n
    return counts


# ============================================================================ step 2: mimic-code concepts
def concept_order():
    """Concept SQL files in the order of mimic-code's duckdb.sql."""
    lines = (VENDOR / "concepts_duckdb" / "duckdb.sql").read_text().splitlines()
    return [line.split(None, 1)[1].strip() for line in lines if line.startswith(".read")]


def step_concepts(out, force, build_db):
    con = connect(build_db)
    raw_views(con, out)
    con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived")
    status = {}
    for rel in concept_order():
        name = Path(rel).stem
        target = out / "derived" / f"{name}.parquet"
        if target.exists() and not force:
            con.execute(f"CREATE OR REPLACE VIEW mimiciv_derived.{name} AS SELECT * FROM read_parquet('{_sql_path(target)}')")
            status[name] = "kept"
            continue
        sql = (VENDOR / "concepts_duckdb" / rel).read_text()
        t = time.time()
        try:
            con.execute(f"DROP VIEW IF EXISTS mimiciv_derived.{name}")
            con.execute(sql)
            n = _write_parquet(con, f"SELECT * FROM mimiciv_derived.{name}", target, force=True)
            # later concepts read the exported file, keeping the build database small
            con.execute(f"DROP TABLE IF EXISTS mimiciv_derived.{name}")
            con.execute(f"CREATE VIEW mimiciv_derived.{name} AS SELECT * FROM read_parquet('{_sql_path(target)}')")
            status[name] = n
            _say(f"concept {name}: {n:,} rows in {time.time() - t:.0f}s")
        except Exception as e:
            status[name] = f"failed: {type(e).__name__}: {str(e)[:300]}"
            _say(f"concept {name} FAILED: {str(e)[:300]}")
    con.close()
    return status


# ============================================================================ main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=config.MIMIC_SOURCE)
    parser.add_argument("--out", type=Path, default=config.MIMIC_DIR)
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {', '.join(STEPS)}")
    parser.add_argument("--force", action="store_true", help="rebuild files that already exist")
    args = parser.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    build_db = Path(tempfile.gettempdir()) / "mes_mimic_build.duckdb"
    log = out / "build_log.json"
    record = json.loads(log.read_text()) if log.exists() else {}
    t0 = time.time()
    try:
        if "raw" in steps:
            _say(f"raw tables from {args.source}")
            record["raw"] = step_raw(args.source.expanduser(), out, args.force)
        if "concepts" in steps:
            _say("mimic-code concepts")
            record["concepts"] = step_concepts(out, args.force, build_db)
        if "mes" in steps:
            from mes.tools.rwd import build_layer
            _say("MES analysis layer")
            record["mes"] = build_layer.build_mes_layer(out, args.force)
        if "dictionary" in steps:
            from mes.tools.rwd import build_layer
            _say("concept dictionary")
            record["dictionary"] = build_layer.build_dictionary(out, args.force)
        if "catalog" in steps:
            from mes.tools.rwd import build_layer
            _say("catalog and validation")
            record["catalog"] = build_layer.build_catalog(out, EXPECTED_V22)
    finally:
        record["last_run"] = {"steps": steps, "seconds": round(time.time() - t0), "finished": datetime.now().isoformat()}
        log.write_text(json.dumps(record, indent=1, default=str))
        for f in (build_db, build_db.with_suffix(".duckdb.wal")):
            f.unlink(missing_ok=True)
        shutil.rmtree(Path(tempfile.gettempdir()) / "mes_mimic_spill", ignore_errors=True)
    _say(f"done in {(time.time() - t0) / 60:.1f} min → {out}")


if __name__ == "__main__":
    main()
