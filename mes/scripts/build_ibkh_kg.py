"""
Build the MES iBKH knowledge-graph store from an iBKH triplet text file.

Input (``--source``, default ``MES_IBKH_SOURCE``): a TSV with a ``full_text``
column (``whole_triplets_text.tsv``, ~35M lines "head relation tail", the relation
phrases of ``RELATIONS`` below) or a CSV with a ``text`` column in the same form.

Output (``--out``, default ``MES_IBKH_DIR``):
    ibkh.duckdb   nodes(id, name, norm, degree), rels(id, phrase, class, weight),
                  edges(h, r, t, n) and adj(node, nbr, r, dir, n) sorted by node
    node_emb.npy  L2-normalized embeddings of the node names (``MES_ENCODER``)
    meta.json     counts, encoder name, source file, build time

Usage:
    python -m mes.scripts.build_ibkh_kg [--source PATH] [--out DIR] [--no-embeddings]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root → `mes` importable

from mes import config  # noqa: E402

logger = logging.getLogger("build_ibkh_kg")

# Relation phrase → (evidence class, weight). Longer phrases first: the regex alternation is leftmost-first.
RELATIONS = [
    ("has an inferred relation with", "inferred", 0.15),
    ("has a semantic relation with", "literature-mined", 0.35),
    ("is a transporter for", "curated", 0.6),
    ("is an enzyme for", "curated", 0.6),
    ("is a carrier for", "curated", 0.6),
    ("is associated with", "curated", 0.6),
    ("interacts with", "curated", 0.5),
    ("covaries with", "curated", 0.4),
    ("downregulates", "curated", 0.6),
    ("upregulates", "curated", 0.6),
    ("presents with", "curated", 0.8),
    ("palliates", "curated", 0.9),
    ("resembles", "curated", 0.4),
    ("regulates", "curated", 0.5),
    ("binds to", "curated", 0.7),
    ("targets", "curated", 0.8),
    ("effects", "curated", 0.6),
    ("treats", "curated", 1.0),
    ("causes", "curated", 0.9),
    ("is a", "curated", 0.5),
]


def _source_relation(path):
    path = str(path)
    if path.endswith(".tsv"):
        return (f"read_csv('{path}', delim='\t', header=true, columns={{'full_text': 'VARCHAR'}}, "
                f"quote='', escape='')", "full_text")
    return f"read_csv('{path}', header=true, all_varchar=true)", "text"


def build(source, out_dir, embeddings=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "ibkh.duckdb"
    tmp_path = out_dir / "ibkh.duckdb.tmp"
    tmp_path.unlink(missing_ok=True)
    t0 = time.time()
    con = duckdb.connect(str(tmp_path))
    relation, column = _source_relation(source)
    alternation = "|".join(phrase for phrase, _, _ in RELATIONS)

    logger.info("Parsing %s", source)
    con.execute(f"""
        CREATE TEMP TABLE trip AS
        SELECT m.h AS h, m.r AS r, m.t AS t, count(*)::INTEGER AS n
        FROM (SELECT regexp_extract(trim({column}), '^(.+?) ({alternation}) (.+)$', ['h', 'r', 't']) AS m
              FROM {relation})
        WHERE m.r <> '' AND m.h <> m.t
        GROUP BY 1, 2, 3
    """)
    con.execute("CREATE TABLE rels (id UTINYINT, phrase VARCHAR, class VARCHAR, weight FLOAT)")
    con.executemany("INSERT INTO rels VALUES (?, ?, ?, ?)",
                    [(i, p, c, w) for i, (p, c, w) in enumerate(RELATIONS)])
    con.execute("""
        CREATE TEMP TABLE names AS
        SELECT (row_number() OVER (ORDER BY name) - 1)::INTEGER AS id, name
        FROM (SELECT h AS name FROM trip UNION SELECT t FROM trip)
    """)
    con.execute("""
        CREATE TABLE edges AS
        SELECT nh.id AS h, rels.id AS r, nt.id AS t, trip.n AS n
        FROM trip JOIN names nh ON trip.h = nh.name JOIN names nt ON trip.t = nt.name
        JOIN rels ON trip.r = rels.phrase
    """)
    con.execute("""
        CREATE TABLE adj AS
        SELECT * FROM (SELECT h AS node, t AS nbr, r, 0::UTINYINT AS dir, n FROM edges
                       UNION ALL SELECT t AS node, h AS nbr, r, 1::UTINYINT AS dir, n FROM edges)
        ORDER BY node
    """)
    con.execute("""
        CREATE TABLE nodes AS
        SELECT names.id, names.name,
               trim(regexp_replace(lower(names.name), '[^a-z0-9]+', ' ', 'g')) AS norm,
               coalesce(d.degree, 0)::INTEGER AS degree
        FROM names LEFT JOIN (SELECT node, count(*) AS degree FROM adj GROUP BY node) d ON d.node = names.id
        ORDER BY names.id
    """)
    counts = {
        "n_nodes": con.execute("SELECT count(*) FROM nodes").fetchone()[0],
        "n_edges": con.execute("SELECT count(*) FROM edges").fetchone()[0],
        "relations": dict(con.execute("""SELECT rels.phrase, count(*) FROM edges JOIN rels ON edges.r = rels.id
                                         GROUP BY 1 ORDER BY 2 DESC""").fetchall()),
    }
    logger.info("Graph: %d nodes, %d edges (%.0fs)", counts["n_nodes"], counts["n_edges"], time.time() - t0)
    names = [row[0] for row in con.execute("SELECT name FROM nodes ORDER BY id").fetchall()]
    con.execute("CHECKPOINT")
    con.close()
    tmp_path.replace(db_path)

    meta = {"source": str(source), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), **counts, "encoder": None}
    if embeddings:
        from mes.tools.ranking_tools import encode
        t1 = time.time()
        logger.info("Embedding %d node names with %s", len(names), config.ENCODER)
        vectors = encode(names, batch_size=512)
        np.save(out_dir / "node_emb.npy", vectors.astype(np.float16))
        meta["encoder"] = config.ENCODER
        logger.info("Node embeddings done (%.0fs)", time.time() - t1)
    meta["build_seconds"] = round(time.time() - t0, 1)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s (%.0f MB) in %.0fs", db_path, db_path.stat().st_size / 1e6, time.time() - t0)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=config.IBKH_SOURCE)
    parser.add_argument("--out", type=Path, default=config.IBKH_DIR)
    parser.add_argument("--no-embeddings", action="store_true", help="skip node-name embeddings")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.source.exists():
        sys.exit(f"Source file not found: {args.source} (set MES_IBKH_SOURCE or pass --source)")
    print(json.dumps(build(args.source, args.out, embeddings=not args.no_embeddings), indent=2))


if __name__ == "__main__":
    main()
