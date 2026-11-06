# Vendored from MIT-LCP/mimic-code

- Repository: https://github.com/MIT-LCP/mimic-code (MIT License, see LICENSE)
- Commit: 303d26c623dcc9c49cc0f204468d4acc2f063797 (Tue Sep 1 09:06:25 2026 -0400)
- `create.sql`: `mimic-iv/buildmimic/postgres/create.sql` — column types of every MIMIC-IV table
  (the DuckDB build in mimic-code uses the same file).
- `concepts_duckdb/`: `mimic-iv/concepts_duckdb/` — community-validated derived concepts (SOFA, Sepsis-3,
  KDIGO, ventilation, vasoactive agents, Charlson, first-day measurements, …) transpiled to DuckDB by mimic-code.
  `duckdb.sql` gives the build order; `scripts/build_mimic_db.py` runs the files in that order.

Do not edit these files; update them by copying a newer mimic-code commit.
