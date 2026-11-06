# Data and external resources

MES needs no local data to answer from the literature: PubMed, Europe PMC, Unpaywall and ClinicalTrials.gov are
queried over the network. Two sources need a local store: the iBKH knowledge graph and real-world data (MIMIC-IV).
When a store is missing, its source is shown as unavailable and switched off. Everything below lives under `data/`,
the default; each path can be changed in `.env`. `data/`, `.cache/` and `.env` are git-ignored.

| Resource | Used by | Default location | How to get it | May it be shared? |
|---|---|---|---|---|
| Sentence encoder `sentence-transformers/all-MiniLM-L6-v2` | ranking, KG linking, concept search | Hugging Face cache | downloaded on first use (~90 MB) | yes (Apache-2.0) |
| iBKH store | the iBKH graph source | `data/ibkh/` (`MES_IBKH_DIR`) | build it (below) | see the iBKH notes below |
| MIMIC-IV v2.2 release | building the RWD store | `data/mimic-iv-2.2/` (`MES_MIMIC_SOURCE`) | PhysioNet, credentialed access | **no** |
| RWD store built from it | the real-world data source | `data/mimic/` (`MES_MIMIC_DIR`) | `python -m mes.scripts.build_mimic_db` | **no** (derived from MIMIC-IV) |
| Your PDFs | the PDF library | `data/pdfs/` (`MES_PDF_DIR`) | copy PDFs there; indexed in the background into `data/pdf_index/` | as their licences allow |
| Caches and saved conversations | answer / HTTP / full-text caches, history, saved teams | `.cache/` (`MES_CACHE_DIR`) | written by MES | **no** when real-world data was used (answers hold its aggregates) |

## iBKH knowledge graph

The iBKH store is not included in this repository. Build it from the iBKH triples:

```bash
python -m mes.scripts.build_ibkh_kg --source <triples file>      # writes the store to data/ibkh/ (MES_IBKH_DIR)
```

The source file holds one triple per line as text, `<head> <relation phrase> <tail>`, with the relation phrases
listed in `RELATIONS` of the script. It can be a TSV with a `full_text` column or a CSV with a `text` column. iBKH
itself is published by its authors at [github.com/wcm-wanglab/iBKH](https://github.com/wcm-wanglab/iBKH). Without the
store MES still runs, with the knowledge-graph source switched off.

**Terms.** iBKH integrates relations from many biomedical sources, such as curated drug and gene databases,
SemMedDB (literature-mined) and CTD (inferred). Those sources keep their own licences and terms of use, and these
apply to the store, which is derived from them. Check them before redistributing the store or using it outside
research.

## MIMIC-IV (real-world data)

MIMIC-IV is a credentialed dataset: access requires a PhysioNet account, the required training and the signed data
use agreement. MES does not distribute MIMIC-IV or anything derived from it.

1. Download the MIMIC-IV **v2.2** `hosp/` and `icu/` modules (the `.csv.gz` files) from PhysioNet into
   `data/mimic-iv-2.2/hosp` and `data/mimic-iv-2.2/icu`.
2. Build the store:

   ```bash
   python -m mes.scripts.build_mimic_db            # all steps; --steps raw,concepts,mes,dictionary,catalog
   ```

   It converts the tables to Parquet with the mimic-code schema, runs the mimic-code DuckDB concepts (vendored in
   `mes/tools/rwd/vendor/mimic_code/`, MIT), builds the MES analysis layer, a concept dictionary with patient counts
   and the data catalog. It checks the row counts of `patients`, `admissions` and `icustays` against the v2.2 release
   and records the checks in `data/mimic/catalog.json` (`checks`, `ok`). The pipeline is written for v2.2. The output is about 6 GB in `data/mimic/` and takes a few minutes.
3. Start MES and switch on Real-world data ▾ → MIMIC-IV.

**A store built on another computer** can be used as is, without building again: copy the `mimic/` folder itself
(e.g. `rsync -a`, or `cp -RL` when the source is a symbolic link) to `data/mimic/` — MES needs `data/mimic/catalog.json`
and `data/mimic/mes/icu_stays.parquet`. Copying only a symbolic link, or only the CSV release, is not enough. Both
computers must be covered by your data use agreement. When `.env` sets `MES_DATA_DIR` or `MES_MIMIC_DIR`, MES looks there
instead of `data/`; `./run.sh` prints the folder it looked in.

**What reaches the language model.** By design, only the database schema, dictionary entries (concept names and
codes) and aggregate results reach the LLM, with counts under 10 suppressed. Patient-level rows stay in the local
DuckDB process. The RWD analyst agent's code runs in a separate sandboxed process: on macOS with `sandbox-exec`, with
no network and no writes outside its temporary folder; on other systems only the Python-level code policy applies.
Its output is checked to be aggregate before it is used.

**Your obligations.** With a cloud LLM API, check that your setup satisfies the PhysioNet data use agreement
and PhysioNet's guidance on using credentialed data with online services. With a local model (LM Studio), nothing
leaves the computer except the literature and registry queries.

**Never commit or share** `data/mimic-iv-2.2/`, `data/mimic/` (it holds row-level tables, and the concept
dictionary has small counts), or the `.cache/` of a computer that answered real-world data questions. All three
are git-ignored. Results computed on MIMIC-IV, such as screenshots and exported answers, follow the data use
agreement's rules for publishing aggregate results.

## Without the local stores

| Present | What works |
|---|---|
| nothing (network only) | Literature (PubMed, Europe PMC full text, Unpaywall) and ClinicalTrials.gov; the iBKH and real-world data sources show as unavailable |
| + iBKH store | + the knowledge-graph source and its network view |
| + MIMIC-IV store | + real-world data studies, the cohort builder, patient explorer and trial comparison |
| + `MES_RWD_ANALYST=on` | + the RWD analyst agent, which writes its own analysis code for questions the study templates cannot express |
