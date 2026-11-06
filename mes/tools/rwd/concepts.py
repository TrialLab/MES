"""
Concept search for RWD study design (deterministic): finds the MIMIC-IV concepts that match each element of a
question, with patient counts, so the study designer can only choose concepts that exist in the data.

Concept ids:
    phe:<id>        curated phenotype (ICD-9 + ICD-10 code set)          library.PHENOTYPES
    drugclass:<id>  curated drug class (ingredient names)                library.DRUG_CLASSES
    drug:<text>     medications whose name contains <text>
    var:<name>      column of the analysis cohort (lab, score, flag…)    library.VARIABLES
    dx9:/dx10:<code>, px9:/px10:<code>   single ICD diagnosis / procedure code (and its sub-codes)
    lab:<itemid>    laboratory test (labevents),  item:<itemid>  ICU chart / procedure item
"""

import re
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mes.tools.rwd import library as L
from mes.tools.rwd.db import get_store

KIND_LABEL = {"phe": "condition (curated codes)", "drugclass": "drug class", "drug": "medication", "var": "variable",
              "dx9": "ICD-9 diagnosis", "dx10": "ICD-10 diagnosis", "px9": "ICD-9 procedure",
              "px10": "ICD-10 procedure", "lab": "lab test", "item": "ICU item", "micro": "microbiology",
              "derived": "treatment duration"}
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Candidate:
    concept_id: str
    label: str
    kind: str
    n_patients: int | None = None
    score: float = 0.0
    detail: dict = field(default_factory=dict)

    def describe(self):
        n = f", {self.n_patients:,} patients" if self.n_patients is not None else ""
        return f"{self.concept_id} — {self.label} [{KIND_LABEL.get(self.kind, self.kind)}{n}]"


def _tokens(text):
    return set(_WORD.findall(str(text).lower())) - {"of", "the", "and", "in", "with", "or", "a", "an", "to", "for"}


def _spelled_like(word, words):
    """``word`` is one of ``words``, starts like one (5 letters) or differs from one by a typo."""
    from difflib import SequenceMatcher
    return any(word == w or (len(w) >= 5 and word[:5] == w[:5]) or SequenceMatcher(None, word, w).ratio() >= 0.8
               for w in words)


def name_matches(name, patterns, exclude=()):
    """A medication name matches an ingredient when the ingredient starts a word ("statin" ≠ "nystatin")."""
    return (any(re.search(r"(?<![a-z])" + re.escape(p.lower()), name) for p in patterns)
            and not any(x.lower() in name for x in exclude))


def code_predicate(icd10, icd9, column="icd_code"):
    """SQL predicate for ICD code sets (prefix match; '_' is a single-character wildcard)."""
    parts = [f"(icd_version = 10 AND {column} LIKE '{c}%')" for c in icd10]
    parts += [f"(icd_version = 9 AND {column} LIKE '{c}%')" for c in icd9]
    return "(" + (" OR ".join(parts) or "FALSE") + ")"


# first words of medication names that are no ingredient (containers, diluents, pseudo-orders of MIMIC-IV)
NOT_INGREDIENTS = {"admission", "intake", "solution", "syringe", "vial", "sterile", "flush", "bolus", "pacu", "water",
                   "diluent", "infusion", "injection", "premix", "drip", "pump", "lock", "order", "unit", "patient",
                   "osmotic"}                  # ("iso-osmotic dextrose" is dextrose)


class ConceptIndex:
    def __init__(self, store=None):
        self.store = store or get_store()
        self._lock = threading.Lock()
        self._loaded = False           # tables and embeddings (search)
        self._tables = False           # tables only (labels, drug names, counts): no sentence encoder needed
        self._counts = {}

    # ------------------------------------------------------------------ loading
    def _load_tables(self):
        with self._lock:
            if self._tables:
                return
            root = self.store.root
            self.dictionary = pd.read_parquet(root / "dictionary.parquet")
            self.drug_names = pd.read_parquet(root / "mes" / "drug_names.parquet")
            rows = []
            for k, (label, _i10, _i9, syn) in L.PHENOTYPES.items():
                rows.append(("phe", f"phe:{k}", label, syn))
            for k, (label, _pats, _exc, syn) in L.DRUG_CLASSES.items():
                rows.append(("drugclass", f"drugclass:{k}", label, syn))
            for k, (label, *_rest) in L.VARIABLES.items():
                rows.append(("var", f"var:{k}", label, [k.replace("_", " ")] + L.VARIABLE_SYNONYMS.get(k, [])))
            for k, (label, _specimens, syn) in L.MICRO.items():
                rows.append(("micro", f"micro:{k}", label, syn))
            for k, (label, _unit, _sql, syn) in L.DERIVED.items():
                rows.append(("derived", f"derived:{k}", label, syn))
            self.library = pd.DataFrame(rows, columns=["kind", "concept_id", "label", "synonyms"])
            self._tables = True

    def _load(self):
        self._load_tables()
        with self._lock:
            if self._loaded:
                return
            self.vectors = np.load(self.store.root / "dictionary_vectors.npy").astype(np.float32)
            from mes.tools.ranking_tools import encode
            texts = [f"{r.label}; {', '.join(r.synonyms)}" for r in self.library.itertuples()]
            self.library_vectors = encode(texts)
            self._loaded = True

    def warm(self):
        self._load()

    # ------------------------------------------------------------------ resolution
    def label(self, concept_id):
        self._load_tables()
        kind, _, key = concept_id.partition(":")
        if kind == "phe" and key in L.PHENOTYPES:
            return L.PHENOTYPES[key][0]
        if kind == "drugclass" and key in L.DRUG_CLASSES:
            return L.DRUG_CLASSES[key][0]
        if kind == "var" and key in L.VARIABLES:
            return L.VARIABLES[key][0]
        if kind == "micro" and key in L.MICRO:
            return L.MICRO[key][0]
        if kind == "derived" and key in L.DERIVED:
            return L.DERIVED[key][0]
        if kind == "drug":
            return f"Medication containing “{key}”"
        hit = self.dictionary[self.dictionary.concept_id == concept_id]
        return hit.label.iloc[0] if len(hit) else concept_id

    def levels(self, base, variable):
        """Values of a category variable in the population (cached), e.g. careunit → ["CCU", "CVICU", …]."""
        key = ("levels", base, variable)
        if key not in self._counts:
            b = L.BASES[base]
            try:
                df = self.store.query(f"SELECT DISTINCT {variable} AS v FROM {b['table']} WHERE {b['where']} "
                                      f"AND {variable} IS NOT NULL LIMIT 100")
                self._counts[key] = sorted(str(v) for v in df.v)
            except Exception:
                self._counts[key] = []
        return self._counts[key]

    def is_known(self, concept_id):
        self._load_tables()
        kind, _, key = concept_id.partition(":")
        if kind == "phe":
            return key in L.PHENOTYPES
        if kind == "drugclass":
            return key in L.DRUG_CLASSES
        if kind == "var":
            return key in L.VARIABLES
        if kind == "micro":
            return key in L.MICRO
        if kind == "derived":
            return key in L.DERIVED
        if kind == "drug":
            return bool(key) and bool(self.drug_names_for(concept_id))
        return bool((self.dictionary.concept_id == concept_id).any())

    def drug_names_for(self, concept_id):
        """Medication names matched by a drugclass:/drug: concept."""
        self._load_tables()
        kind, _, key = concept_id.partition(":")
        if kind == "drugclass":
            _label, pats, exc, _syn = L.DRUG_CLASSES[key]
        else:
            pats, exc = [key], []
        names = self.drug_names.drug
        return sorted(names[names.apply(lambda d: name_matches(d, pats, exc))].tolist())

    def count(self, concept_id):
        """Patients with the concept anywhere in MIMIC-IV (cached)."""
        if concept_id in self._counts:
            return self._counts[concept_id]
        self._load_tables()
        kind, _, key = concept_id.partition(":")
        n = None
        try:
            if kind == "phe":
                _l, i10, i9, _s = L.PHENOTYPES[key]
                n = self.store.query(f"SELECT count(DISTINCT subject_id) n FROM mes.diagnoses WHERE "
                                     f"{code_predicate(i10, i9)}").n[0]
            elif kind == "micro" and key in L.MICRO:
                n = self.store.query(f"SELECT count(DISTINCT subject_id) n FROM mimiciv_hosp.microbiologyevents m "
                                     f"WHERE {L.micro_predicate(key)}").n[0]
            elif kind in ("drugclass", "drug"):
                names = self.drug_names_for(concept_id)
                if names:
                    lst = ", ".join("'" + x.replace("'", "''") + "'" for x in names)
                    n = self.store.query(f"SELECT count(DISTINCT subject_id) n FROM mes.drugs WHERE drug IN ({lst})").n[0]
                else:
                    n = 0
            elif kind == "var":
                n = None
            else:
                hit = self.dictionary[self.dictionary.concept_id == concept_id]
                n = int(hit.n_patients.iloc[0]) if len(hit) else 0
        except Exception:
            n = None
        self._counts[concept_id] = None if n is None else int(n)
        return self._counts[concept_id]

    # ------------------------------------------------------------------ search
    def search(self, text, k=8, kinds=None, query_vector=None):
        """Best concepts for ``text``: curated library first, then the MIMIC dictionary."""
        self._load()
        from mes.tools.ranking_tools import encode
        qv = query_vector if query_vector is not None else encode([text], is_query=True)[0]
        words = _tokens(text)
        out = []
        lib_scores = self.library_vectors @ qv
        for i, row in enumerate(self.library.itertuples()):
            if kinds and row.kind not in kinds:
                continue
            overlap = len(words & _tokens(row.label + " " + " ".join(row.synonyms))) / max(1, len(words))
            exact = any(s.lower() == text.lower().strip() for s in [row.label] + list(row.synonyms))
            # curated concepts are preferred over single codes of the dictionary
            out.append(Candidate(row.concept_id, row.label, row.kind,
                                 score=float(lib_scores[i]) + 0.35 * overlap + 0.3 * exact + 0.2))
        dict_scores = self.vectors @ qv
        ingredients = {}
        for i in np.argsort(-dict_scores)[:300]:
            row = self.dictionary.iloc[int(i)]
            kind = row.concept_id.split(":", 1)[0]
            if kind == "drug":
                # a medication name stands for its ingredient (first word), e.g. "metformin xr (glucophage xr)" — not
                # a word of a container or a pseudo-order ("pre-admission/non-icu intake" once matched "medical ICU
                # admission")
                word = next((w for w in _WORD.findall(row.label) if len(w) >= 4 and not w.isdigit()
                             and w not in NOT_INGREDIENTS), None)
                # medication names look alike to the encoder ("unobtainium" ~ "vecuronium"): spelling must agree
                if word and word not in ingredients and dict_scores[i] >= 0.6 and _spelled_like(word, words):
                    ingredients[word] = float(dict_scores[i])
                continue
            if kinds and kind not in kinds:
                continue
            overlap = len(words & _tokens(row.label)) / max(1, len(words))
            prior = 0.04 * np.log10(1 + row.n_patients)
            out.append(Candidate(row.concept_id, row.label, kind, int(row.n_patients),
                                 score=float(dict_scores[i]) + 0.3 * overlap + prior))
        if not kinds or "drug" in kinds:
            term = text.lower().strip()
            if len(term) >= 4 and re.fullmatch(r"[a-z][a-z\- ]+", term):
                ingredients[term] = max(ingredients.get(term, 0.0), 0.9)
            for word, sim in list(ingredients.items())[:4]:
                out.append(Candidate(f"drug:{word}", f"Medication containing “{word}”", "drug",
                                     score=sim + (0.25 if word in words else 0.0)))
        best = {}
        for c in sorted(out, key=lambda c: c.score, reverse=True):
            best.setdefault(c.concept_id, c)
        result = []
        for c in best.values():
            if c.kind in ("phe", "drugclass", "drug", "micro"):
                c.n_patients = self.count(c.concept_id)
                if not c.n_patients:
                    continue
            result.append(c)
            if len(result) == k:
                break
        return result

    def candidates_for(self, plan, question, per_element=6, limit=40):
        """Candidate concepts for every PICO element of ``plan`` plus the question text, grouped by element."""
        groups = {}
        for entity in plan.entities or []:
            terms = [entity.get("name", "")] + list(entity.get("synonyms") or [])[:2]
            found = {}
            for term in terms:
                if not term:
                    continue
                for c in self.search(term, k=per_element):
                    if c.concept_id not in found or c.score > found[c.concept_id].score:
                        found[c.concept_id] = c
            label = f"{entity.get('role', 'other')}: {entity.get('name', '')}"
            groups[label] = sorted(found.values(), key=lambda c: c.score, reverse=True)[:per_element]
        groups["question"] = self.search(question, k=per_element)
        seen, total = set(), 0
        for key in list(groups):
            kept = []
            for c in groups[key]:
                if c.concept_id in seen or total >= limit:
                    continue
                seen.add(c.concept_id)
                kept.append(c)
                total += 1
            groups[key] = kept
        return groups


_index = None
_index_lock = threading.Lock()


def get_index():
    global _index
    with _index_lock:
        if _index is None:
            _index = ConceptIndex()
    return _index
