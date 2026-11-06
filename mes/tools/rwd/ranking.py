"""
The commonest items of one kind in a population — the medications given, the diagnoses coded, the procedures done
("which drugs are most often given in the first 24 h?", "the commonest diagnoses of ICU patients who die"): the share
of patients with each item, for everyone or for two groups (by an outcome, an exposure or a characteristic, as a
characterization compares them). Medications are counted by ingredient (the first word of the name; containers and
pseudo-orders aside), within a drug class when one is named; diagnoses by ICD code with its title; procedures by name.
Aggregates only; items of fewer than 10 patients are not shown.

    run_top(spec, cohort, mode) -> result  (tables.ranking; the card draws it as a dot chart)
"""

import re

import numpy as np
import pandas as pd

from mes.tools.rwd import library as L
from mes.tools.rwd import stats as S

SMALL, TOP = 10, 25
RANKED = {"drugs": "medications", "diagnoses": "diagnoses", "procedures": "procedures"}
_WORD = re.compile(r"[a-z][a-z0-9]{3,}")


# abbreviations of common fluids and electrolytes, and the salts whose second word says which one it is
ALIASES = {"nacl": "sodium chloride", "ns": "sodium chloride", "kcl": "potassium chloride", "d5w": "dextrose",
           "lr": "lactated ringers", "mgso4": "magnesium sulfate", "nahco3": "sodium bicarbonate"}
SALTS = ("sodium", "potassium", "calcium", "magnesium")


def ingredient(name):
    """The ingredient a medication name stands for — its first word of 4+ letters that is no container, with the
    salt for electrolytes ("sodium chloride", "potassium phosphate") and common abbreviations spelled out — or None."""
    from mes.tools.rwd.concepts import NOT_INGREDIENTS
    text = str(name or "").lower()
    alias = next((ALIASES[w] for w in re.findall(r"[a-z0-9]+", text) if w in ALIASES), None)
    if alias:
        return alias
    words = [w for w in _WORD.findall(text) if w not in NOT_INGREDIENTS]
    if not words:
        return None
    if words[0] in SALTS and len(words) > 1 and not words[1].isdigit():
        return f"{words[0]} {words[1]}"
    return words[0]


def _end(spec):
    """SQL of the end of the window: the horizon after time zero, else the end of the stay."""
    if float(spec.horizon_h or 0) > 0:
        return f"u.t0 + INTERVAL ({float(spec.horizon_h)!r}) HOUR"
    return "s.outtime" if spec.base == "icu" else "u.dischtime"


def pairs(store, cohort, spec, index=None):
    """DataFrame [unit_id, item, code]: each patient's distinct items of the kind ranked."""
    units = cohort.df[["unit_id", "hadm_id", "t0", "dischtime"]].drop_duplicates("unit_id")
    stay = "JOIN mes.icu_stays s ON s.stay_id = u.unit_id" if spec.base == "icu" else ""
    end = _end(spec)
    if spec.ranked == "drugs":
        local = any(c.concept.startswith("drugclass:") and L.systemic_only(c.concept.split(":", 1)[1])
                    for c in spec.treatments or [] if c.concept.split(":", 1)[1] in L.DRUG_CLASSES)
        route = (" AND lower(coalesce(d.route, '')) NOT IN (" + ", ".join(f"'{r}'" for r in L.LOCAL_ROUTES) + ")"
                 if local else "")             # a systemic class: no creams, drops or inhalations (as its criterion)
        sql = (f"SELECT DISTINCT u.unit_id, d.drug AS name FROM mes_rank_units u {stay} JOIN mes.drugs d ON "
               f"d.hadm_id = u.hadm_id WHERE d.starttime >= u.t0 AND d.starttime < {end}{route}")
    elif spec.ranked == "procedures":
        sql = (f"SELECT DISTINCT u.unit_id, coalesce(p.label, p.code) AS name FROM mes_rank_units u {stay} JOIN "
               f"mes.procedures p ON p.hadm_id = u.hadm_id WHERE p.time >= date_trunc('day', u.t0) AND p.time < {end}")
    else:                                   # by title: an ICD-9 and an ICD-10 code of the same title are one item
        sql = ("SELECT DISTINCT u.unit_id, CAST(d.icd_version AS INTEGER) AS version, d.icd_code AS code{title} FROM "
               "mes_rank_units u JOIN mes.diagnoses d ON d.hadm_id = u.hadm_id{join}")
        plain = sql.format(title="", join="")
        sql = sql.format(title=", t.long_title AS title", join=" LEFT JOIN mimiciv_hosp.d_icd_diagnoses t ON "
                         "CAST(t.icd_version AS INTEGER) = CAST(d.icd_version AS INTEGER) AND t.icd_code = d.icd_code")
    cur = store.cursor()
    try:
        cur.register("mes_rank_units", units)
        try:
            df = cur.execute(sql).df()
        except Exception:
            if spec.ranked != "diagnoses":
                raise
            df = cur.execute(plain).df()             # (no dictionary of titles: the codes themselves)
    finally:
        try:
            cur.unregister("mes_rank_units")
        except Exception:
            pass
        cur.close()
    if spec.ranked == "drugs":
        allowed = None
        if spec.treatments and index is not None:          # "the commonest antibiotics": within the class
            allowed = {n for c in spec.treatments for n in index.drug_names_for(c.concept)}
        if allowed is not None:
            df = df[df["name"].isin(allowed)]
        names = {n: ingredient(n) for n in df["name"].unique()}
        df = df.assign(item=df["name"].map(names)).dropna(subset=["item"])
        df["item"] = df["item"].str.capitalize()
        return df[["unit_id", "item"]].drop_duplicates().assign(code="")
    if spec.ranked == "procedures":
        return df.rename(columns={"name": "item"}).dropna(subset=["item"]).assign(code="")[["unit_id", "item", "code"]]
    df["code"] = "ICD-" + df["version"].astype(str) + " " + df["code"].astype(str)
    df["item"] = df["title"].fillna(df["code"]) if "title" in df else df["code"]
    return df[["unit_id", "item", "code"]].drop_duplicates()


def _share(k, n):
    if not n or 0 < k < SMALL:
        return None, None, None
    p, lo, hi = S.wilson(k, n)
    return round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)


def run_top(spec, cohort, mode):
    from mes.tools.rwd.analyses import base_result
    from mes.tools.rwd.concepts import get_index
    from mes.tools.rwd.db import get_store
    from mes.tools.rwd.patterns import groups_of
    res = base_result(spec, cohort)
    df = cohort.df.reset_index(drop=True)
    n = len(df)
    kind = RANKED.get(spec.ranked, "items")
    store = getattr(cohort, "store", None) or get_store()
    try:
        index = get_index()
    except Exception:
        index = None
    got = pairs(store, cohort, spec, index)
    t, names, how = groups_of(spec, df)
    counts = got.groupby("item")["unit_id"].nunique()
    counts = counts[counts >= SMALL]
    codes = got.groupby("item")["code"].agg(lambda c: ", ".join(sorted(set(c))[:3])) if len(got) else pd.Series(dtype=str)
    n1 = int(np.nansum(t == 1)) if t is not None else 0
    n0 = int(np.nansum(t == 0)) if t is not None else 0
    k1s = k0s = pd.Series(dtype=float)
    if t is not None:
        g = got.merge(pd.DataFrame({"unit_id": df["unit_id"].values, "g": t}), on="unit_id", how="left")
        k1s = g[g["g"] == 1].groupby("item")["unit_id"].nunique()
        k0s = g[g["g"] == 0].groupby("item")["unit_id"].nunique()
    # "the commonest diagnoses of patients who die": ranked in the group with the outcome
    order = k1s.reindex(counts.index).fillna(0) if how == "outcome" else counts
    rows = []
    for item in order.sort_values(ascending=False).index[:TOP]:
        k = int(counts[item])
        share, lo, hi = _share(k, n)
        row = {"item": item, "code": codes.get(item, "") if spec.ranked == "diagnoses" else "", "patients": k,
               "share": share, "lo": lo, "hi": hi}
        if t is not None:
            row.update(a=_share(int(k1s.get(item, 0)), n1)[0], b=_share(int(k0s.get(item, 0)), n0)[0])
        rows.append(row)
    res["tables"]["ranking"] = rows
    res["groups"] = {"names": names, "how": how, "n": [n1, n0] if t is not None else [n]}
    window = f"within {spec.horizon_h:g} h of {L.BASES[spec.base]['t0_label']}" if float(spec.horizon_h or 0) > 0 \
        else ("during the ICU stay" if spec.base == "icu" else "during the admission")
    for i, r in enumerate(rows[:3]):
        res["estimates"].append({"label": f"{r['item']}: patients with it ({kind}" + (f" {window}" if spec.ranked != "diagnoses"
                                                                                       else "") + ")",
                                 "measure": "rate", "value": r["share"], "lo": r["lo"], "hi": r["hi"], "n": n,
                                 "events": r["patients"], "primary": i == 0})
    if not rows:
        res["warnings"].append(f"No item of the {kind} was found in 10 or more patients.")
    res["methods"] = (f"The {kind} of each patient " + (window if spec.ranked != "diagnoses" else "of the admission (ICD "
                      "codes assigned at discharge)") + ("; medications by ingredient (the first word of the name) "
                                                          "from orders and ICU infusions — an order is not proof the "
                                                          "drug was given" if spec.ranked == "drugs" else "")
                      + f"; the share of patients with each, Wilson 95% CIs; items of fewer than {SMALL} patients are "
                      "not shown.")
    return res


def text(spec, res):
    """The passage of a ranking (aggregates only)."""
    rows = (res.get("tables") or {}).get("ranking") or []
    kind = RANKED.get(spec.ranked, "items")
    g = res.get("groups") or {}
    names = g.get("names") or []
    parts = []
    for r in rows[:12]:
        bit = f"{r['item']}{' (' + r['code'] + ')' if r.get('code') else ''}: {r['share']}%"
        if len(names) == 2 and (r.get("a") is not None or r.get("b") is not None):
            bit += f" ({names[0]} {r.get('a') if r.get('a') is not None else '<10 patients'}% vs {names[1]} " \
                   f"{r.get('b') if r.get('b') is not None else '<10 patients'}%)"
        parts.append(bit)
    return f"The commonest {kind} of the {res.get('n') or 0:,} patients: " + "; ".join(parts) + "." if parts else \
        f"No item of the {kind} was found in 10 or more patients."
