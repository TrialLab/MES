"""
A battery of real-world-data questions of many kinds, asked of MIMIC-IV alone with the configured language model:
where each one goes (study type, population, outcome), what it answers, its warnings — for finding the questions a
design misreads (each fix then becomes a rule with a test: tools/rwd/intent_rules.py, designer.to_spec). With
``--why``, the model's intent of every attempt, the term → concept mapping and the notes are printed too.

    python -m mes.scripts.rwd_battery [--out battery.json] [--why] ["a question" …]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

QUESTIONS = [
    "How many patients in MIMIC-IV received ECMO?",
    "What is the median age of ICU patients with sepsis?",
    "Is early norepinephrine within 6 hours associated with lower 28-day mortality in septic shock compared with no "
    "norepinephrine?",
    "What proportion of septic shock patients receive vasopressin, and when is it started?",
    "Which factors predict 28-day mortality in ICU patients with pneumonia?",
    "How has in-hospital mortality of ICU patients with AKI changed over the years?",
    "Compare ICU length of stay between the medical and the surgical ICU.",
    "A 75-year-old woman with cirrhosis, bilirubin 4, creatinine 2.1, on norepinephrine: what happened to similar "
    "patients?",
    "What lab tests are available in MIMIC-IV for kidney function?",
    "How does creatinine change over the first 3 days in ICU patients with sepsis, survivors vs non-survivors?",
    "In what order are vancomycin, piperacillin-tazobactam and cefepime started in sepsis?",
    "How do ICU patients readmitted within 30 days differ from those not readmitted?",
    "How many ICU patients have diabetes?",
    "What is the incidence of delirium in the ICU?",
    "Average time from hospital admission to ICU admission by admission type",
    "Survival of ventilated vs non-ventilated patients with sepsis over one year",
    "What percentage of ICU patients die in the ICU, by age group?",
    "Are women less likely than men to receive invasive mechanical ventilation in the ICU?",
    "What is the 1-year mortality of ICU patients over 80 years old?",
    "How often do patients see their primary care doctor after discharge?",
    "Which drugs are most commonly given in the first 24 hours of ICU admission?",
    "How many ICU patients had a positive blood culture?",
    "Show the distribution of SOFA scores by ICU type",
    "Does the use of statins relate to lower mortality in sepsis?",
    "脓毒症休克患者的28天死亡率是多少？",
    "一位70岁男性，SOFA 12分，乳酸5，类似病人的结局如何？",
    "Is a lactate above 4 associated with higher mortality in pneumonia?",
    "What happens to heart rate over the first 48 hours in septic shock?",
    "How many patients are in MIMIC-IV?",
    "Compare 90-day mortality of hydrocortisone vs no hydrocortisone in septic shock",
    "What are the most commonly used antibiotics in sepsis?",
    "What are the most common diagnoses of ICU patients who die within 28 days?",
    "How many hours of mechanical ventilation do patients with ARDS receive?",
]


def ask(question, mode, why=False):
    """One question through the planner and the MIMIC-IV search: a summary row (and the design's trace)."""
    from mes.llm_zoo import create_llm
    from mes.tools.planner_tools import plan_query
    from mes.tools.rwd.report import estimate_text
    from mes.tools.rwd_tools import search_rwd
    t = time.time()
    llm = create_llm()
    plan = plan_query(question, llm)
    items, stats = search_rwd(plan, mode, question, llm=llm)
    studies = stats.get("studies") or []
    main = next((s for s in studies if s.get("status") == "done"), studies[0] if studies else {})
    res, info = main.get("result") or {}, main.get("design_info") or {}
    row = {"q": question, "s": round(time.time() - t, 1),
           "studies": [(s.get("status"), (s.get("spec") or {}).get("analysis"), (s.get("result") or {}).get("n"))
                       for s in studies],
           "estimates": [estimate_text(e) for e in (res.get("estimates") or [])][:3],
           "rows": ((res.get("tables") or {}).get("result") or [])[:3], "notes": info.get("notes") or [],
           "routed": info.get("routed"), "warnings": res.get("warnings") or []}
    if why:
        row["attempts"] = [{"intent": a.get("intent"), "errors": a.get("errors")} for a in info.get("attempts") or []]
        row["mapping"], row["spec"], row["sql"] = info.get("mapping"), main.get("spec"), res.get("sql")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", nargs="*", help="questions to ask (default: the battery)")
    parser.add_argument("--out", type=Path, default=None, help="JSON of every answer")
    parser.add_argument("--why", action="store_true", help="also the model's intents, the mapping and the SQL")
    parser.add_argument("--mode", default="osler")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mes import config
    from mes.tools.pipeline_tools import warm_up
    warm_up()
    mode = config.get_mode(args.mode)
    rows, t0 = [], time.time()
    for question in args.questions or QUESTIONS:
        try:
            row = ask(question, mode, args.why)
        except Exception as e:
            row = {"q": question, "exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        rows.append(row)
        kind = next((s[1] for s in row.get("studies") or [] if s[0] == "done"), "—")
        first = (row.get("estimates") or [""])[0] or json.dumps(row.get("rows") or row.get("exception") or "")[:120]
        print(f"{kind or '—':13} {question[:70]:70} | {first[:110]}", flush=True)
        if args.why:
            print(json.dumps({k: row.get(k) for k in ("attempts", "mapping", "notes", "spec", "sql")}, ensure_ascii=False,
                             default=str, indent=1)[:4000], flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print(f"{len(rows)} questions in {round(time.time() - t0)} s; {sum('exception' in r for r in rows)} exceptions")


if __name__ == "__main__":
    main()
