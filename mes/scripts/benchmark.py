"""
MES latency / grounding benchmark.

Runs questions through the pipeline (no cache) and reports, per question:
time to first answer token, total time, LLM calls, evidence counts, citation
coverage, invalid citations and number grounding. Results are written to
``.cache/benchmark/<timestamp>.json``.

Usage:
    python -m mes.scripts.benchmark --mode osler [--n 6] [--questions "Q1" "Q2"]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root → `mes` importable

from mes import config  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is the risk of lactic acidosis in patients taking metformin, and how does it change with chronic kidney disease severity?",
    "In type 2 diabetes, is semaglutide more effective for weight loss than empagliflozin?",
    "Is a high C-reactive protein (CRP) level a prognostic factor for future cardiovascular events?",
    "Does sacubitril/valsartan reduce the risk of hospitalization in patients with heart failure (HFrEF)?",
    "Is apixaban superior to warfarin for stroke prevention in non-valvular atrial fibrillation?",
    "Is childhood obesity a risk factor for developing type 2 diabetes in adulthood?",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="osler", type=config.mode_key,
                        choices=list(config.MODES), help="osler|sackett|snow, or fast|balanced|deep")
    parser.add_argument("--n", type=int, default=len(DEFAULT_QUESTIONS))
    parser.add_argument("--questions", nargs="*")
    args = parser.parse_args()

    from mes.llm_zoo import create_llm
    from mes.tools.pipeline_tools import answer_question, warm_up

    print(f"warm-up {warm_up():.1f}s")
    llm = create_llm()
    rows = []
    for question in (args.questions or DEFAULT_QUESTIONS)[:args.n]:
        t = time.time()
        try:
            result = answer_question(question, llm, mode=args.mode, use_cache=False)
        except Exception as e:
            rows.append({"question": question, "error": str(e)})
            print(f"FAILED ({time.time() - t:.0f}s): {question}\n  {e}")
            continue
        v = result.verification
        counts = v.get("counts", {})
        row = {
            "question": question, "mode": args.mode, "llm": result.llm,
            "first_token_s": result.timings.get("first_token"), "total_s": result.timings.get("total"),
            "timings": result.timings, "llm_calls": result.n_llm_calls,
            "evidence": {t: sum(1 for e in result.evidence if e.source_type == t) for t in ("literature", "trial", "kg")},
            "cited": len(result.citation_order), "citation_coverage": v.get("citation_coverage"),
            "invalid_citations": len(v.get("invalid_citations", [])),
            "numbers": v.get("n_numbers"), "numbers_grounded": v.get("n_numbers_grounded"),
            "flagged": counts.get("unsupported_number", 0) + counts.get("uncited_number", 0) + counts.get("unsupported", 0),
            "status_counts": counts, "answer_chars": len(result.answer_markdown),
        }
        rows.append(row)
        print(f"{row['total_s']:5.1f}s total | first token {row['first_token_s']:5.1f}s | {row['llm_calls']} LLM calls | "
              f"evidence {row['evidence']} | cited {row['cited']} | coverage {row['citation_coverage']} | "
              f"numbers {row['numbers_grounded']}/{row['numbers']} | invalid {row['invalid_citations']} | "
              f"flagged {row['flagged']} | {question[:70]}")
    ok = [r for r in rows if "error" not in r]
    if ok:
        mean = lambda k: sum(r[k] for r in ok) / len(ok)
        print(f"\nmean total {mean('total_s'):.1f}s | mean first token {mean('first_token_s'):.1f}s | "
              f"mean LLM calls {mean('llm_calls'):.1f} | mean coverage {mean('citation_coverage'):.2f} | "
              f"numbers grounded {sum(r['numbers_grounded'] for r in ok)}/{sum(r['numbers'] for r in ok)} | "
              f"invalid citations {sum(r['invalid_citations'] for r in ok)} | failures {len(rows) - len(ok)}")
    out = config.CACHE_DIR / "benchmark" / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
