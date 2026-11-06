"""
Synthesis tools — evidence pack, streamed synthesis, citations and claim checks.

* ``build_evidence_pack``: numbered evidence blocks ([L1], [T1], [K1], [R1]) with
  design, venue and the selected passages, fitted to a character budget.
* ``synthesis_prompt`` / ``stream_synthesis``: one streamed LLM call writing a
  markdown answer with fixed headings and source-ID citations. The principles
  follow the MES Supervisor final report (bottom line, evidence classes kept
  apart, conflicts and gaps reported, only numbers from the inputs, KG relations
  as mechanistic context only).
* ``finalize_answer``: citation parsing, removal of unknown IDs, renumbering
  [1], [2], ... in order of first citation, section and takeaway parsing.
* ``ground_sentences``: deterministic check that every number of a cited
  sentence occurs in the cited evidence (no LLM call).
* ``extract_study_table`` (batched study extraction) and ``verify_claims``
  (batched LLM claim check).
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from decimal import ROUND_HALF_UP, Decimal

from mes.tools.result_schema import PREFIX_TO_TYPE, SOURCE_TYPES
from mes.utils import parse_json_response

logger = logging.getLogger(__name__)

SECTION_ORDER = ["Key takeaways", "Answer", "Certainty of evidence", "Evidence from published studies",
                 "Clinical trials", "Knowledge-graph context", "Real-world evidence",
                 "Agreement, conflicts and gaps", "Limitations"]


# ============================================================================ evidence pack
def _header(item):
    bits = [item.design]
    if item.source_type == "literature":
        bits += [item.venue, str(item.year or "")]
        if item.identifiers.get("pmid"):
            bits.append(f"PMID {item.identifiers['pmid']}")
        if item.full_text:
            bits.append("full text")
        if item.meta.get("named"):
            bits.append("the article the question names")
        if item.meta.get("landmark"):
            bits.append(f"landmark study ({item.meta.get('cites_per_year', 0):.0f} citations per year)")
        if item.origin in ("Local PDF library", "Uploaded PDF"):
            bits.append(f"{item.origin}: {item.identifiers.get('file', '')}")
    elif item.source_type == "trial":
        m = item.meta
        bits += [item.identifiers.get("nct", ""), str(m.get("status", "")).replace("_", " ").lower()]
        if m.get("enrollment"):
            bits.append(f"n={m['enrollment']}")
    elif item.source_type == "kg":
        bits += [f"reliability {item.meta.get('reliability')}"]
        if item.meta.get("contradicted"):
            bits.append("contradictory relations exist for this pair")
    elif item.source_type == "rwd":
        bits += [item.venue, "new analysis of local real-world data for this question"]
        if item.meta.get("analysis") == "custom":
            bits.append(f"custom analysis: review {item.meta.get('review') or 'not done'}, "
                        f"reproduced {'yes' if item.meta.get('reproduced') else 'no'}")
        if item.meta.get("n"):
            bits.append(f"n={item.meta['n']:,}")
    return " · ".join(b for b in bits if b)


_EFFECT_TERMS = re.compile(r"\b(?:a?HR|a?OR|a?RR|IRR|SMD|WMD|NNT|hazard|odds|risk ratio|relative risk|95\s?%|CI\b|"
                           r"p\s?[<=>]|significant)|\d+(?:\.\d+)?\s?%", re.I)


def compress_passage(text, limit):
    """
    Shorten ``text`` to about ``limit`` characters by keeping whole sentences, preferring those with effect
    estimates and numbers (kept in their original order), instead of cutting off the end where results usually are.
    """
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z0-9(])", text)
    ranked = sorted(range(len(sentences)),
                    key=lambda i: (-len(_EFFECT_TERMS.findall(sentences[i])) - 0.5 * bool(re.search(r"\d", sentences[i])), i))
    keep, used = set(), 0
    for i in ranked:
        if used + len(sentences[i]) > limit and keep:
            continue
        keep.add(i)
        used += len(sentences[i]) + 1
    out = " ".join(sentences[i] for i in sorted(keep))
    return out if len(out) <= limit * 1.2 else out[:limit].rsplit(" ", 1)[0] + " …"


def _block(item, passage_chars):
    lines = [f"[{item.id}] {_header(item)}", f"Title: {item.title}"]
    if item.source_type != "kg":
        # RWD results are few and carry the numbers of the analysis: keep them nearly whole
        limit = max(passage_chars, 2400) if item.source_type == "rwd" else passage_chars
        for p in item.passages:
            lines.append(f"> ({p.section}) {compress_passage(p.text, limit)}")
    extraction = item.meta.get("extraction")
    if extraction:
        fields = ", ".join(f"{k}: {v}" for k, v in extraction.items() if v not in (None, "", "Not reported"))
        lines.append(f"Extracted: {fields}")
    return "\n".join(lines)


def build_evidence_pack(items, budget_chars):
    """Evidence blocks grouped by source class, shortened (then truncated) to fit ``budget_chars``."""
    groups = {t: [it for it in items if it.source_type == t] for t in SOURCE_TYPES}
    for passage_chars in (1400, 1000, 700, 450, 250):
        sections = []
        for t, members in groups.items():
            if members:
                blocks = "\n\n".join(_block(it, passage_chars) for it in members)
                sections.append(f"### {SOURCE_TYPES[t]['label']}\n{blocks}")
        pack = "\n\n".join(sections)
        if len(pack) <= budget_chars:
            return pack
    return pack[:budget_chars]


# ============================================================================ prompt
SYSTEM_PROMPT = """You are MES, a medical evidence synthesis assistant for clinicians and researchers.
You answer clinical questions strictly from the numbered evidence you are given, with inline citations."""

RULES = """RULES (mandatory)
1. Start the "Answer" section with a direct answer to the question and say how strong and consistent the evidence is.
2. Every factual statement must cite its evidence IDs in square brackets right after the statement, e.g. [L2] or [L2][T1].
   Cite only IDs that appear in the EVIDENCE list. Never invent studies, IDs, numbers or URLs.
3. Use only numbers that appear in the evidence (effect estimates with 95% CI and p-values when given, sample sizes, rates).
   Report what is given; never write that a number is "not provided" or "not reported".
4. Keep evidence classes apart and name them: systematic reviews / meta-analyses / guidelines and randomized trials
   are stronger than observational studies, which are stronger than case reports, narrative reviews and editorials.
   A trial registration without posted results only shows that the trial exists (its design and status), not its results.
   Knowledge-graph (iBKH) relations are mechanistic or contextual information, never evidence of a clinical effect.
5. When sources disagree, say so explicitly, give each estimate with its source, and suggest reasons (population,
   design, dose, outcome definition, follow-up). Do not force a consensus.
6. When the evidence does not cover part of the question (population, comparator, outcome), state the gap instead of filling it.
7. Describe each study's population and setting as the evidence states it; do not transfer the setting of the question
   (e.g. ICU) to a study whose population differs. A trial of the same comparison in a broader population (e.g.
   hospitalized adults for a question about ICU patients) still informs the answer: report it and name its population.
8. General medical knowledge may only explain mechanisms or context, must be labelled "Background:" and is never cited.
9. If a section has no relevant evidence, write one sentence saying so.
10. Output only the final answer: no plan, no notes about the task. Write plain text symbols (≥, ≤, m², ±), never LaTeX.
11. Real-world data items (R…) are new observational analyses of MIMIC-IV (one US academic hospital, adult hospital
   and ICU stays 2008-2019) computed for this question. Report them in "Real-world evidence" as "In MIMIC-IV, ..."
   with the design, the number of patients and the estimate with its 95% CI, copying the numbers exactly; state the
   warnings they list (residual imbalance, limited overlap, few events, exploratory query). Never present them as trial
   evidence; when they disagree with randomized trials, prefer the trials and name residual confounding as a likely
   reason. Descriptive results describe this population only. If an R item says the question is not answerable with
   MIMIC-IV, say so in one sentence. An R item whose design is "Custom analysis (code written by the model)" was
   written for this question by the MES analyst: call it an exploratory custom analysis and report its review and
   reproduction status when they are not clean. When two R items analyse the same question, compare them."""

ANSWER_PREFILL = "## Key takeaways\n"

TABLE_INSTRUCTION = ("a markdown table of the most informative studies and trials (at most {rows} rows) with the "
                     "columns | Study | Design | Population | n | Comparison | Outcome | Effect (95% CI) | — put the "
                     "citation ID in the Study cell (e.g. \"[L2] Smith 2021\"), copy numbers exactly, write NR when "
                     "not reported")

MODE_STYLE = {
    "osler": ("Be concise: about 300-450 words in total. Build the answer on the strongest and most direct evidence: "
              "the randomized trials and meta-analyses that address the question itself (the same treatments, "
              "comparison and outcome) and the landmark studies of the list — name them, cite them and give their main "
              "effect estimates with 95% CI."),
    "sackett": ("Weigh the evidence: about 450-700 words of text plus the table. In \"Certainty of evidence\", rate "
                "the certainty for each main outcome (High / Moderate / Low / Very low, GRADE-style) with a one-line "
                "reason (design, risk of bias, inconsistency, indirectness, imprecision)."),
    "snow": ("Write a comprehensive survey: about 700-1100 words of text plus the table. After \"Answer\", add one "
             "section per facet listed below (use the facet name as the heading). In \"Certainty of evidence\", rate "
             "each main outcome (High / Moderate / Low / Very low) with a one-line reason."),
}


def _format_instructions(mode, plan, sources):
    short = mode == "osler"       # fast: the same sections, each kept short
    sections = ["## Key takeaways\n(2-3 bullet points, each with citations)" if short else
                "## Key takeaways\n(2-4 bullet points, each with citations)", "## Answer"]
    if mode in ("sackett", "snow"):
        sections.append("## Evidence table\n(" + TABLE_INSTRUCTION.format(rows=8 if mode == "sackett" else 12) + ")")
        sections.append("## Certainty of evidence")
    if mode == "snow":
        sections += [f"## {f['name']}" for f in plan.facets]
    if "literature" in sources:
        sections.append("## Evidence from published studies" + ("\n(the 3-5 strongest studies, one or two sentences "
                                                                 "each)" if short else ""))
    if "trials" in sources:
        sections.append("## Clinical trials" + ("\n(one to three sentences)" if short else ""))
    if "kg" in sources:
        sections.append("## Knowledge-graph context" + ("\n(one sentence)" if short else ""))
    if "rwd" in sources:
        sections.append("## Real-world evidence")
    sections += ["## Agreement, conflicts and gaps" + ("\n(one or two sentences)" if short else ""),
                 "## Limitations" + ("\n(one or two sentences)" if short else "")]
    return "\n".join(sections)


def _history_text(history):
    if not history:
        return ""
    turns = "\n".join(f"Q: {h['question']}\nA (summary): {h.get('summary', '')[:500]}" for h in history[-3:])
    return (f"EARLIER TURNS (context only; answer the CURRENT QUESTION; never cite earlier turns, cite only the "
            f"evidence IDs below):\n{turns}\n\n")


def synthesis_prompt(plan, pack, mode, sources, history=None):
    pico = "; ".join(f"{k}: {v}" for k, v in plan.pico.items() if v)
    facets = "\n".join(f"- {f['name']}: {f['question']}" for f in plan.facets) if mode == "snow" and plan.facets else ""
    return f"""{_history_text(history)}CURRENT QUESTION: {plan.query}
Question type: {plan.question_type}{f'; PICO: {pico}' if pico else ''}
{f'FACETS TO COVER:{chr(10)}{facets}{chr(10)}' if facets else ''}
EVIDENCE
{pack or '(no evidence was retrieved)'}

{RULES}

STYLE: {MODE_STYLE.get(mode, MODE_STYLE['osler'])}
Write the content in {plan.language}, but keep these markdown headings exactly as written (in English):
{_format_instructions(mode, plan, sources)}

Write the answer now, starting with "## Key takeaways"."""


def stream_synthesis(plan, items, llm, mode_cfg, sources, budget_chars, history=None):
    """Yield ``(kind, text)`` chunks of the synthesis; returns nothing else."""
    pack = build_evidence_pack(items, budget_chars)
    prompt = synthesis_prompt(plan, pack, mode_cfg.key, sources, history)
    system = SYSTEM_PROMPT
    if plan.language and plan.language.strip().lower() not in ("english", "en"):
        system += (f"\nWrite every sentence, including the key takeaways, in {plan.language}, using the standard "
                   f"{plan.language} names of drugs and diseases (do not mix languages inside a term). "
                   f"Only the markdown headings stay in English.")
    yield from llm.stream(prompt, system=system, max_tokens=mode_cfg.synthesis_tokens,
                          reasoning=mode_cfg.reasoning, prefill=ANSWER_PREFILL)


class PreambleGate:
    """
    Streaming filter that drops anything a model writes before the first "## " heading
    (some models narrate their plan). Text is released unchanged once a heading is seen,
    or after ``max_hold`` characters without one.
    """

    def __init__(self, max_hold=3500):
        self.max_hold = max_hold
        self.buffer = ""
        self.open = False
        self.released = ""
        self.repeated = False   # the model started writing the whole answer a second time

    def feed(self, text):
        out = self._feed(text)
        if out:
            self.released += out
            first = self.released.find("## Key takeaways")
            again = self.released.find("## Key takeaways", first + 1) if first >= 0 else -1
            if again > 0:
                self.repeated = True
                out = out[:max(0, len(out) - (len(self.released) - again))]
                self.released = self.released[:again]
        return out

    def _feed(self, text):
        if self.open:
            return text
        self.buffer += text
        m = re.search(r"(?m)^\s*##\s", self.buffer)
        if m:
            self.open = True
            return self.buffer[m.start():].lstrip()
        if len(self.buffer) > self.max_hold:
            self.open = True
            return self.buffer
        return ""


# ============================================================================ citations and sections
_CITE_GROUP = re.compile(r"\[\s*((?:[LTKR]\d+)(?:\s*[,;/]\s*\[?\s*[LTKR]\d+\s*\]?)*)\s*\]")
_PAREN_CITE = re.compile(r"\(\s*((?:[LTKR]\d+)(?:\s*[,;]\s*[LTKR]\d+)*)\s*\)")
_ID = re.compile(r"[LTKR]\d+")


# Made-up citations of the conversation itself, e.g. "[Summary from previous turn]"
_TURN_CITE = re.compile(r"\s?\[[^\]\n]{0,40}\b(?:previous|earlier|prior)\s+(?:turn|answer|response|question)s?\b"
                        r"[^\]\n]{0,30}\]", re.I)


def normalize_citations(text):
    """Rewrite citation variants — "(L1)", "[L1, T2]", "[L1; T2]" — as "[L1][T2]"; drop citations of earlier turns."""
    text = _TURN_CITE.sub("", text)
    text = _PAREN_CITE.sub(lambda m: "".join(f"[{i}]" for i in _ID.findall(m.group(1))), text)
    return _CITE_GROUP.sub(lambda m: "".join(f"[{i}]" for i in _ID.findall(m.group(1))), text)


def split_sections(text):
    """{heading: body} for "## heading" blocks (text before the first heading goes to "Answer")."""
    sections, current, buf = {}, "Answer", []
    for line in text.splitlines():
        m = re.match(r"^\s*#{1,4}\s*(.+?)\s*#*\s*$", line)
        if m:
            if "".join(buf).strip():
                sections[current] = (sections.get(current, "") + "\n" + "\n".join(buf)).strip()
            current, buf = m.group(1).strip().rstrip(":"), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections[current] = (sections.get(current, "") + "\n" + "\n".join(buf)).strip()
    return sections


def _drop_leading_citations(line):
    """'- [L1] Text ... [L1].' → '- Text ... [L1].' (a citation at the start of a bullet that is repeated later)."""
    m = re.match(r"^(\s*(?:[-*•]|\d+\.)?\s*)((?:\[[LTKR]\d+\])+)\s*(.*)$", line)
    if not m:
        return line
    prefix, lead, rest = m.groups()
    if all(c in rest for c in re.findall(r"\[[LTKR]\d+\]", lead)):
        return prefix + rest
    return line


def finalize_answer(raw, valid_ids):
    """
    Normalize and validate citations, renumber them [1], [2], ... by first appearance.
    Returns (markdown with source IDs, display markdown with numbers, citation_order, invalid_ids, sections, takeaways).
    """
    text = normalize_citations(raw.strip())
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    first_heading = re.search(r"(?m)^\s*##\s", text)
    if first_heading:  # drop any narrated plan before the first heading
        text = text[first_heading.start():].lstrip()
    text = re.sub(r"\$\\(?:ge|geq)\$", "≥", text)
    text = re.sub(r"\$\\(?:le|leq)\$", "≤", text)
    text = re.sub(r"\$\^?\{?2\}?\$", "²", text)
    text = re.sub(r"\]\s*[,;]\s*\[(?=[LTKR]\d+\])", "][", text)
    # a source ID written without brackets ("… was not reported in L1"): a citation, not an internal code
    text = re.sub(r"\b(in|from|see|by|per|study|source|trial)\s([LTKR]\d{1,2})\b(?![\w\-])",
                  lambda m: f"{m.group(1)} [{m.group(2)}]" if m.group(2) in valid_ids else m.group(0), text)
    text = "\n".join(_drop_leading_citations(line) for line in text.splitlines())
    invalid = sorted({i for i in re.findall(r"\[([LTKR]\d+)\]", text) if i not in valid_ids})
    text = re.sub(r"\[([LTKR]\d+)\]", lambda m: m.group(0) if m.group(1) in valid_ids else "", text)
    order = []
    for i in re.findall(r"\[([LTKR]\d+)\]", text):
        if i not in order:
            order.append(i)
    index = {i: n + 1 for n, i in enumerate(order)}
    display = re.sub(r"\[([LTKR]\d+)\]", lambda m: f"[{index[m.group(1)]}]", text)
    sections = split_sections(text)
    takeaways = [re.sub(r"^\s*[-*•]\s*", "", ln).strip() for ln in sections.get("Key takeaways", "").splitlines()
                 if re.match(r"^\s*[-*•]\s+", ln)]
    return text, display, order, invalid, sections, takeaways


# ============================================================================ grounding (deterministic)
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z一-鿿(])|\n+")
_CLAIM_SECTIONS_EXEMPT = ("Limitations", "Agreement, conflicts and gaps")
_DESIGN_NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s?(?:mg|g|µg|mcg|ml|mL|IU|units?|weeks?|wk|months?|mo|years?|yr|days?|"
                            r"hours?|h|min)\b", re.I)
# Doses listed with a slash ("1.0/2.4 mg"): every number of the list is part of the design
_UNITS = (r"mg|g|µg|mcg|ml|mL|IU|units?|weeks?|wk|months?|mo|years?|yr|days?|hours?|h|min|mg/g|"
          r"ml/min(?:/1\.73\s?m2)?|mmol/mol|kg/m2")
# Doses and periods listed or given as ranges ("1.0/2.4 mg", "26–52 weeks"): every number of the list is design
_DOSE_LIST = re.compile(rf"(\d+(?:\.\d+)?)(?=(?:\s?(?:/|–|-|to)\s?\d+(?:\.\d+)?)+\s?(?:{_UNITS})\b)", re.I)
# Thresholds that define a group ("eGFR <30", "LVEF ≤40%") rather than report a result
_THRESHOLD = re.compile(r"(?:<|>|≤|≥|=<|>=)\s?(\d+(?:\.\d+)?)")
# ... and the ones that are certainly cut-offs of a clinical measure, never checked against the sources
_CLINICAL_THRESHOLD = re.compile(r"\b(?:eGFR|GFR|CrCl|LVEF|EF|BMI|HbA1c|A1c|age|SBP|DBP|NT-proBNP|BNP|UACR|ACR)"
                                 r"\s?(?:of\s)?(?:<|>|≤|≥|=<|>=|(?:below|under|above|over|less than|more than|"
                                 r"at least)\s)\s?(\d+(?:\.\d+)?)"
                                 r"|\b(?:below|under|above|over)\s(?:an?\s)?(?:eGFR|GFR|CrCl|LVEF|BMI|HbA1c)(?:\sof)?\s"
                                 r"(\d+(?:\.\d+)?)", re.I)
_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:[,\u2009\u202f\u00a0 ]\d{3})+(?![\d])")
_MIDDLE_DOT = re.compile(r"(?<=\d)[\u00b7\u2219\u22c5](?=\d)")      # Lancet decimals written with a middle dot
_BARE_DECIMAL = re.compile(r"(?<![\w.\d])\.(?=\d)")                  # JAMA style: P = .002
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_ONES = "one two three four five six seven eight nine".split()
_REDUCTION = re.compile(r"reduc|lower|decreas|fewer|relative", re.I)
_ABSOLUTE = re.compile(r"absolute|difference|\bARR\b|\bARD\b", re.I)


def _number_word(n):
    """English words of 10–99 as a pattern: "twelve", "thirty-seven" / "thirty seven"."""
    if 10 <= n < 20:
        return _TEENS[n - 10]
    if 20 <= n < 100:
        tens = _TENS[n // 10 - 2]
        return tens if n % 10 == 0 else rf"{tens}[- ]{_ONES[n % 10 - 1]}"
    return None


def normalize_numbers(text):
    """
    Write numbers one way so they compare across journal styles: no thousands separators ("13,194", "13 194"),
    "." for middle-dot decimals ("0\u00b732"), and a leading zero on bare decimals ("P = .002").
    """
    text = _MIDDLE_DOT.sub(".", _BARE_DECIMAL.sub("0.", text))
    return _THOUSANDS.sub(lambda m: re.sub(r"[^\d]", "", m.group(0)), text)


def _numbers(text):
    """Numbers of a sentence that a claim could rest on (years, citation numbers, list markers and the 1.73 m² of
    eGFR units excluded)."""
    text = normalize_numbers(re.sub(r"\[[LTKR]?\d+\]", " ", text))
    text = re.sub(r"1\.73\s*m(?:²|\^?2)", "m²", text)
    out = []
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "")
        try:
            number = float(value)
        except ValueError:
            continue
        if 1900 <= number <= 2100 and "." not in value:
            continue
        if number < 10 and "." not in value:   # "type 2", "phase 3", small counts: too ambiguous to check
            continue
        out.append(value)
    return out


def _in_evidence(value, blob):
    if value in blob:
        return True
    if "." in value:  # 0.80 vs 0.8
        return value.rstrip("0").rstrip(".") in blob
    word = _number_word(int(value))  # "Twelve studies were included", "Thirty-seven studies"
    return bool(word and re.search(rf"\b{word}\b", blob, re.I))


def _rounded_forms(text):
    """Correct roundings of the source's decimals ("1.0881" → "1.09", "1.1"; "49.7" → "50"), which answers may
    report. Whole numbers below 10 are never checked, so rounding to one does not hide a wrong small number."""
    forms = set()
    for raw in set(re.findall(r"(?<![\d.])\d{1,12}\.\d+(?!\d)", text)):
        value, decimals = Decimal(raw), len(raw.split(".")[1])
        # long digit runs (identifiers, survival-curve arrays) are not results an answer rounds
        forms.update(str(value.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)) for d in range(min(decimals, 8)))
    return " ".join(sorted(forms))


def _grounding_text(item):
    # "1.29 to 2. 17": some PubMed abstracts carry a space inside a decimal number
    text = re.sub(r"(?<=\d)\. (?=\d)", ".", normalize_numbers(item.text_blob()).replace(",", ""))
    return f"{text}\n{_rounded_forms(text)}"


def _distinctive(value):
    """A single number specific enough to identify its source (e.g. "13194", "0.6712"; not "1.09" or "12")."""
    return len(re.sub(r"\D", "", value).lstrip("0")) >= 4


def _absolute_difference(value, sentence, blob):
    """An absolute difference ("0.33%/yr") between two percentages of the source (1.60% vs 1.27% per year)."""
    if "." not in value or not _ABSOLUTE.search(sentence):
        return False
    decimals = len(value.split(".")[1])
    rates = {Decimal(r) for r in re.findall(rf"(?<![\d.])(\d+\.\d{{{decimals}}})\s?%", blob)}
    return any(abs(a - b) == Decimal(value) for a in rates for b in rates)


def _strong_fingerprint(numbers):
    """Enough numbers to identify the one source they come from."""
    return len(numbers) >= 3 or (len(numbers) == 2 and any("." in n or _distinctive(n) for n in numbers))


def _units(body):
    """Sentences of a section, each with the next sentence of its paragraph (None at the end); a table row stays one
    unit (a cell such as "(incl. ARNI)" is not a sentence end)."""
    for line in re.split(r"\n+", body):
        if line.lstrip().startswith("|"):
            yield line, None
        else:
            parts = _SENTENCE.split(line)
            yield from zip(parts, parts[1:] + [None])


def _relative_reduction(value, sentence, blob):
    """A "21% lower risk" read off a ratio of 0.79 in the source (1 − ratio, as a percentage)."""
    if not (re.search(rf"(?<![\d.]){re.escape(value)}\s?%", sentence) and _REDUCTION.search(sentence)):
        return False
    ratios = {float(r) for r in re.findall(r"(?<![\d.])0\.\d{2,3}(?!\d)", blob)}
    return any(abs((1 - r) * 100 - float(value)) < 0.51 for r in ratios)


def _percent_of_fraction(value, sentence, blob):
    """A percentage ("37.2%") written for a proportion the source gives as a fraction (0.372 or 0.3724)."""
    if "." not in value and len(value) > 2 or not re.search(rf"(?<![\d.]){re.escape(value)}\s?%", sentence):
        return False
    try:
        fraction = str((Decimal(value) / 100).normalize())
    except ArithmeticError:
        return False
    return bool(re.search(rf"(?<![\d.]){re.escape(fraction)}(?!\d)", blob))


def ground_sentences(markdown, items_by_id):
    """Per-sentence citation and number-grounding status (no LLM)."""
    blobs = {i: _grounding_text(it) for i, it in items_by_id.items()}
    results = []
    for heading, body in split_sections(markdown).items():
        for sentence, following in _units(body):
            sentence = sentence.strip(" -*•\t")
            if len(sentence) < 25 or re.fullmatch(r"[|\s:\-]+", sentence):
                continue
            cites = list(dict.fromkeys(re.findall(r"\[([LTKR]\d+)\]", sentence)))
            if sentence.startswith("|") and not cites:   # table header row
                continue
            cutoffs = {x for m in _CLINICAL_THRESHOLD.findall(normalize_numbers(sentence)) for x in m if x}
            numbers = [n for n in _numbers(sentence) if n not in cutoffs]
            missing = [n for n in numbers if not any(_in_evidence(n, blobs[c]) or _relative_reduction(n, sentence, blobs[c])
                                                     or _absolute_difference(n, sentence, blobs[c])
                                                     or _percent_of_fraction(n, sentence, blobs[c])
                                                     for c in cites if c in blobs)]
            suggested = []
            if cites and missing and (len(missing) >= 2 or _distinctive(missing[0])):
                # numbers cited to the wrong source: do they all occur together in exactly one other item?
                # (a single short number such as "1.09" occurs in many papers by chance)
                owners = [i for i, b in blobs.items() if i not in cites and all(_in_evidence(n, b) for n in missing)]
                suggested = owners if len(owners) == 1 else []
            if cites:
                status = "unsupported_number" if missing else ("grounded" if numbers else "cited")
            else:
                # doses and time points ("14 mg", "26 weeks") describe designs, not results
                plain = normalize_numbers(sentence)
                design = set(_DESIGN_NUMBER.findall(plain) + _DOSE_LIST.findall(plain) + _THRESHOLD.findall(plain))
                result_numbers = [n for n in numbers if n not in design]
                status = "uncited_number" if result_numbers and heading not in _CLAIM_SECTIONS_EXEMPT else "uncited"
                after = [c for c in dict.fromkeys(re.findall(r"\[([LTKR]\d+)\]", following or "")) if c in blobs]
                if status == "uncited_number" and after and all(any(_in_evidence(n, blobs[c]) for c in after)
                                                                for n in result_numbers):
                    # a lead-in whose numbers the next sentence gives with its citation ("The best estimate is
                    # about 18% at 28 days. The analysis found 18.4% (95% CI 17.9–19.0) [R1].")
                    status, missing = "grounded", []
                elif status == "uncited_number" and not sentence.startswith("|") and _strong_fingerprint(result_numbers):
                    # a result written without its citation: add it when exactly one source has all the numbers
                    owners = [i for i, b in blobs.items() if all(_in_evidence(n, b) for n in result_numbers)]
                    suggested = owners if len(owners) == 1 else []
            results.append({"section": heading, "text": sentence, "citations": cites, "numbers": numbers,
                            "missing_numbers": missing, "status": status, "suggested_citations": suggested})
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    claims = [r for r in results if r["section"] not in ("Key takeaways",)]
    cited = sum(1 for r in claims if r["citations"])
    return {"sentences": results, "counts": counts,
            "citation_coverage": round(cited / len(claims), 3) if claims else 0.0,
            "n_numbers": sum(len(r["numbers"]) for r in results),
            "n_numbers_grounded": sum(len(r["numbers"]) - len(r["missing_numbers"]) for r in results if r["citations"])}


def repair_citations(markdown, grounding):
    """
    Add the source that actually contains a sentence's numbers (``suggested_citations``) right after the
    sentence's last citation, or at its end when it has none. Returns (markdown, list of repaired sentences).
    """
    repaired = []
    for s in grounding["sentences"]:
        if not s.get("suggested_citations"):
            continue
        new_id, text = s["suggested_citations"][0], s["text"]
        if text not in markdown:
            continue
        cites = list(re.finditer(r"\[[LTKR]\d+\]", text))
        if cites:
            fixed = text[:cites[-1].end()] + f"[{new_id}]" + text[cites[-1].end():]
        else:
            end = len(text.rstrip(".;: "))
            fixed = text[:end] + f" [{new_id}]" + text[end:]
        markdown = markdown.replace(text, fixed, 1)
        repaired.append({"text": fixed, "added": new_id, "numbers": s["missing_numbers"]})
    return markdown, repaired


_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def drop_repeats(markdown, min_len=40, threshold=0.9):
    """Sentences and bullets written twice in the same section (small local models sometimes loop): a sentence
    with the same numbers as an earlier one and nearly the same words is dropped. Tables and headings are kept."""
    from difflib import SequenceMatcher
    lines, seen = [], []
    for line in markdown.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            seen = []
            lines.append(line)
            continue
        if stripped.startswith("|") or not stripped:
            lines.append(line)
            continue
        bullet = _BULLET.match(line)
        head = bullet.group(0) if bullet else ""
        kept = []
        for part in _SENTENCE_END.split(line[len(head):]):
            text = re.sub(r"\[[LTKR]?\d+\]", "", part).lower()
            norm = " ".join(re.sub(r"[^\w%.]+", " ", text).split())
            numbers = sorted(_NUMBER.findall(text))
            if len(norm) >= 10:
                if any(numbers == n and (norm == s or (len(norm) >= min_len and
                                                       SequenceMatcher(None, norm, s).ratio() >= threshold))
                       for s, n in seen):
                    continue
                seen.append((norm, numbers))
            kept.append(part)
        if kept or not head:
            lines.append(head + " ".join(kept))
    return "\n".join(lines)


_LATEX = {"\\geq": "≥", "\\ge": "≥", "\\leq": "≤", "\\le": "≤", "\\pm": "±", "\\times": "×", "\\approx": "≈",
          "\\sim": "~", "\\%": "%", "\\cdot": "·", "\\neq": "≠", "\\ne": "≠", "\\rightarrow": "→", "\\to": "→",
          "\\alpha": "α", "\\beta": "β", "\\mu": "µ", "\\chi": "χ", "\\Delta": "Δ", "\\lt": "<", "\\gt": ">"}
_MATH = re.compile(r"\$\$?([^$\n]{1,80}?)\$\$?")
_SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def plain_math(text):
    """LaTeX math a model sometimes writes ("$\\ge 45\\%$", "mL/min/1.73 m$^2$") as plain text ("≥ 45%",
    "1.73 m²"); dollar amounts ("$5 vs $10") stay as they are."""
    def repl(m):
        inner = m.group(1)
        if re.fullmatch(r"\s*[A-Za-z]{1,3}\s*[<>=≤≥]{1,2}\s*-?[\d.]+\s*", inner):     # "$p < 0.05$"
            return inner.strip()
        if not re.search(r"\\[A-Za-z%]|\^", inner):
            return m.group(0)
        out = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^}]*)\}", r"\1", inner)
        out = re.sub(r"\\[A-Za-z]+|\\%", lambda c: _LATEX.get(c.group(0), ""), out)
        out = re.sub(r"\^\{?([0-9+-]+)\}?", lambda c: c.group(1).translate(_SUPERSCRIPT), out)
        return re.sub(r"[{}\\]", "", out).strip()
    return _MATH.sub(repl, text)


def postprocess_answer(raw, items_by_id):
    """
    Deterministic checks of a synthesized answer: repeated sentences removed, citation validation and renumbering,
    number grounding, and repair of citations whose numbers belong to another source.
    Returns (markdown, display, citation_order, sections, takeaways, grounding).
    """
    raw = plain_math(drop_repeats(raw))
    valid = set(items_by_id)
    markdown, display, order, invalid, sections, takeaways = finalize_answer(raw, valid)
    try:
        grounding = ground_sentences(markdown, items_by_id)
        fixed, repaired = repair_citations(markdown, grounding)
    except Exception as e:      # the check must never cost the answer
        logger.exception("Number grounding failed")
        grounding = {"sentences": [], "counts": {}, "citation_coverage": 0.0, "n_numbers": 0, "n_numbers_grounded": 0,
                     "error": f"{type(e).__name__}: {e}"[:300], "invalid_citations": invalid, "repaired": []}
        return markdown, display, order, sections, takeaways, grounding
    if repaired:
        markdown, display, order, _, sections, takeaways = finalize_answer(fixed, valid)
        grounding = ground_sentences(markdown, items_by_id)
        repaired_texts = {r["text"] for r in repaired}
        for s in grounding["sentences"]:
            if s["text"] in repaired_texts and not s["missing_numbers"]:
                s["status"] = "repaired"
        grounding["counts"] = {}
        for s in grounding["sentences"]:
            grounding["counts"][s["status"]] = grounding["counts"].get(s["status"], 0) + 1
    grounding["invalid_citations"] = invalid
    grounding["repaired"] = repaired
    return markdown, display, order, sections, takeaways, grounding


# ============================================================================ Sackett / Snow: extraction table
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"studies": {"type": "array", "items": {
        "type": "object",
        "properties": {k: {"type": "string"} for k in (
            "id", "design", "population", "sample_size", "intervention", "comparator", "outcome",
            "effect_estimate", "direction", "risk_of_bias", "key_limitation")},
        "required": ["id", "design", "population", "sample_size", "intervention", "comparator", "outcome",
                     "effect_estimate", "direction", "risk_of_bias", "key_limitation"]}}},
    "required": ["studies"],
}

EXTRACT_PROMPT = """Extract a structured summary of each study below for the question: "{question}".

For each study return: "id" (as given), "design", "population", "sample_size", "intervention", "comparator",
"outcome" (the outcome most relevant to the question), "effect_estimate" (measure, value, 95% CI and p-value exactly as
reported, e.g. "HR 0.79 (95% CI 0.66-0.94), p=0.01"), "direction" (benefit | harm | no difference | unclear),
"risk_of_bias" (Low | Moderate | High | Unclear, from the design and reporting) and "key_limitation".
Copy numbers exactly from the text. Use "Not reported" when the text does not state something.

STUDIES
{blocks}"""


def _check_numbers(row, blob):
    """Blank out extracted fields whose numbers do not occur in the source text (extraction guard)."""
    for key in ("sample_size", "effect_estimate"):
        value = str(row.get(key, ""))
        nums = [n for n in _NUMBER.findall(value) if not (n.isdigit() and int(n) < 10)]
        if nums and not all(_in_evidence(n.replace(",", ""), blob) for n in nums):
            row[key] = "Not reported (unverified value removed)"
    return row


def extract_study_table(question, items, llm, batch_size=4, max_workers=4):
    """Structured rows for literature and trial items (one LLM call per batch, batches in parallel)."""
    targets = [it for it in items if it.source_type in ("literature", "trial")]
    batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
    by_id = {it.id: it for it in targets}

    def run(batch):
        blocks = "\n\n".join(_block(it, 1200) for it in batch)
        raw = llm.invoke(EXTRACT_PROMPT.format(question=question, blocks=blocks), json_schema=EXTRACT_SCHEMA,
                         reasoning=False, max_tokens=300 * len(batch) + 200)
        parsed = parse_json_response(raw)
        rows = parsed.get("studies", []) if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
        return [r for r in rows if isinstance(r, dict) and r.get("id") in by_id]

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for batch_rows in pool.map(run, batches):
            rows += batch_rows
    for row in rows:
        item = by_id[row["id"]]
        _check_numbers(row, item.text_blob().replace(",", ""))
        item.meta["extraction"] = {k: v for k, v in row.items() if k != "id"}
    return rows


# ============================================================================ Sackett / Snow: claim verification
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "string"},
                       "status": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                       "issue": {"type": "string", "maxLength": 120}},
        "required": ["claim", "status", "issue"]}}},
    "required": ["verdicts"],
}

VERIFY_PROMPT = """You are a meticulous fact-checker. Check each CLAIM against the EVIDENCE it cites (and nothing else).
status "supported": the cited evidence states it; "partial": partly stated, overstated or missing a qualifier;
"unsupported": not stated in, or contradicted by, the cited evidence.
"issue": an empty string when supported, otherwise what is wrong in at most 12 words.

EVIDENCE
{evidence}

CLAIMS
{claims}"""

_VERDICT = re.compile(r'"claim"\s*:\s*"(C\d+)"\s*,\s*"status"\s*:\s*"(supported|partial|unsupported)"'
                      r'(?:\s*,\s*"issue"\s*:\s*"([^"]*)")?')


def _verify_batch(batch, items_by_id, llm):
    cited = list(dict.fromkeys(c for s in batch for c in s["citations"] if c in items_by_id))
    evidence = "\n".join(f"[{c}] {items_by_id[c].title}: " + " ".join(p.text for p in items_by_id[c].passages)[:600]
                         for c in cited)
    claims = "\n".join(f"{s['claim_id']}: {s['text']}" for s in batch)
    raw = llm.invoke(VERIFY_PROMPT.format(evidence=evidence, claims=claims), json_schema=VERIFY_SCHEMA,
                     reasoning=False, max_tokens=60 * len(batch) + 150)
    # Regex parsing also recovers the complete verdicts of a truncated response
    return {m.group(1): {"status": m.group(2), "issue": m.group(3) or ""} for m in _VERDICT.finditer(str(raw))}


def verify_claims(grounding, items_by_id, llm, max_claims=16, batch_size=6, max_workers=3):
    """Batched LLM check of cited sentences (numeric claims first); updates statuses in place."""
    cited = [s for s in grounding["sentences"] if s["citations"] and s["section"] not in ("Key takeaways", "Evidence table")]
    candidates = sorted(cited, key=lambda s: (not s["numbers"], s["section"] != "Answer"))[:max_claims]
    if not candidates:
        return grounding
    for n, s in enumerate(candidates, start=1):
        s["claim_id"] = f"C{n}"
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    verdicts = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for part in pool.map(lambda b: _verify_batch(b, items_by_id, llm), batches):
            verdicts.update(part)
    for s in candidates:
        v = verdicts.get(s["claim_id"])
        if not v:
            continue
        s["llm_check"] = v
        if v["status"] == "supported" and s["status"] in ("grounded", "cited"):
            s["status"] = "verified"
        elif v["status"] == "partial" and s["status"] != "unsupported_number":
            s["status"] = "partially_supported"
        elif v["status"] == "unsupported":
            s["status"] = "unsupported"
    counts = {}
    for r in grounding["sentences"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    grounding["counts"] = counts
    grounding["llm_verified"] = len(verdicts)
    return grounding


def parse_evidence_table(markdown_section):
    """Rows of the markdown table in the "Evidence table" section (header → value, citations kept)."""
    lines = [ln.strip() for ln in markdown_section.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    cells = lambda ln: [c.strip() for c in ln.strip("|").split("|")]
    header = cells(lines[0])
    rows, seen = [], set()
    for ln in lines[1:]:
        if re.fullmatch(r"[|\s:\-]+", ln):
            continue
        values = cells(ln)
        key = tuple(v.lower() for v in values)
        # a table written twice: its header again and the same rows
        if values == header or key in seen:
            continue
        seen.add(key)
        if len(values) >= len(header) - 1:
            rows.append(dict(zip(header, values + [""] * (len(header) - len(values)))))
    return rows


def answer_summary(sections, max_chars=600):
    """Short summary of an answer (takeaways or the start of "Answer") used as conversation memory."""
    text = sections.get("Key takeaways") or sections.get("Answer") or ""
    return re.sub(r"\[[LTKR]\d+\]", "", text)[:max_chars]


def source_type_of(evidence_id):
    return PREFIX_TO_TYPE.get(evidence_id[:1], "literature")


def to_json(obj):
    return json.dumps(obj, default=str)
