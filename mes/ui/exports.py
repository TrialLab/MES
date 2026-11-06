"""
Answer exports: copy to the clipboard (a small custom component), a standalone HTML report (print it to PDF, or
open it in Word) and the Markdown / JSON downloads built in components.py.
"""

import html
import re

import streamlit as st
from markdown_it import MarkdownIt

from mes.tools.rwd import DATASET
from mes.tools.rwd.report import estimate_text
from mes.ui import components as C
from mes.ui import study_card, study_report

_COPY_JS = """
export default function ({ data, parentElement }) {
  const button = parentElement.querySelector("button");
  if (!button) return;
  const label = button.querySelector("span");
  button.onclick = async () => {
    try {
      await navigator.clipboard.writeText(data.text);
    } catch (e) {                                   // older browsers / no permission: select-and-copy fallback
      const area = document.createElement("textarea");
      area.value = data.text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    label.textContent = "Copied ✓";
    setTimeout(() => { label.textContent = data.label; }, 1600);
  };
  label.textContent = data.label;
}
"""
_COPY_CSS = """
button { width: 100%; height: 40px; border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
  border-radius: 8px; background: transparent; font: inherit; font-size: 0.875rem; color: inherit; cursor: pointer;
  transition: border-color .15s, color .15s; }   /* transparent: right in the light and the dark theme */
button:hover { border-color: #CF4520; color: #CF4520; }   /* WCM Dark Orange */
"""
_copy = st.components.v2.component("mes_copy", html="<button type='button'><span></span></button>", css=_COPY_CSS,
                                   js=_COPY_JS)


def copy_button(text, key, label="⧉ Copy answer"):
    """A button that copies ``text`` to the clipboard (no rerun)."""
    _copy(key=key, data={"text": text, "label": label}, height=40)


def plain_text(result):
    """The answer with numbered citations and the references, as plain text for the clipboard."""
    text = re.sub(r"(?m)^##\s*", "", result.answer_display)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    refs = C.export_markdown(result).split("## References", 1)[-1]
    return f"{result.question}\n\n{text.strip()}\n\nReferences{refs.rstrip()}"


_CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1A1A1A; max-width: 860px;
  margin: 40px auto; padding: 0 24px; line-height: 1.55; font-size: 15px; }
h1 { font-size: 24px; line-height: 1.3; margin: 0 0 6px; }
h2 { font-size: 17px; margin: 26px 0 8px; border-bottom: 1px solid #DDDDDD; padding-bottom: 4px; }
.meta { color: #666666; font-size: 13px; margin-bottom: 18px; }
.takeaways { border: 1px solid #F2CBBE; background: #FBEEE9; border-radius: 10px; padding: 10px 16px; }
sup a { color: #CF4520; text-decoration: none; font-weight: 600; font-size: 11px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { border: 1px solid #DDDDDD; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #F7F7F7; }
ol.refs li { margin: 4px 0; font-size: 13px; }
.rwd { border: 1px solid #99F6E4; border-left: 4px solid #0F766E; border-radius: 10px; padding: 10px 16px; margin: 10px 0; font-size: 13px; }
.protocol { border: 1px solid #F2CBBE; background: #FBEEE9; border-radius: 10px; padding: 8px 14px; margin: 8px 0; font-size: 13px; }
pre.diff { font-size: 11px; background: #F7F7F7; border: 1px solid #DDDDDD; border-radius: 8px; padding: 8px; white-space: pre-wrap; }
.note { color: #666666; font-size: 12px; margin-top: 28px; }
@media print { body { margin: 0; } a { color: inherit; } }
"""


def protocol_lines(protocol):
    """The edited protocol of an answer in plain lines: version, what changed, what ran again, checks, versions."""
    from mes.tools.protocol_tools import names
    from mes.ui.protocol import trail_text
    from mes.ui.protocol import edit_title
    lines = [f"Protocol version {protocol.get('version')}, edited by the user from version "
             f"{protocol.get('from_version', 1)} ({protocol.get('edited_at', '')}): {edit_title(protocol)}."]
    for c in protocol.get("changes") or []:
        before, after = (" · ".join(str(x or "—").splitlines()) for x in (c.get("before"), c.get("after")))
        lines.append(f"{c['what']}: {before[:400]} → {after[:400]}")
    lines.append(f"Ran again: {names(protocol.get('rerun') or []) if protocol.get('rerun') else 'the answer only'}; "
                 f"kept from version {protocol.get('from_version', 1)}: {names(protocol.get('kept') or [])}.")
    for e in (protocol.get("comparison") or {}).get("estimates") or []:
        changed = "; ".join(f"{r['label']}: {r['before'] or '—'} → {r['after'] or '—'}" for r in e.get("changed") or [])
        lines.append(f"Estimates of “{e['study']}”: " + (f"{e.get('n_changed')} changed — {changed}" if changed
                                                       else "unchanged") + ".")
    for t in (protocol.get("comparison") or {}).get("tables") or []:
        lines.append(f"Query result of “{t['study']}”: " + (f"{t['before_rows']} → {t['after_rows']} rows"
                                                            if t.get("changed") else "unchanged") + ".")
    lines += list(protocol.get("checks") or [])
    lines.append(f"Versions: {trail_text(protocol)} ({len(protocol.get('trail') or [])} edit"
                 f"{'s' if len(protocol.get('trail') or []) != 1 else ''}).")
    return lines


def protocol_html(protocol):
    items = "".join(f"<li>{html.escape(line)}</li>" for line in protocol_lines(protocol)[1:])
    diff = f'<pre class="diff">{html.escape(protocol["diff"])}</pre>' if protocol.get("diff") else ""
    return f'<div class="protocol"><b>{html.escape(protocol_lines(protocol)[0])}</b><ul>{items}</ul>{diff}</div>'


def report_html(result):
    """Standalone HTML report: question, answer with numbered citations, real-world data studies, references."""
    md = MarkdownIt("commonmark", {"html": False}).enable("table")    # model text: no raw HTML
    by_id = result.evidence_by_id()
    text = re.sub(r"\[(\d+)\]", r'<sup><a href="#ref-\1">[\1]</a></sup>', result.answer_display)
    sections = re.split(r"(?m)^##\s*", text)
    body = []
    for block in sections:
        if not block.strip():
            continue
        heading, _, content = block.partition("\n")
        rendered = md.render(content.strip()).replace("&lt;sup&gt;", "<sup>").replace("&lt;/sup&gt;", "</sup>")
        rendered = re.sub(r'&lt;a href=&quot;#ref-(\d+)&quot;&gt;\[(\d+)\]&lt;/a&gt;', r'<a href="#ref-\1">[\2]</a>',
                          rendered)
        if heading.strip() == "Key takeaways":
            body.append(f'<h2>{html.escape(heading)}</h2><div class="takeaways">{rendered}</div>')
        else:
            body.append(f"<h2>{html.escape(heading)}</h2>{rendered}")
    studies = []
    for study in result.rwd_studies or []:
        res = study.get("result") or {}
        if not res.get("estimates"):
            continue
        spec = study_card.card(result, study)
        if spec is not None:                 # the study card on paper: every tab, its charts as SVG
            studies.append(f'<div class="rwd">{study_report.card_html(spec, study_report.TAB_NAMES)}</div>')
            continue
        kind = "custom analysis (code written by the MES analyst)" if study.get("kind") == "analyst" else \
            (study.get("spec") or {}).get("analysis", "")
        rows = "".join(f"<li>{html.escape(estimate_text(e))}</li>" for e in res["estimates"][:8])
        studies.append(f'<div class="rwd"><b>[{html.escape(str(study.get("evidence_id", "R")))}] '
                       f'{html.escape(res.get("title") or "")}</b> — {html.escape(study.get("dataset") or DATASET)}, {html.escape(kind)}'
                       + (f", {res['n']:,} patients" if res.get("n") else "") + f"<ul>{rows}</ul></div>")
    if studies:
        body.append("<h2>Real-world data studies</h2>" + "".join(studies))
    if result.protocol:
        body.append("<h2>Protocol</h2>" + protocol_html(result.protocol))
    refs = []
    for n, i in enumerate(result.citation_order, start=1):
        e = by_id.get(i)
        if e is None:
            continue
        title = html.escape(e.title)
        if e.url:
            title = f'<a href="{html.escape(e.url, quote=True)}">{title}</a>'
        detail = ", ".join(html.escape(str(x)) for x in (e.design, e.venue, e.year) if x)
        refs.append(f'<li id="ref-{n}">{title}. {detail}.</li>')
    v = result.verification or {}
    grounding = (f"{v.get('n_numbers_grounded', 0)}/{v.get('n_numbers', 0)} numbers found in the cited sources; "
                 f"{int(round(100 * v.get('citation_coverage', 0)))}% of statements cited.")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(result.question)}</title>
<style>{_CSS}{study_report.CSS if studies else ""}</style></head><body>
<h1>{html.escape(result.question)}</h1>
<div class="meta">MES · {html.escape(result.mode.title())} mode · {html.escape(result.llm)} · {html.escape(result.created_at)} ·
{len(result.evidence)} sources reviewed</div>
{''.join(body)}
<h2>References</h2><ol class="refs">{''.join(refs)}</ol>
<div class="note">Grounding: {grounding} MES is a research prototype for evidence synthesis, not a medical device;
check the cited sources before any clinical decision.</div>
</body></html>"""
