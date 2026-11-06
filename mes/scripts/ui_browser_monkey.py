"""
The MES page in a real browser (headless Chrome) with the fake pipeline of ui_monkey.py: the question box and the
interactions of an impatient user. No language model, no network; about two minutes.

    python -m mes.scripts.ui_browser_monkey [--out DIR]

Scenarios (after each: no exception on the page, no JavaScript error, the page is not blank):
  1. a question typed over two lines, Enter pressed three times quickly: one question runs, the typed one
  2. "New search" while it runs: the landing page comes back with its evidence sources still selected
  3. MIMIC-IV on: the first answer comes with the analysis card, "Skip" keeps it (with a note)
  4. a follow-up with HTML and LaTeX in it, answered; the history opened, a conversation opened and deleted
  5. "Go deeper" then "New search" twice quickly; no evidence source: the send button is disabled
  6. the page reloaded while a question runs
  7. a first answer followed at once by the final one (fast MIMIC-IV analysis), in the light and the dark theme
  8. "Stop" while a question runs, then "Retry"
  9. the agent team builder: drag an agent in (wired automatically), switch its tools off (error, "Use this team"
     disabled), undo / redo, add and delete SafeChecker, a wrong connection refused, an edge removed and drawn again,
     a node moved, zoom, locked settings not editable, save a team, load a template, use it in the question box; the
     team in use shown in the builder; back to the page and a question asked with the team (team state while it
     runs, the team above the sources); light and dark theme; a narrow
     window (nothing wider than the window)
 10. the 🔒 search protocol editor: locked by default; the switch unlocks it (its section stays open); queries
     written by hand re-run as v2 below the edited answer, which is kept as v1; "What changed" opens
 11. the charts of an answer with MIMIC-IV: every tab of the study card, the Table switch of each chart, the
     sensitivity, survival / cumulative incidence and band toggles, the crosshair of the survival curves, the replay of
     the weighting, a dragged range on a histogram; the glance lighting up citations; the map of the evidence and the
     knowledge-graph network — no JavaScript error, nothing that could not be drawn
 12. the cohort builder: a criterion switched off counts the patients again; unlocked, the edited design runs as v2
 13. the patient explorer: an older patient finds older similar patients; unlocked, the edited patient runs as v2
     (12 and 13 in a narrower window too: nothing spills out of a criterion's or characteristic's card)
 14. trial vs RWD: a typed NCT number compared with the study's cohort (ClinicalTrials.gov faked by ui_monkey.py),
     every tab of the comparison drawn; a trial without posted results says so
 15. the app as run.sh starts it (mes/app.py, the real one, no question asked): MES's landing at /, the browser tab and
     the header named MES, the Options menu opens
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).with_name("ui_monkey_app.py")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BOX = ("[...document.querySelectorAll('[data-testid=stBidiComponentIsolated]')].map(e => e.shadowRoot)"
       ".find(r => r && r.querySelector('.mesc-box') && !r.querySelector('.mesc-compact'))")
FOLLOW = ("[...document.querySelectorAll('[data-testid=stBidiComponentIsolated]')].map(e => e.shadowRoot)"
          ".find(r => r && r.querySelector('.mesc-compact'))")
TEAM = ("[...document.querySelectorAll('[data-testid=stBidiComponentIsolated]')].map(e => e.shadowRoot)"
        ".find(r => r && r.querySelector('.tb'))")
# the draft the team canvas keeps in the browser (mes/ui/team_canvas/canvas.js: globalThis.__teamCanvas[key])
TEAM_MEM = "globalThis.__teamCanvas['mes_team_canvas']"
# what the source bar has on: [Literature, ClinicalTrials.gov, iBKH graph, Real-world data] — the two menus are on
# when their box holds the mark of a source that is on ([] while the bar is not shown)
PILLS = ("(() => { const pressed = (t) => [...document.querySelectorAll('.st-key-mes_src_mid button')]"
         ".some(b => b.innerText.includes(t) && b.getAttribute('aria-pressed') === 'true');"
         " return document.querySelector('.st-key-mes_sources_bar') ? ["
         "!!document.querySelector('.st-key-mes_src_litbox .mes-src-on'), pressed('ClinicalTrials'), pressed('iBKH'), "
         "!!document.querySelector('.st-key-mes_src_rwdbox .mes-src-on')] : []; })()")


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(fixtures, delay, extra=None, app=APP):
    """The fake-pipeline app (or ``app``) on a free port; started again on another port when it does not come up (a
    port that is free on 127.0.0.1 can be taken on another interface)."""
    env = {**os.environ, "MES_MONKEY_REAL_COMPOSER": "1", "MES_MONKEY_DELAY": str(delay), "MES_MONKEY_BEHAVIOUR": "ok",
           "MES_MONKEY_CACHE": tempfile.mkdtemp(prefix="mes_monkey_"),
           "MES_HISTORY_DIR": tempfile.mkdtemp(prefix="mes_monkey_history_"),
           "MES_TEAMS_DIR": tempfile.mkdtemp(prefix="mes_monkey_teams_"), **(extra or {})}
    if fixtures:
        env["MES_MONKEY_FIXTURES"] = str(fixtures)
    log = Path(tempfile.mkdtemp(prefix="mes_monkey_app_")) / "streamlit.log"
    for _attempt in range(3):
        port = _port()
        with open(log, "ab") as out:
            proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true",
                                     "--server.port", str(port), "--server.fileWatcherType", "none",
                                     "--browser.gatherUsageStats", "false"],   # its fetch fails offline (JS error)
                                    cwd=str(ROOT), env=env, stdout=out,
                                    stderr=subprocess.STDOUT)
        for _ in range(120):
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1)
                return proc, port
            except Exception:
                time.sleep(0.5)
        proc.terminate()
    raise RuntimeError(f"the test app did not start; see {log}")


class Page:
    """One headless Chrome tab over the DevTools protocol."""

    def __init__(self, shots, dark=False):
        self.port, self.shots, self.dark = _port(), shots, dark
        self.profile = tempfile.mkdtemp(prefix="mes_chrome_")
        self.chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                        f"--remote-debugging-port={self.port}", f"--user-data-dir={self.profile}",
                                        "--window-size=1440,1000", "about:blank"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws, self.n, self.errors = None, 0, []

    async def open(self, url):
        import websockets
        for _ in range(100):
            try:
                target = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list"))
                              if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(0.2)
        self.ws = await websockets.connect(target["webSocketDebuggerUrl"], max_size=200_000_000)
        await self.call("Runtime.enable")
        if self.dark:
            await self.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
        await self.goto(url)

    async def goto(self, url):
        await self.call("Page.navigate", url=url)
        await self.wait(f"!!{BOX}", 90)
        await asyncio.sleep(1.5)

    async def call(self, method, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                return msg.get("result", {})
            if msg.get("method") == "Runtime.exceptionThrown":
                self.errors.append(msg["params"]["exceptionDetails"].get("exception", {}).get("description", "")[:300])
            elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"].get("type") == "error":
                text = " ".join(str(a.get("description") or a.get("value") or "") for a in msg["params"].get("args", []))
                if "Bad delta" in text or "Uncaught" in text:
                    self.errors.append(text[:300])

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def wait(self, expr, timeout=60):
        t = time.time()
        while time.time() - t < timeout:
            try:
                if await self.js(expr):
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.4)
        return False

    async def click(self, text, selector="button"):
        box = await self.js(f"""(() => {{ const el = [...document.querySelectorAll('{selector}')]
            .find(e => e.innerText && e.innerText.includes({json.dumps(text)}) && e.offsetParent !== null);
            if (!el) return null; el.scrollIntoView({{block: 'center'}}); const r = el.getBoundingClientRect();
            return [r.x + r.width / 2, r.y + r.height / 2]; }})()""")
        if not box:
            return False
        await asyncio.sleep(0.3)
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=kind, x=box[0], y=box[1], button="left", clickCount=1)
        return True

    async def key(self, key="Enter", code="Enter", vk=13, modifiers=0):
        for kind in ("keyDown", "keyUp"):
            await self.call("Input.dispatchKeyEvent", type=kind, key=key, code=code, windowsVirtualKeyCode=vk,
                            modifiers=modifiers)

    async def type_in(self, root, text):
        await self.js(f"(() => {{ const r = {root}; if (r) r.querySelector('textarea').focus(); }})()")
        await self.call("Input.insertText", text=text)

    async def shot(self, name):
        data = await self.call("Page.captureScreenshot", format="png")
        (self.shots / f"monkey-{name}.png").write_bytes(base64.b64decode(data["data"]))

    async def state(self):
        """(exceptions on the page, blank page)"""
        exceptions = await self.js("[...document.querySelectorAll('[data-testid=stException]')]"
                                   ".map(e => e.innerText.slice(0, 300))") or []
        blank = (await self.js("document.body.innerText.trim().length") or 0) < 20
        return exceptions, blank

    def close(self):
        self.chrome.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


async def scenarios(page, port, results):
    async def check(name, ok=True, detail=""):
        exceptions, blank = await page.state()
        errors, page.errors = page.errors, []
        passed = ok and not exceptions and not blank and not errors
        results.append({"scenario": name, "ok": passed, "detail": detail, "exceptions": exceptions, "blank": blank,
                        "js_errors": errors})
        if not passed:
            await page.shot(name.split(":")[0].replace(" ", "-"))
        print(f"{'PASS' if passed else 'FAIL'} {name} {detail}", flush=True)

    questions = "[...document.querySelectorAll('.mes-question')].map(e => e.innerText)"
    # 1. two lines, Enter three times
    await page.type_in(BOX, "First line of the question")
    await page.key("Enter", "Enter", 13, modifiers=8)
    await page.call("Input.insertText", text=" — second line: does aspirin reduce stroke?")
    for _ in range(3):
        await page.key()
    await page.wait(f"{questions}.length > 0", 20)
    await asyncio.sleep(1.5)
    shown = await page.js(questions) or []
    await check("1 triple Enter: one question, the typed one", len(shown) == 1 and shown[0].startswith("First line"),
                str(shown)[:120])
    # 2. New search during the run
    await page.click("New search")
    await page.wait(f"!!{BOX} && ({PILLS}).length >= 4", 20)
    await asyncio.sleep(1)
    pills = await page.js(PILLS) or []
    await check("2 New search during a run: sources kept", pills[:3] == [True, True, True], str(pills))
    # 3. MIMIC-IV, first answer, Skip
    await _rwd_on(page)
    await asyncio.sleep(1.5)
    await page.type_in(BOX, "Is hydrocortisone associated with 28-day mortality in septic shock?")
    await page.key()
    live = await page.wait("document.body.innerText.includes('Real-world data analysis in progress')", 60)
    await page.click("Skip")
    await asyncio.sleep(3)
    note = await page.js("document.body.innerText.includes('skipped')")
    await check("3 first answer with the MIMIC-IV card, Skip", bool(live and note), f"card {live}, note {note}")
    # 4. follow-up with markup, history
    await page.type_in(FOLLOW, "What about in children <img src=x onerror=alert(1)> $$x$$?")
    await page.key()
    answered = await page.wait(f"{questions}.length >= 2 && document.body.innerText.includes('Grounding:') "
                               "&& !document.body.innerText.includes('analysis in progress')", 90)
    raw = await page.js(f"{questions}.some(q => q.includes('<img src=x'))")
    await check("4a follow-up with HTML and LaTeX shown as typed", bool(answered and raw))
    await page.click("History")
    await asyncio.sleep(1)
    await page.click("", '[class*="st-key-mes_hist_thread_"] button')          # a saved conversation
    await asyncio.sleep(2.5)
    await page.click("History")
    await asyncio.sleep(1)
    await page.click("✕")
    await asyncio.sleep(2)
    await check("4b history: open and delete a conversation")
    # 5. deeper + New search twice; no sources
    await page.key("Escape", "Escape", 27)
    await page.click("Balanced")
    await asyncio.sleep(1)
    await page.click("New search")
    await asyncio.sleep(0.4)
    await page.click("New search")
    await asyncio.sleep(2.5)
    await check("5a Go deeper, then New search twice")
    for label in ("PubMed", "Your PDF library"):             # literature off: its parts off (full text follows PubMed)
        if (await page.js(PILLS) or [False])[0]:
            await _menu_check(page, "Literature", label)
    for i, name in ((1, "ClinicalTrials.gov"), (2, "iBKH graph")):
        if (await page.js(PILLS) or [False] * 4)[i]:
            await page.click(name)
            await page.wait(f"!({PILLS})[{i}]", 10)
            await asyncio.sleep(0.5)
    if (await page.js(PILLS) or [False] * 4)[3]:
        await _menu_check(page, "Real-world data", "MIMIC-IV")
    disabled = await page.js(f"{BOX}.querySelector('.mesc-send').disabled")
    await page.type_in(BOX, "Does it work?")
    await page.key()
    await asyncio.sleep(2)
    started = await page.js(questions) or []
    await check("5b no evidence source: sending is disabled", bool(disabled) and not started,
                f"disabled {disabled}, started {started}")
    # 6. reload during a run
    await _menu_check(page, "Literature", "PubMed")
    await asyncio.sleep(1.5)
    await page.type_in(BOX, "Does metformin reduce cardiovascular death?")
    await page.key()
    await asyncio.sleep(1.5)
    await page.goto(f"http://127.0.0.1:{port}/")
    await check("6 reload while a question runs")


async def stop(page, results):
    """8. Stop while a question runs (a slow pipeline), then Retry."""
    await page.type_in(BOX, "Does colchicine reduce cardiovascular events after myocardial infarction?")
    await page.key()
    await page.wait("!!document.querySelector(\"[class*='st-key-stop_'] button\")", 20)
    stopped = await page.click("Stop", '[class*="st-key-stop_"] button')        # MES's Stop, not Streamlit's toolbar
    shown = await page.wait("document.body.innerText.includes('Stopped')", 15)
    await page.click("Retry")
    answered = await page.wait("document.body.innerText.includes('Grounding:')", 90)
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    ok = bool(stopped and shown and answered) and not exceptions and not blank and not errors
    results.append({"scenario": "8 Stop while it runs, then Retry", "ok": ok, "exceptions": exceptions,
                    "blank": blank, "js_errors": errors})
    if not ok:
        await page.shot("8-stop")
    print(f"{'PASS' if ok else 'FAIL'} 8 Stop while it runs, then Retry (stop {stopped}, note {shown}, answered "
          f"{answered})", flush=True)


async def fast(page, theme, results):
    """A first answer followed at once by the final one (the case that once left a blank page)."""
    await _rwd_on(page)
    await asyncio.sleep(1.5)
    await page.type_in(BOX, "Is hydrocortisone associated with 28-day mortality in septic shock?")
    await page.key()
    done = await page.wait("document.body.innerText.includes('Grounding:') && "
                           "!document.body.innerText.includes('analysis in progress')", 60)
    await asyncio.sleep(2)
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    ok = done and not exceptions and not blank and not errors
    results.append({"scenario": f"7 fast first and final answer ({theme} theme)", "ok": ok, "exceptions": exceptions,
                    "blank": blank, "js_errors": errors})
    if not ok:
        await page.shot(f"7-{theme}")
    print(f"{'PASS' if ok else 'FAIL'} 7 fast first and final answer ({theme} theme)", flush=True)


SWITCH = "[class*='st-key-mes_lock_'][class*='_search']:not([class*='_go'])"
EDIT_BUTTON = "[class*='st-key-mes_lock_'][class*='_search_go'] button"


async def protocol(page, results):
    """10. The 🔒 search protocol editor: unlock, write the PubMed query by hand, re-run as v2."""
    async def press(selector):
        box = await page.js(f"""(() => {{ const e = document.querySelector({json.dumps(selector)});
            if (!e || e.offsetParent === null) return null; e.scrollIntoView({{block: 'center'}});
            const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()""")
        if not box:
            return False
        await asyncio.sleep(0.3)
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await page.call("Input.dispatchMouseEvent", type=kind, x=box[0], y=box[1], button="left", clickCount=1)
        return True

    await page.type_in(BOX, "Is apixaban superior to warfarin for stroke prevention in atrial fibrillation?")
    await page.key()
    answered = await page.wait("document.body.innerText.includes('Grounding:')", 60)
    await asyncio.sleep(1.5)
    await page.click("Search protocol", "summary")
    await asyncio.sleep(1.0)
    locked = await page.js(f"(() => {{ const b = document.querySelector({json.dumps(EDIT_BUTTON)}); "
                           f"return !!b && b.disabled; }})()")
    await press(f"{SWITCH} label")
    unlocked = await page.wait(f"(() => {{ const b = document.querySelector({json.dumps(EDIT_BUTTON)}); "
                               f"return !!b && !b.disabled && b.offsetParent !== null; }})()", 15)
    await press(EDIT_BUTTON)
    opened = await page.wait("!!document.querySelector('[role=dialog]')", 15)
    await asyncio.sleep(1.0)
    await press("[role=dialog] [class*='st-key-mes_proto_'][class*='_hand'] label")
    await page.wait("!!document.querySelector(\"[role=dialog] [class*='_pubmed'] textarea\")", 15)
    await page.js("(() => { const t = document.querySelector(\"[role=dialog] [class*='_pubmed'] textarea\"); t.focus(); "
                  "t.select(); })()")
    await page.call("Input.insertText", text="(apixaban) AND (warfarin) AND (atrial fibrillation) AND randomized[ti]")
    await page.js("document.querySelector(\"[role=dialog] [class*='_pubmed'] textarea\").blur()")
    await asyncio.sleep(1.5)
    await page.click("Re-run with this protocol")
    v2 = await page.wait("document.body.innerText.includes('Protocol v2 · edited by you from v1')", 60)
    await asyncio.sleep(1.5)
    v1 = await page.js("[...document.querySelectorAll('summary')].some(e => e.innerText.includes('protocol v1'))")
    changed = await page.click("What changed in v2", "summary")
    await asyncio.sleep(1.0)
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    ok = all((answered, locked, unlocked, opened, v2, v1, changed)) and not exceptions and not blank and not errors
    name = "10 the 🔒 search protocol editor: unlock, edit by hand, re-run as v2"
    results.append({"scenario": name, "ok": ok, "exceptions": exceptions, "blank": blank, "js_errors": errors,
                    "detail": f"locked {locked}, unlocked {unlocked}, dialog {opened}, v2 {v2}, v1 kept {v1}"})
    if not ok:
        await page.shot("10-protocol")
    print(f"{'PASS' if ok else 'FAIL'} {name} (locked {locked}, unlocked {unlocked}, dialog {opened}, v2 {v2}, "
          f"v1 kept {v1})", flush=True)


CARDS = ("[...document.querySelectorAll('[data-testid=stBidiComponentIsolated]')].map(e => e.shadowRoot)"
         ".filter(r => r && r.querySelector('.mv'))")


async def charts(page, results):
    """11. The interactive charts of an answer with a MIMIC-IV study (see the module docstring)."""
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:200]))

    await _rwd_on(page)
    await asyncio.sleep(1.2)
    await page.type_in(BOX, "In septic shock, is early hydrocortisone associated with 28-day mortality?")
    await page.key()
    done = await page.wait("document.body.innerText.includes('Grounding:') && "
                           "!document.body.innerText.includes('analysis in progress')", 90)
    await asyncio.sleep(2.5)
    card = await page.wait(f"{CARDS}.some(r => r.querySelector('.mv-study'))", 20)
    check("the answer and its study card are shown", done and card)
    study = f"{CARDS}.find(r => r.querySelector('.mv-study'))"
    tabs = await page.js(f"[...{study}.querySelectorAll('.mv-tab')].length") or 0
    for i in range(tabs):
        await page.js(f"{study}.querySelectorAll('.mv-tab')[{i}].click()")
        await asyncio.sleep(0.8)
        # every Chart / Table switch and toggle of the tab, twice (on and back)
        n = await page.js(f"[...{study}.querySelectorAll('.mv-chip-btn, .mv-seg button')].length") or 0
        for j in range(min(n, 12)):
            await page.js(f"(() => {{ const b = [...{study}.querySelectorAll('.mv-chip-btn, .mv-seg button')][{j}]; "
                          "if (b) b.click(); })()")
            await asyncio.sleep(0.25)
        # pointer over every plot surface (crosshairs, tooltips) and a drag across one (the range of a histogram)
        boxes = await page.js(f"""[...{study}.querySelectorAll('.mv-hit')].map(e => {{ e.scrollIntoView({{block: 'center'}});
            const r = e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }})""") or []
        for x, y, w, h_ in boxes[:6]:
            for k in range(1, 5):
                await page.call("Input.dispatchMouseEvent", type="mouseMoved", x=x + w * k / 5, y=y + h_ / 2)
            await page.call("Input.dispatchMouseEvent", type="mousePressed", x=x + w * 0.2, y=y + h_ / 2, button="left",
                            clickCount=1)
            await page.call("Input.dispatchMouseEvent", type="mouseMoved", x=x + w * 0.6, y=y + h_ / 2, buttons=1)
            await page.call("Input.dispatchMouseEvent", type="mouseReleased", x=x + w * 0.6, y=y + h_ / 2, button="left",
                            clickCount=1)
        await asyncio.sleep(0.3)
    errors = await page.js(f"{CARDS}.reduce((n, r) => n + r.querySelectorAll('.mv-error').length, 0)")
    check("every tab, switch and chart of the study card works", tabs >= 3 and errors == 0, f"{tabs} tabs, {errors} errors")
    # the glance: hovering a part of the evidence lights up its citations in the answer
    lit = await page.js(f"""(() => {{ const g = {CARDS}.find(r => r.querySelector('.mv-glance')); if (!g) return null;
        const b = g.querySelector('.mv-mixleg button, .mv-pyr-row'); if (!b) return null;
        b.dispatchEvent(new PointerEvent('pointerenter', {{bubbles: true}}));
        const n = document.querySelectorAll('a.mes-cite.mes-hl').length;
        b.dispatchEvent(new PointerEvent('pointerleave', {{bubbles: true}})); return n; }})()""")
    check("the glance lights up the citations it counts", lit, lit)
    # the map of the evidence in the sources rail
    if await page.click("Map", "[role=tab]"):
        await asyncio.sleep(1.2)
        bubbles = await page.js(f"""(() => {{ const m = {CARDS}.find(r => r.querySelector('.mv-bubble')); if (!m) return 0;
            const b = [...m.querySelectorAll('.mv-bubble')]; b.forEach(x => x.dispatchEvent(new MouseEvent('click', {{bubbles: true}})));
            return b.length; }})()""")
        check("the map of the evidence draws and answers clicks", bubbles, bubbles)
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    failed = [c for c in checks if not c[1]]
    ok = not failed and not exceptions and not blank and not errors
    results.append({"scenario": "11 the charts of an answer with MIMIC-IV", "ok": ok, "exceptions": exceptions,
                    "blank": blank, "js_errors": errors, "detail": "; ".join(f"{n}: {d}" for n, _, d in failed)})
    if not ok:
        await page.shot("11-charts")
    print(f"{'PASS' if ok else 'FAIL'} 11 the charts of an answer with MIMIC-IV: {len(checks) - len(failed)}/{len(checks)} "
          f"checks" + "".join(f"\n     FAIL {n} {d}" for n, _, d in failed) + (f"\n     JS {errors[:2]}" if errors else ""),
          flush=True)


def fixture_dir(fixtures, analysis):
    """A folder with one stored answer whose MIMIC-IV study is of ``analysis`` (the fake pipeline answers with it),
    or None when the fixtures have none."""
    for path in sorted(Path(fixtures or "").glob("*.json")) if fixtures else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(isinstance(s, dict) and s.get("status") == "done" and s.get("kind") != "analyst"
               and (s.get("spec") or {}).get("analysis") == analysis for s in data.get("rwd_studies") or []):
            folder = Path(tempfile.mkdtemp(prefix=f"mes_fx_{analysis}_"))
            shutil.copy(path, folder / path.name)
            return folder
    return None


async def _press(page, selector):
    box = await page.js(f"""(() => {{ const e = document.querySelector({json.dumps(selector)});
        if (!e || e.offsetParent === null) return null; e.scrollIntoView({{block: 'center'}});
        const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()""")
    if not box:
        return False
    await asyncio.sleep(0.3)
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        await page.call("Input.dispatchMouseEvent", type=kind, x=box[0], y=box[1], button="left", clickCount=1)
    return True


async def _menu_check(page, menu, label):
    """In the source bar's menu ``menu`` (Literature / Real-world data) switch the box named ``label``; the menu is
    closed again. False when the box is not there."""
    await page.click(menu, "[data-testid=stPopoverButton]")
    await page.wait("!!document.querySelector('[data-testid=stPopoverBody] .mes-src-menu')", 10)
    await asyncio.sleep(0.5)
    box = await page.js(f"""(() => {{ const l = [...document.querySelectorAll('[data-testid=stPopoverBody] label')]
        .find(e => e.innerText.trim().startsWith({json.dumps(label)})); if (!l) return null;
        const r = l.getBoundingClientRect(); return [r.x + 10, r.y + r.height / 2]; }})()""")
    if box:
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await page.call("Input.dispatchMouseEvent", type=kind, x=box[0], y=box[1], button="left", clickCount=1)
        await asyncio.sleep(1.2)
    await page.key("Escape", "Escape", 27)
    await asyncio.sleep(0.6)
    return bool(box)


async def _rwd_on(page):
    """Real-world data on (MIMIC-IV, in its menu on the source bar)."""
    if not (await page.js(PILLS) or [False] * 4)[3]:
        await _menu_check(page, "Real-world data", "MIMIC-IV")


async def _ask_mimic(page, question):
    await _rwd_on(page)
    await asyncio.sleep(1.2)
    await page.type_in(BOX, question)
    await page.key()
    done = await page.wait("document.body.innerText.includes('Grounding:') && "
                           "!document.body.innerText.includes('analysis in progress')", 90)
    await asyncio.sleep(2.5)
    return done


# what spills out of a criterion's or characteristic's card (a text through its border, a control cut off at its edge),
# and a dashed box drawn around the Add button (the add row's own style matched the button's key once)
SPILL = r"""(() => {
  const out = [];
  const hidden = (e, c) => {          // a label kept for screen readers only (clipped to a pixel)
    for (let a = e; a && a !== c; a = a.parentElement) if (getComputedStyle(a).clipPath !== 'none') return true;
    return false; };
  for (const c of document.querySelectorAll("[class*='st-key-PREFIX'][class*='_card_'], [class*='st-key-PREFIX'][class*='_feat_']")) {
    const r = c.getBoundingClientRect();
    for (const e of c.querySelectorAll('p, .mes-cb-name, .mes-cb-meta, input, button, [data-baseweb="select"]')) {
      const b = e.getBoundingClientRect();
      if (b.width > 2 && b.height > 2 && !hidden(e, c)
          && (b.bottom > r.bottom + 1 || b.right > r.right + 1 || b.left < r.left - 1))
        out.push((e.innerText || e.value || e.tagName).slice(0, 40));
    }
  }
  for (const e of document.querySelectorAll("[class*='st-key-PREFIX'][class*='_addgo']"))
    if (getComputedStyle(e).borderStyle.includes('dashed')) out.push('a dashed box around Add');
  return out;
})()"""


async def rwd_tool_scenario(page, results, tool):
    """12. the cohort builder: open it, switch a criterion off (the flow of patients is counted again), unlock, run
    the edited design as v2. 13. the patient explorer: open it, change the patient's age (the similar patients are
    found again), unlock, run as v2."""
    checks = []
    prefix = {"builder": "mes_cb_", "explorer": "mes_px_"}[tool]

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:200]))

    asked = await _ask_mimic(page, "In septic shock, is hydrocortisone associated with 28-day mortality?" if tool ==
                             "builder" else "A 68-year-old man with septic shock: how did similar patients do?")
    check("the answer and its study card", asked)
    opened = await _press(page, f"[class*='st-key-{prefix}'][class*='_toggle'] button")
    shown = await page.wait(f"!!document.querySelector(\"[class*='st-key-{prefix}'][class*='_box']\")", 30)
    check("the panel opens", opened and shown)
    await asyncio.sleep(1.5)
    for width in (1180, 1440):                     # a narrower window too, then back
        await page.call("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=1, mobile=False)
        await asyncio.sleep(1.5)
        spilled = await page.js(SPILL.replace("PREFIX", prefix))
        check(f"nothing spills out of a card at {width} px, the Add button has no box of its own", not spilled,
              spilled[:3] if spilled else "")
    if tool == "builder":
        drawn = await page.wait(f"{CARDS}.some(r => r.querySelector('.mv-cf-n'))", 20)
        await asyncio.sleep(1.5)
        before = await page.js(f"({CARDS}.find(r => r.querySelector('.mv-cf-n')) || {{querySelector: () => null}})"
                               ".querySelector('.mv-cf-n')?.textContent")
        check("the flow of patients is drawn", drawn, before)
        flipped = await _press(page, f"[class*='st-key-{prefix}'][class*='_on'] label")
        changed = await page.wait(f"(() => {{ const r = {CARDS}.find(x => x.querySelector('.mv-cf-n')); "
                                  f"return r && r.querySelector('.mv-cf-n').textContent !== {json.dumps(before)} "
                                  "&& !!r.querySelector('.mv-cf-off, .mv-cf-changed, .mv-cf-new'); })()", 20)
        check("a criterion switched off counts the patients again", flipped and changed)
    else:
        preview = f"{CARDS}.find(r => r.textContent.includes('not yet in the answer'))"
        await page.wait(f"!!{preview}", 25)
        await page.js(f"(() => {{ const r = {preview}; const t = r && [...r.querySelectorAll('.mv-tab')]"
                      ".find(b => b.innerText.startsWith('Profile')); if (t) t.click(); })()")      # the Profile tab
        drawn = await page.wait(f"(() => {{ const r = {preview}; return !!r && !!r.querySelector('.mv-pf-row'); }})()", 15)
        await asyncio.sleep(1.0)
        age = ("(() => { const r = " + CARDS + ".find(x => x.textContent.includes('not yet in the answer'));"
               " if (!r) return null; const row = [...r.querySelectorAll('.mv-pf-row')].find(e => e.innerText.startsWith('Age'));"
               " return row ? row.innerText : null; })()")
        before = await page.js(age)
        check("the similar patients are drawn", drawn and before, before)
        field = f"[class*='st-key-{prefix}'][class*='_p0_value'] input"
        await page.js(f"(() => {{ const e = document.querySelector({json.dumps(field)}); if (e) {{ e.focus(); e.select(); }} }})()")
        await page.call("Input.insertText", text="85")
        await page.key("Enter", "Enter", 13)
        changed = await page.wait(f"(() => {{ const t = {age}; return t && t !== {json.dumps(before)}; }})()", 25)
        check("an older patient finds older similar patients", changed, await page.js(age))
    await _press(page, f"[class*='st-key-{prefix}'][class*='_lock'] label")
    ready = await page.wait(f"(() => {{ const b = document.querySelector(\"[class*='st-key-{prefix}'][class*='_run'] button\");"
                            " return !!b && !b.disabled; })()", 15)
    check("unlocked, the run button is enabled", ready)
    await _press(page, f"[class*='st-key-{prefix}'][class*='_run'] button")
    v2 = await page.wait("document.body.innerText.includes('Protocol v2 · edited by you from v1')", 60)
    check("the edited study answers again as v2", v2)
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    failed = [c for c in checks if not c[1]]
    ok = not failed and not exceptions and not blank and not errors
    name = {"builder": "12 the cohort builder", "explorer": "13 the patient explorer"}[tool]
    results.append({"scenario": name, "ok": ok, "exceptions": exceptions, "blank": blank, "js_errors": errors,
                    "detail": "; ".join(f"{n}: {d}" for n, _, d in failed)})
    if not ok:
        await page.shot(name.split(" ")[0] + "-" + tool)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {len(checks) - len(failed)}/{len(checks)} checks"
          + "".join(f"\n     FAIL {n} {d}" for n, _, d in failed) + (f"\n     JS {errors[:2]}" if errors else ""),
          flush=True)


async def trial_scenario(page, results):
    """14. trial vs RWD (see the module docstring)."""
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:200]))

    asked = await _ask_mimic(page, "In septic shock, is hydrocortisone associated with 28-day mortality?")
    check("the answer and its study card", asked)
    opened = await _press(page, "[class*='st-key-mes_tp_'][class*='_toggle'] button")
    shown = await page.wait("!!document.querySelector(\"[class*='st-key-mes_tp_'][class*='_box']\")", 30)
    check("the panel opens", opened and shown)
    field = "[class*='st-key-mes_tp_'][class*='_nct'] input"
    if not await page.js(f"!!document.querySelector({json.dumps(field)})"):      # the answer's trials are offered
        await _press(page, "[class*='st-key-mes_tp_'][class*='_pick'] input")       # the select's combobox
        await asyncio.sleep(0.6)
        await page.click("Another trial", "[role=option]")
    typed = await page.wait(f"!!document.querySelector({json.dumps(field)})", 15)
    check("an NCT number can be typed", typed)
    card = f"{CARDS}.find(r => r.textContent.includes('Trial vs real-world population'))"
    for nct, expect in (("NCT00000001", "card"), ("NCT00000010", "no baseline characteristics")):
        await page.js(f"(() => {{ const e = document.querySelector({json.dumps(field)}); if (e) {{ e.focus(); e.select(); }} }})()")
        await page.call("Input.insertText", text=nct)
        await page.key("Enter", "Enter", 13)
        ready = await page.wait("(() => { const b = document.querySelector(\"[class*='st-key-mes_tp_'][class*='_go'] button\");"
                                " return !!b && !b.disabled; })()", 15)
        await _press(page, "[class*='st-key-mes_tp_'][class*='_go'] button")
        if expect == "card":
            drawn = await page.wait(f"!!{card} && !!{card}.querySelector('.mv-tab')", 40)
            tabs = await page.js(f"[...({card} || document).querySelectorAll('.mv-tab')].length") or 0
            for i in range(tabs):
                await page.js(f"{card}.querySelectorAll('.mv-tab')[{i}].click()")
                await asyncio.sleep(0.6)
            errors = await page.js(f"({card} || document).querySelectorAll('.mv-error').length")
            check("the comparison is drawn, every tab", ready and drawn and tabs >= 2 and not errors,
                  f"ready {ready}, drawn {drawn}, {tabs} tabs, {errors} errors")
        else:
            said = await page.wait(f"document.body.innerText.includes({json.dumps(expect)})", 30)
            check("a trial without posted results says so", ready and said)
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    failed = [c for c in checks if not c[1]]
    ok = not failed and not exceptions and not blank and not errors
    results.append({"scenario": "14 trial vs RWD", "ok": ok, "exceptions": exceptions, "blank": blank,
                    "js_errors": errors, "detail": "; ".join(f"{n}: {d}" for n, _, d in failed)})
    if not ok:
        await page.shot("14-trial")
    print(f"{'PASS' if ok else 'FAIL'} 14 trial vs RWD: {len(checks) - len(failed)}/{len(checks)} checks"
          + "".join(f"\n     FAIL {n} {d}" for n, _, d in failed) + (f"\n     JS {errors[:2]}" if errors else ""),
          flush=True)


class Builder:
    """Pointer and keyboard on the team builder canvas (inside its shadow root)."""

    def __init__(self, page):
        self.page = page

    async def center(self, selector):
        return await self.page.js(f"""(() => {{ const e = {TEAM} && {TEAM}.querySelector({json.dumps(selector)});
            if (!e) return null; e.scrollIntoView({{block: 'nearest'}}); const r = e.getBoundingClientRect();
            return [r.x + r.width / 2, r.y + r.height / 2]; }})()""")

    async def mouse(self, kind, x, y, buttons=1, modifiers=0):
        await self.page.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1,
                             buttons=0 if kind == "mouseReleased" else buttons, modifiers=modifiers)

    async def drag(self, a, b, steps=12):
        await self.mouse("mouseMoved", *a, buttons=0)
        await self.mouse("mousePressed", *a)
        for i in range(1, steps + 1):
            await self.mouse("mouseMoved", a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)
            await asyncio.sleep(0.02)
        await self.mouse("mouseReleased", *b)
        await asyncio.sleep(0.4)

    async def tap(self, selector):
        c = await self.center(selector)
        if not c:
            return False
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.mouse(kind, *c)
        await asyncio.sleep(0.4)
        return True

    async def edges(self):
        return set(await self.page.js(f"{TEAM_MEM}.graph.edges.map(e => e.from + '>' + e.to)") or [])

    async def text(self, selector):
        return await self.page.js(f"({TEAM}.querySelector({json.dumps(selector)}) || {{}}).innerText || ''")


async def builder(page, port, results, theme):
    """9. The agent team builder in the browser (see the module docstring)."""
    b = Builder(page)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:200]))

    await page.click("Agent team builder")
    opened = await page.wait(f"!!{TEAM} && {TEAM}.querySelectorAll('.tb-node').length >= 8", 30)
    await asyncio.sleep(1)
    check("opens with the default team", opened and await b.text(".tb-badge"), await b.text(".tb-badge"))
    # the default team has every evidence agent: take the RWD Analyst out with the key, then drag it back in
    await b.tap('.tb-node[data-id="rwd"] .tb-node-title')
    await page.key("Backspace", "Backspace", 8)
    await asyncio.sleep(0.4)
    check("an agent deleted with the key", not await page.js(f"!!{TEAM}.querySelector('.tb-node[data-id=rwd]')"))
    canvas = await b.center(".tb-canvas")
    await b.drag(await b.center('.tb-pal-item[data-type="rwd"]'), (canvas[0] - 260, canvas[1] - 60))
    e = await b.edges()
    check("an agent dragged in is wired", {"planner>rwd", "rwd>synthesizer"} <= e, sorted(e))
    flagged = await page.js(f"{TEAM}.querySelector('.tb-node[data-id=rwd]').classList.contains('tb-warn')")
    unbuilt = "not built" in await b.text(".tb-inspector")
    check("how this computer runs MIMIC-IV is a note, not a warning", not flagged or unbuilt, (flagged, unbuilt))
    for tool in ("templates", "analyst"):
        await page.js(f"{TEAM}.querySelector('[data-tool={tool}]').click()")
        await asyncio.sleep(0.3)
    blocked = await page.js(f"{TEAM}.querySelector('[data-act=apply]').disabled")
    check("no tool left: an error, Use this team disabled", blocked and "problem" in await b.text(".tb-badge"))
    locked = await page.js(f"[...{TEAM}.querySelectorAll('.tb-inspector input, .tb-inspector select')]"
                           ".filter(i => !i.dataset.tool && !i.disabled).length")
    check("locked settings cannot be edited", locked == 0, locked)
    await page.js(f"{TEAM}.querySelector('.tb-canvas').focus()")
    for _ in range(2):
        await page.key("z", "KeyZ", 90, modifiers=4)
        await asyncio.sleep(0.25)
    tools = await page.js(f"{TEAM_MEM}.graph.nodes.find(n => n.type === 'rwd').tools")
    await page.key("z", "KeyZ", 90, modifiers=12)
    await asyncio.sleep(0.25)
    redo = await page.js(f"{TEAM_MEM}.graph.nodes.find(n => n.type === 'rwd').tools")
    await page.key("z", "KeyZ", 90, modifiers=4)
    check("undo / redo", tools == {"templates": True, "analyst": True} and redo == {"templates": False, "analyst": True},
          (tools, redo))
    await b.tap('.tb-pal-item[data-type="safechecker"] .tb-add')
    e1 = await b.edges()
    await b.tap('.tb-node[data-id="safechecker"] .tb-node-title')
    await page.key("Backspace", "Backspace", 8)
    await asyncio.sleep(0.4)
    e2 = await b.edges()
    check("SafeChecker added before the Answer, deleted with the key",
          {"grounding>safechecker", "safechecker>answer"} <= e1 and "grounding>answer" in e2
          and not any("safechecker" in x for x in e2), (sorted(e1), sorted(e2)))
    before = await b.edges()
    await b.drag(await b.center('.tb-node[data-id="planner"] .tb-out'),
                 await b.center('.tb-node[data-id="answer"] .tb-node-title'))
    check("a wrong connection is refused with the reason", await b.edges() == before and "needs" in await b.text(".tb-toast"),
          await b.text(".tb-toast"))
    await page.js(f"""[...{TEAM}.querySelectorAll('.tb-edge-hit')].find(x => x.dataset.from === 'kg')
        .dispatchEvent(new PointerEvent('pointerdown', {{bubbles: true, composed: true, button: 0}}))""")
    await asyncio.sleep(0.3)
    await b.tap(".tb-edge-del")
    removed = "kg>synthesizer" not in await b.edges()
    await b.drag(await b.center('.tb-node[data-id="kg"] .tb-out'),
                 await b.center('.tb-node[data-id="synthesizer"] .tb-node-title'))
    check("an edge removed (×) and drawn again", removed and "kg>synthesizer" in await b.edges())
    n0 = await page.js(f"{TEAM_MEM}.graph.nodes.find(n => n.type === 'kg')")
    a = await b.center('.tb-node[data-id="kg"] .tb-node-title')
    await b.drag(a, (a[0] + 50, a[1] + 40))
    n1 = await page.js(f"{TEAM_MEM}.graph.nodes.find(n => n.type === 'kg')")
    z0 = await page.js(f"{TEAM_MEM}.view.z")
    await page.call("Input.dispatchMouseEvent", type="mouseWheel", x=canvas[0], y=canvas[1], deltaX=0, deltaY=-150,
                    modifiers=2)
    await asyncio.sleep(0.3)
    check("a node moved; ctrl + scroll zooms", (n1["x"], n1["y"]) != (n0["x"], n0["y"])
          and await page.js(f"{TEAM_MEM}.view.z") > z0)
    await b.tap('[data-act="layout"]')
    await b.tap('[data-act="team"]')
    await b.tap('[data-depth="sackett"]')
    await b.tap('[data-act="save"]')
    await page.js(f"(() => {{ const i = {TEAM}.querySelector('.tb-modal input'); i.focus(); i.select(); }})()")
    await page.call("Input.insertText", text=f"Browser team {theme}")
    await page.key()
    await asyncio.sleep(2.5)
    await b.tap('.tb-tab[data-tab="templates"]')
    names = await page.js(f"[...{TEAM}.querySelectorAll('.tb-tpl-name')].map(e => e.innerText)") or []
    check("the saved team is under Templates", f"Browser team {theme}" in names, names)
    await page.js(f"[...{TEAM}.querySelectorAll('.tb-tpl')].find(e => e.innerText.includes('Real-world evidence')).click()")
    await asyncio.sleep(0.5)
    await b.tap('.tb-tab[data-tab="agents"]')
    await page.js(f"{TEAM}.querySelector('.tb').scrollIntoView({{block: 'start'}})")
    await asyncio.sleep(0.4)
    await page.shot(f"9-builder-{theme}")
    await b.tap('[data-act="apply"]')
    applied = await page.wait("!!document.querySelector('.mes-team-line') && "
                              "document.body.innerText.includes('Real-world evidence')", 20)
    await asyncio.sleep(1)
    check("Use this team: the question box takes the team", applied and await page.js(PILLS) == [True, True, False, True],
          await page.js(PILLS))
    await page.click("Agent team builder")
    await page.wait(f"!!{TEAM} && {TEAM}.querySelectorAll('.tb-node').length >= 8", 30)
    await asyncio.sleep(0.8)
    in_use = await b.text(".tb-inspector .tb-pill")
    await b.tap('.tb-tab[data-tab="templates"]')
    tagged = await page.js(f"[...{TEAM}.querySelectorAll('.tb-tpl-active .tb-tpl-name')].map(e => e.innerText)")
    await b.tap('.tb-tab[data-tab="agents"]')
    await b.tap('[data-depth="snow"]')
    changed = await b.text(".tb-inspector .tb-pill")
    await page.js(f"{TEAM}.querySelector('.tb-canvas').focus()")
    await page.key("z", "KeyZ", 90, modifiers=4)
    await asyncio.sleep(0.3)
    back = await b.text(".tb-inspector .tb-pill")
    check("the builder shows the team in use, and when the canvas differs from it",
          "In use" in in_use and tagged == ["Real-world evidence"] and "Changed" in changed and "In use" in back,
          (in_use, tagged, changed, back))
    await b.tap('[data-act="back"]')                    # the question is asked in the page's box, with the team
    await page.wait(f"!!{BOX}", 20)
    await asyncio.sleep(0.8)
    await page.type_in(BOX, "In septic shock, is norepinephrine associated with lower mortality?")
    await page.key()
    live = await page.wait("!!document.querySelector('.mes-team-live')", 20)
    done = await page.wait("document.body.innerText.includes('Grounding:') && "
                           "!document.body.innerText.includes('analysis in progress')", 90)
    await asyncio.sleep(0.8)
    rail = await page.js("[...document.querySelectorAll('.mes-team-rail .mes-team-chip')].map(e => e.innerText)")
    check("Asked with the team: its state while it runs, the team above the sources of the answer",
          live and done and rail and len(rail) >= 5, (live, done, rail))
    # a narrow window: the builder stacks its panels and stays inside the window
    await page.wait("!document.querySelector('[data-testid=stToast]')", 15)      # a toast can cover the button
    await page.click("New search")
    await asyncio.sleep(1.5)
    await page.call("Emulation.setDeviceMetricsOverride", width=700, height=1000, deviceScaleFactor=1, mobile=False)
    await page.click("Agent team builder")
    await page.wait(f"!!{TEAM} && {TEAM}.querySelectorAll('.tb-node').length >= 8", 30)
    await asyncio.sleep(1)
    wide = await page.js(f"(() => {{ const r = {TEAM}.querySelector('.tb').getBoundingClientRect(); "
                         "return [Math.round(r.right), document.documentElement.clientWidth]; })()")
    hidden = await page.js(f"""(() => {{ const c = {TEAM}.querySelector('.tb-canvas').getBoundingClientRect();
        return [...{TEAM}.querySelectorAll('.tb-node')].filter(n => {{ const r = n.getBoundingClientRect();
            return r.left < c.left - 1 || r.right > c.right + 1 || r.top < c.top - 1 || r.bottom > c.bottom + 1; }})
            .map(n => n.dataset.id); }})()""")
    await page.js(f"{TEAM}.querySelector('.tb').scrollIntoView({{block: 'start'}})")
    await asyncio.sleep(0.4)
    await page.shot(f"9-builder-narrow-{theme}")
    await page.call("Emulation.clearDeviceMetricsOverride")
    check("a narrow window: nothing wider than the window, every agent in view",
          wide and wide[0] <= wide[1] and hidden == [], (wide, hidden))
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    failed = [c for c in checks if not c[1]]
    ok = not failed and not exceptions and not blank and not errors
    results.append({"scenario": f"9 agent team builder ({theme} theme)", "ok": ok, "exceptions": exceptions,
                    "blank": blank, "js_errors": errors, "detail": "; ".join(f"{n}: {d}" for n, _, d in failed)})
    if not ok:
        await page.shot(f"9-{theme}")
    print(f"{'PASS' if ok else 'FAIL'} 9 agent team builder ({theme} theme): {len(checks) - len(failed)}/{len(checks)} "
          f"checks" + "".join(f"\n     FAIL {n} {d}" for n, _, d in failed), flush=True)


async def entry_scenario(page, port, results):
    """15. The app as run.sh starts it (see the module docstring)."""
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:200]))

    brand = "(document.querySelector('.brand-name strong') || {}).innerText"
    home = await page.wait("document.body.innerText.includes('What would you like to')", 30)
    named = await page.js(f"[{brand}, document.title]")
    check("MES's landing at /, named MES", home and named == ["MES", "MES"], named)
    await page.click("Options", "summary, button")                 # what the page shows only when asked for
    opened = await page.wait("[...document.querySelectorAll('[data-testid=stPopoverBody]')].length > 0", 10)
    check("the Options menu opens", opened)
    exceptions, blank = await page.state()
    errors, page.errors = page.errors, []
    failed = [c for c in checks if not c[1]]
    ok = not failed and not exceptions and not blank and not errors
    results.append({"scenario": "15 started as run.sh starts it", "ok": ok, "exceptions": exceptions, "blank": blank,
                    "js_errors": errors, "detail": "; ".join(f"{n}: {d}" for n, _, d in failed)})
    if not ok:
        await page.shot("15-entry")
    print(f"{'PASS' if ok else 'FAIL'} 15 started as run.sh starts it: {len(checks) - len(failed)}/{len(checks)} checks"
          + "".join(f"\n     FAIL {n} {d}" for n, _, d in failed), flush=True)


def run(out=None, fixtures=None):
    """Run every scenario; returns the list of results (dicts with scenario, ok …)."""
    out = Path(out or tempfile.mkdtemp(prefix="mes_browser_monkey_"))
    out.mkdir(parents=True, exist_ok=True)
    results = []

    async def main():
        proc, port = start_app(None, 0, app=ROOT / "mes" / "app.py")
        page = Page(out)
        try:
            await page.open(f"http://127.0.0.1:{port}/")
            await asyncio.wait_for(entry_scenario(page, port, results), timeout=180)
        finally:
            page.close()
            proc.terminate()
        proc, port = start_app(fixtures, 0.3)
        page = Page(out)
        try:
            await page.open(f"http://127.0.0.1:{port}/")
            await asyncio.wait_for(scenarios(page, port, results), timeout=420)
        finally:
            page.close()
            proc.terminate()
        proc, port = start_app(fixtures, 1.0)
        page = Page(out)
        try:
            await page.open(f"http://127.0.0.1:{port}/")
            await asyncio.wait_for(stop(page, results), timeout=180)
        finally:
            page.close()
            proc.terminate()
        proc, port = start_app(fixtures, 0)
        page = Page(out)
        try:
            await page.open(f"http://127.0.0.1:{port}/")
            await asyncio.wait_for(protocol(page, results), timeout=180)
        finally:
            page.close()
            proc.terminate()
        # the charts of a treatment comparison (the richest card) when the fixtures have one
        proc, port = start_app(fixture_dir(fixtures or Path(os.environ.get("MES_MONKEY_FIXTURES", "")), "compare")
                               or fixtures, 0)
        page = Page(out)
        try:
            await page.open(f"http://127.0.0.1:{port}/")
            await asyncio.wait_for(charts(page, results), timeout=240)
        finally:
            page.close()
            proc.terminate()
        for tool, analysis in (("builder", "compare"), ("explorer", "similar")):
            folder = fixture_dir(fixtures or Path(os.environ.get("MES_MONKEY_FIXTURES", "")), analysis)
            if folder is None:
                print(f"SKIP {tool}: no stored answer with a {analysis} study in the fixtures", flush=True)
                continue
            proc, port = start_app(folder, 0)
            page = Page(out)
            try:
                await page.open(f"http://127.0.0.1:{port}/")
                await asyncio.wait_for(rwd_tool_scenario(page, results, tool), timeout=240)
            finally:
                page.close()
                proc.terminate()
        folder = fixture_dir(fixtures or Path(os.environ.get("MES_MONKEY_FIXTURES", "")), "compare")
        if folder is None:
            print("SKIP trial vs RWD: no stored answer with a compare study in the fixtures", flush=True)
        else:
            proc, port = start_app(folder, 0)
            page = Page(out)
            try:
                await page.open(f"http://127.0.0.1:{port}/")
                await asyncio.wait_for(trial_scenario(page, results), timeout=240)
            finally:
                page.close()
                proc.terminate()
        for theme in ("light", "dark"):
            proc, port = start_app(fixtures, 0)
            page = Page(out, dark=theme == "dark")
            try:
                await page.open(f"http://127.0.0.1:{port}/")
                await asyncio.wait_for(fast(page, theme, results), timeout=120)
            finally:
                page.close()
                proc.terminate()
            proc, port = start_app(fixtures, 0.3)
            page = Page(out, dark=theme == "dark")
            try:
                await page.open(f"http://127.0.0.1:{port}/")
                await asyncio.wait_for(builder(page, port, results, theme), timeout=300)
            finally:
                page.close()
                proc.terminate()

    asyncio.run(main())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="folder for screenshots of failures")
    parser.add_argument("--fixtures", type=Path, default=None, help="cached MES results to answer from")
    args = parser.parse_args()
    results = run(args.out, args.fixtures)
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
