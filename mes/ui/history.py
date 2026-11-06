"""
Saved MES conversations: every answered question is stored, so a conversation can be reopened after a page refresh
or on another day (``.cache/history/<id>.json``, one file per conversation; the newest MAX_THREADS are kept).
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from mes import config
from mes.tools.result_schema import MESResult

logger = logging.getLogger(__name__)

MAX_THREADS = 200
# "edit", "parent", "root", "version": an answer written with an edited protocol (page._edit_protocol)
TURN_KEYS = ("id", "question", "mode", "sources", "verify", "use_cache", "rwd_engine", "history_until", "rwd_note",
             "feedback", "team", "edit", "parent", "root", "version")


def _dir():
    path = Path(os.environ.get("MES_HISTORY_DIR") or config.CACHE_DIR / "history")     # tests use their own folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def save(thread_id, thread):
    """Store the answered turns of a conversation (unanswered ones are skipped)."""
    turns = [t for t in thread if t.get("status") == "done" and t.get("result") is not None
             and not getattr(t["result"], "notice", "")]          # questions MES did not search are not kept
    if not thread_id or not turns:
        return
    record = {"id": thread_id, "title": turns[0]["question"], "updated_at": time.time(),
              "turns": [{**{k: t.get(k) for k in TURN_KEYS}, "result": t["result"].to_dict()} for t in turns]}
    try:
        path = _dir() / f"{thread_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, default=str), encoding="utf-8")
        tmp.replace(path)
        _prune()
    except Exception as e:
        logger.warning("Could not save the conversation: %s", e)


def _prune():
    files = sorted(_dir().glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[MAX_THREADS:]:
        f.unlink(missing_ok=True)


def _entry(f, record):
    turns = record.get("turns") or []
    return {"id": record.get("id", f.stem), "title": record.get("title", ""), "n_turns": len(turns),
            "mode": (turns[-1] if turns else {}).get("mode", "osler"),
            "sources": (turns[0] if turns else {}).get("sources") or [],
            "updated_at": record.get("updated_at", f.stat().st_mtime)}


def _files():
    return sorted(_dir().glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)


def recent(limit=8):
    """[{id, title, mode, n_turns, updated_at, sources}] newest first."""
    out = []
    for f in _files()[:limit]:
        try:
            out.append(_entry(f, json.loads(f.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


def search(text, limit=10):
    """Conversations whose questions contain every word of ``text`` (case-insensitive), newest first."""
    words = str(text or "").lower().split()
    out = []
    for f in _files():
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        questions = " ".join([record.get("title", "")] + [t.get("question", "") for t in record.get("turns") or []])
        if all(w in questions.lower() for w in words):
            out.append(_entry(f, record))
            if len(out) >= limit:
                break
    return out


def load(thread_id):
    """The turns of a saved conversation, ready for the MES page (status done, results restored)."""
    record = json.loads((_dir() / f"{thread_id}.json").read_text(encoding="utf-8"))
    turns = []
    for t in record.get("turns") or []:
        result = MESResult.from_dict(t["result"])
        turns.append({**{k: t.get(k) for k in TURN_KEYS}, "uploads": [], "status": "done", "log": [], "error": "",
                      "result": result, "shown": True})
    return turns


def delete(thread_id):
    (_dir() / f"{thread_id}.json").unlink(missing_ok=True)


def record_feedback(entry):
    """Append a rating of an answer to feedback.jsonl next to the conversations (it syncs like them)."""
    try:
        with open(_dir() / "feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), **entry}, default=str) + "\n")
    except OSError as e:
        logger.warning("Could not save the feedback: %s", e)


def count():
    return sum(1 for _ in _dir().glob("*.json"))


def clear():
    """Delete every saved conversation."""
    for f in _dir().glob("*.json"):
        f.unlink(missing_ok=True)


def ago(timestamp):
    """'just now', '5 min ago', 'yesterday', '3 Oct' …"""
    seconds = time.time() - float(timestamp or 0)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    if seconds < 2 * 86400:
        return "yesterday"
    return datetime.fromtimestamp(float(timestamp)).strftime("%-d %b")
