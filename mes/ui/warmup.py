"""
Background loading of the MES search engine (sentence encoder, iBKH store, PDF index).

Light on purpose: the page calls ``start()`` when it opens, so the engine is ready by the time the user asks a
question; the heavy modules are imported inside the thread.
"""

import threading

state = {"thread": None, "done": False, "seconds": None, "error": None}
_lock = threading.Lock()


def start():
    with _lock:
        if state["thread"] is not None:
            return

        def run():
            try:
                from mes.tools.pipeline_tools import warm_up
                state["seconds"] = warm_up()
            except Exception as e:  # the pipeline reports source errors per question
                state["error"] = str(e)
            state["done"] = True

        state["thread"] = threading.Thread(target=run, daemon=True, name="mes-warmup")
        state["thread"].start()


def ready():
    return state["done"]
