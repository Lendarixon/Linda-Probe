"""Indexing queue: watcher/scheduler enqueue folders, one worker thread
embeds sequentially (single embedder, no contention).

Bounded + deduplicated: at most _MAXSIZE folders wait in the queue, a path
that is already queued or being indexed is not enqueued again, and the
_results dict is capped by entry count and TTL so it cannot grow forever.
"""
import queue as _queue
import threading as _threading
import time as _time

_MAXSIZE = 8
_RESULTS_MAX = 128
_RESULT_TTL_SECS = 3600.0

_q = _queue.Queue(maxsize=_MAXSIZE)
_results = {}
_pending = set()
_worker = None
_lock = _threading.Lock()


def _evict(now=None):
    """Drop expired results and trim to _RESULTS_MAX entries (oldest first)."""
    now = _time.time() if now is None else now
    for key, rec in list(_results.items()):
        ts = rec.get("_ts", now) if isinstance(rec, dict) else now
        if now - ts > _RESULT_TTL_SECS:
            del _results[key]
    while len(_results) > _RESULTS_MAX:
        oldest = min(
            _results,
            key=lambda k: _results[k].get("_ts", 0)
            if isinstance(_results[k], dict) else 0,
        )
        del _results[oldest]


def _store(folder, payload):
    rec = dict(payload)
    rec["_ts"] = _time.time()
    _results[folder] = rec
    _evict()


def enqueue(folder):
    """Queue a folder unless full or already queued/running.

    Returns the queue size on success, False when the queue is full
    (maxsize 8, non-blocking) or the path is already queued/running.
    """
    key = str(folder)
    with _lock:
        if key in _pending:
            return False
        try:
            _q.put_nowait(key)
        except _queue.Full:
            return False
        _pending.add(key)
        return _q.qsize()


def _work():
    from project_search.backend import index_project
    while True:
        folder = _q.get()
        try:
            t0 = _time.time()
            rep = index_project(folder)
            with _lock:
                _store(folder, {"ok": True, "files": rep["files"],
                                "new": rep["new_embeddings"],
                                "secs": round(_time.time() - t0, 1)})
        except Exception as e:
            with _lock:
                _store(folder, {"ok": False, "error": str(e)[:200]})
        finally:
            with _lock:
                _pending.discard(folder)
            _q.task_done()


def start():
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = _threading.Thread(target=_work, daemon=True)
            _worker.start()
    return True


def status():
    with _lock:
        _evict()
        done = {k: {kk: vv for kk, vv in v.items() if kk != "_ts"}
                if isinstance(v, dict) else v
                for k, v in _results.items()}
        return {"queued": _q.qsize(), "done": done}
