"""Event-driven re-indexing: file-system events (no polling) trigger a re-index only when something indexable changed."""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

try:
    from localdocs_dirwatch import DirWatcher
except Exception:  # pragma: no cover - the watcher is optional
    DirWatcher = None


class EventIndexer:
    """Watches every project folder; after a quiet period it asks for a re-index of the changed project.

    get_projects() -> {name: folder}; start_index(name) -> bool (False when one is already running).
    """

    def __init__(self, get_projects: Callable[[], dict], start_index: Callable[[str], bool],
                 extensions, skip_dirs=(), ignore_prefixes=(), quiet_s: float = 15.0, retry_s: float = 30.0,
                 log: Callable[[dict], None] | None = None):
        self._get = get_projects
        self._start = start_index
        self._ext = {e.lower() for e in extensions}
        self._skip = {s.lower() for s in skip_dirs}
        self._ignore = [p for p in ignore_prefixes if p]
        self._quiet = max(0.5, float(quiet_s))
        self._retry = max(0.5, float(retry_s))
        self._log = log or (lambda record: None)
        self._lock = threading.RLock()
        self._watchers: dict = {}      # name -> (folder, DirWatcher)
        self._timers: dict = {}        # name -> threading.Timer
        self._info: dict = {}          # name -> {"events": n, "last_event": ts, "reindexes": n}
        self._stopped = False

    # ---- relevance -------------------------------------------------------------------------
    def relevant(self, event: dict) -> bool:
        if event.get("overflow"):
            return True
        path = str(event.get("path") or "")
        if not path:
            return False
        parts = [p.lower() for p in path.replace("/", chr(92)).split(chr(92))]
        if any(p in self._skip for p in parts[:-1]):
            return False
        ext = os.path.splitext(parts[-1])[1]
        return (not ext and event.get("action") in ("removed", "renamed_old", "renamed_new")) or ext in self._ext

    # ---- lifecycle -------------------------------------------------------------------------
    def sync(self) -> dict:
        """Start watchers for new projects and stop watchers of removed ones."""
        if DirWatcher is None or self._stopped:
            return self.status()
        wanted = {name: folder for name, folder in (self._get() or {}).items() if isinstance(folder, str) and os.path.isdir(folder)}
        with self._lock:
            for name in list(self._watchers):
                if name not in wanted or self._watchers[name][0] != wanted[name]:
                    self._stop_one(name)
            for name, folder in wanted.items():
                if name in self._watchers:
                    continue
                try:
                    watcher = DirWatcher(folder, lambda events, n=name: self._on_events(n, events), recursive=True,
                                         debounce_s=1.0, ignore=list(self._ignore),
                                         on_error=lambda exc, n=name: self._log({"project": n, "watch_error": str(exc)[:200]}))
                    watcher.start()
                    self._watchers[name] = (folder, watcher)
                    self._info.setdefault(name, {"events": 0, "last_event": None, "reindexes": 0})
                except Exception as exc:  # never break the app because of a watcher
                    self._log({"project": name, "watch_start_error": str(exc)[:200]})
        return self.status()

    def _stop_one(self, name: str) -> None:
        item = self._watchers.pop(name, None)
        timer = self._timers.pop(name, None)
        if timer:
            timer.cancel()
        if item:
            try:
                item[1].stop()
            except Exception:
                pass

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            for name in list(self._watchers):
                self._stop_one(name)

    # ---- events -> debounced re-index ---------------------------------------------------------
    def _on_events(self, name: str, events: list) -> None:
        hits = [e for e in events or [] if self.relevant(e)]
        if not hits:
            return
        with self._lock:
            info = self._info.setdefault(name, {"events": 0, "last_event": None, "reindexes": 0})
            info["events"] += len(hits)
            info["last_event"] = time.time()
        self._log({"project": name, "events": len(hits), "sample": [str(h.get("path", "overflow"))[-120:] for h in hits[:3]]})
        self._arm(name, self._quiet)

    def _arm(self, name: str, delay: float) -> None:
        with self._lock:
            if self._stopped:
                return
            old = self._timers.pop(name, None)
            if old:
                old.cancel()
            timer = threading.Timer(delay, self._fire, args=(name,))
            timer.daemon = True
            self._timers[name] = timer
            timer.start()

    def _fire(self, name: str) -> None:
        with self._lock:
            self._timers.pop(name, None)
            if self._stopped or name not in self._watchers:
                return
        try:
            started = bool(self._start(name))
        except Exception as exc:
            self._log({"project": name, "reindex_error": str(exc)[:200]})
            started = False
        if started:
            with self._lock:
                self._info.setdefault(name, {"events": 0, "last_event": None, "reindexes": 0})["reindexes"] += 1
        else:
            self._arm(name, self._retry)  # an index run is in progress: try again later

    def status(self) -> dict:
        with self._lock:
            return {"available": DirWatcher is not None,
                    "projects": {n: {"watching": n in self._watchers, "pending": n in self._timers, **self._info.get(n, {})}
                                 for n in set(self._watchers) | set(self._info)}}
