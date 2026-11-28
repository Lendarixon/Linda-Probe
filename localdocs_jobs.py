"""Unified background-job registry: stable IDs + ID-guarded cancellation.

Five families: ocr-batch, ocr-all, image-index, duplicate, content-index.
No tokens or file contents are stored. Scopes carry counts/labels only.
Persistence is bounded atomic JSON; running jobs become interrupted on restart.
No I/O happens at import; state file is resolved lazily.
"""
from __future__ import annotations

import copy
import atexit as _atexit
import json
import os
import threading
import time
import uuid

KINDS = ("ocr-batch", "ocr-all", "image-index", "duplicate", "content-index")
ACTIVE = ("queued", "running", "stopping")
TERMINAL = ("stopped", "completed", "failed", "interrupted")
MAX_RECENT = 20
_ERROR_LIMIT = 200


def _trunc(text, limit=_ERROR_LIMIT):
    try:
        s = str(text or "")
    except Exception:
        s = ""
    return s[:limit]


def _as_int(value, default=0):
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _total_known_while_running(total):
    # Distinguish empty (0 terminal) from unknown (None while running).
    try:
        t = int(total)
    except (TypeError, ValueError, OverflowError):
        return None
    if t < 0:
        return None
    return t


# ---- pure normalizers: raw helper dict -> (status, done, total|None, error) ----
def normalize_ocr_batch(raw, cancel_requested=False):
    raw = raw or {}
    phase = raw.get("phase", "idle")
    done = _as_int(raw.get("done"))
    total = raw.get("total", 0)
    stopped = bool(raw.get("stopped"))
    if phase == "running":
        st = "stopping" if (cancel_requested or stopped) else "running"
        tot = _total_known_while_running(total)
        if tot == 0 and done == 0:
            # total not yet computed while expanding inputs
            tot = None
        return st, done, tot, ""
    if phase == "done":
        try:
            tot = int(total)
            if tot < 0:
                tot = 0
        except (TypeError, ValueError, OverflowError):
            tot = done
        if cancel_requested or stopped:
            return "stopped", done, tot, ""
        try:
            results = raw.get("results") or []
        except Exception:
            results = []
        first_err = ""
        try:
            for r in results:
                if isinstance(r, dict) and r.get("error"):
                    first_err = str(r["error"])
                    break
        except Exception:
            first_err = ""
        if first_err:
            return "failed", done, tot, _trunc(first_err)
        return "completed", done, tot, ""
    # idle (never ran or reset)
    if cancel_requested or stopped:
        return "stopped", done, _as_int(total), ""
    return "completed", done, _as_int(total), ""


def normalize_ocr_all(raw, cancel_requested=False):
    raw = raw or {}
    phase = raw.get("phase", "idle")
    done = _as_int(raw.get("done"))
    total = raw.get("total", 0)
    stop = bool(raw.get("stop"))
    err = _trunc(raw.get("error") or "")
    if phase == "running":
        st = "stopping" if (cancel_requested or stop) else "running"
        tot = _total_known_while_running(total)
        if tot == 0 and done == 0:
            tot = None
        return st, done, tot, ""
    if phase == "error":
        if cancel_requested or stop:
            try:
                tot = int(total)
                if tot < 0:
                    tot = 0
            except (TypeError, ValueError, OverflowError):
                tot = done
            return "stopped", done, tot, ""
        try:
            tot = int(total)
            if tot < 0:
                tot = 0
        except (TypeError, ValueError, OverflowError):
            tot = done
        return "failed", done, tot, err or "error"
    try:
        tot = int(total)
        if tot < 0:
            tot = 0
    except (TypeError, ValueError, OverflowError):
        tot = done
    if cancel_requested or stop:
        return "stopped", done, tot, ""
    if err:
        return "failed", done, tot, err
    return "completed", done, tot, ""


def normalize_image(raw, cancel_requested=False):
    raw = raw or {}
    phase = raw.get("phase", "idle")
    done = _as_int(raw.get("done"))
    total = raw.get("total", 0)
    err = _trunc(raw.get("error") or "")
    if phase == "running":
        st = "stopping" if cancel_requested else "running"
        tot = _total_known_while_running(total)
        if tot == 0 and done == 0:
            tot = None
        return st, done, tot, ""
    if phase == "error":
        if cancel_requested:
            return "stopped", done, _as_int(total), ""
        return "failed", done, _as_int(total), err or "error"
    if phase == "done":
        if cancel_requested:
            return "stopped", done, _as_int(total), ""
        return "completed", done, _as_int(total), ""
    # idle
    if cancel_requested:
        return "stopped", done, _as_int(total), ""
    return "completed", done, _as_int(total), ""


def normalize_duplicate(raw, cancel_requested=False):
    raw = raw or {}
    phase = raw.get("phase", "idle")
    scanned = _as_int(raw.get("scanned"))
    stop = bool(raw.get("stop"))
    err = _trunc(raw.get("error") or "")
    if phase == "running":
        st = "stopping" if (cancel_requested or stop) else "running"
        return st, scanned, None, ""
    if phase == "error":
        if cancel_requested or stop:
            return "stopped", scanned, scanned, ""
        return "failed", scanned, scanned, err or "error"
    if cancel_requested or stop:
        return "stopped", scanned, scanned, ""
    if err:
        return "failed", scanned, scanned, err
    return "completed", scanned, scanned, ""


def normalize_content_index(raw, cancel_requested=False):
    raw = raw or {}
    phase = raw.get("phase", "idle")
    files = _as_int(raw.get("files"))
    err = _trunc(raw.get("error") or "")
    if phase == "running":
        if cancel_requested:
            return "stopping", files, None, ""
        return "running", files, None, ""
    if phase in ("interrupted", "stopped"):
        return "stopped", files, files, ""
    if phase == "error":
        if cancel_requested:
            return "stopped", files, files, ""
        return "failed", files, files, err or "error"
    if phase == "done":
        if cancel_requested:
            return "stopped", files, files, ""
        return "completed", files, files, ""
    if cancel_requested:
        return "stopped", files, files, ""
    return "completed", files, files, ""


NORMALIZERS = {
    "ocr-batch": normalize_ocr_batch,
    "ocr-all": normalize_ocr_all,
    "image-index": normalize_image,
    "duplicate": normalize_duplicate,
    "content-index": normalize_content_index,
}


def _sanitize_scope(scope):
    if scope is None:
        return {}
    if not isinstance(scope, dict):
        return {}
    out = {}
    for key, value in list(scope.items())[:8]:
        if not isinstance(key, str) or len(key) > 32:
            continue
        low = key.lower()
        if "token" in low or "secret" in low or "path" in low or "file" in low:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            s = _trunc(value, 80) if isinstance(value, str) else value
            out[key] = s
        elif isinstance(value, list):
            safe = []
            for item in value[:10]:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    safe.append(_trunc(item, 80) if isinstance(item, str) else item)
            out[key] = safe
    return out


def _sanitize_tab(tab_id):
    if isinstance(tab_id, str) and tab_id and len(tab_id) <= 64:
        return tab_id
    return None


class Registry:
    def __init__(self, state_file=None, max_recent=MAX_RECENT):
        self._lock = threading.RLock()
        self._jobs = {}  # id -> record
        self._active = {}  # family_key -> id
        self._recent = []  # terminal ids, newest last
        self._backends = {}  # family_key -> {status, stop, normalize}
        self._override = state_file
        self._max_recent = max(1, int(max_recent or MAX_RECENT))
        self._loaded = False
        # Mutation generation: bumped on every state change while holding
        # _lock. Lets an unlocked disk flush detect a concurrent mutation
        # and re-snapshot instead of persisting a stale view.
        self._gen = 0
        # Families with a backend dispatch in flight (record created, backend
        # not yet started). Status polls skip these: a sample taken before
        # our dispatch would otherwise retire the fresh record on stale data.
        self._dispatching = set()

    # ---- configuration ----
    def configure(self, path):
        with self._lock:
            self._override = path
            self._loaded = False
            self._jobs.clear()
            self._active.clear()
            self._recent = []
            self._backends.clear()
            self._dispatching.clear()
            self._gen += 1

    def _state_file(self):
        if self._override:
            return self._override
        env = os.environ.get("LOCALDOCS_JOBS_FILE")
        if env:
            return env
        try:
            import localdocs_appcfg as _appcfg

            return str(_appcfg.data_root() / "indexes" / "_jobs.json")
        except Exception:
            root = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(os.environ.get("LOCALDOCS_WRITE_DIR") or root, "indexes", "_jobs.json")

    # ---- persistence ----
    # Locking contract: _ensure_loaded_locked / _snapshot_locked must be
    # called with self._lock held (they only touch in-memory state).
    # Disk I/O happens in _flush, which never holds the lock while writing,
    # so a stalled disk cannot block list/get/cancel on the RLock.
    def _ensure_loaded(self):
        """Locking wrapper for external callers; prefers _ensure_loaded_locked."""
        with self._lock:
            self._ensure_loaded_locked()

    def _ensure_loaded_locked(self):
        if self._loaded:
            return
        self._loaded = True
        path = self._state_file()
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
        except (FileNotFoundError, NotADirectoryError):
            return
        except (OSError, ValueError):
            return
        try:
            items = data.get("jobs") if isinstance(data, dict) else None
            if not isinstance(items, list):
                return
            for entry in items[: self._max_recent + 8]:
                if not isinstance(entry, dict):
                    continue
                if entry.get("kind") not in KINDS:
                    continue
                jid = entry.get("id")
                if not isinstance(jid, str) or not jid:
                    continue
                rec = {
                    "id": jid,
                    "kind": entry.get("kind"),
                    "tab_id": _sanitize_tab(entry.get("tab_id")),
                    "scope": _sanitize_scope(entry.get("scope")),
                    "family": str(entry.get("family") or entry.get("kind")),
                    "status": "interrupted" if entry.get("status") in ACTIVE else entry.get("status"),
                    "done": _as_int(entry.get("done")),
                    "total": entry.get("total") if entry.get("total") is None else _as_int(entry.get("total")),
                    "error": _trunc(entry.get("error") or ""),
                    "can_cancel": bool(entry.get("can_cancel", True)),
                    "cancel_requested": False,
                    "created_at": entry.get("created_at") or 0,
                    "updated_at": time.time(),
                }
                if rec["status"] not in TERMINAL:
                    rec["status"] = "interrupted"
                self._jobs[jid] = rec
                if rec["status"] in TERMINAL:
                    self._recent.append(jid)
            # Persisted file is newest-first (active + reversed recent);
            # keep the newest entries then restore oldest-first order.
            self._recent = self._recent[: self._max_recent][::-1]
            self._gen += 1
        except Exception:
            return

    def _snapshot_locked(self):
        """Build (path, payload) purely in memory. Caller must hold _lock. No I/O."""
        path = self._state_file()
        items = []
        for jid in list(self._active.values()):
            rec = self._jobs.get(jid)
            if rec is not None:
                items.append(self._persist_view(rec))
        for jid in reversed(self._recent):
            if jid in self._active.values():
                continue
            rec = self._jobs.get(jid)
            if rec is not None:
                items.append(self._persist_view(rec))
        items = items[: self._max_recent + 8]
        return path, {"version": 1, "jobs": items}

    @staticmethod
    def _write_file(path, payload):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    def _flush(self, path, payload, gen):
        """Write a snapshot taken under lock. The lock is never held during
        I/O; if the generation moved while writing, re-snapshot and retry
        (bounded) instead of persisting a stale view."""
        snap = (path, payload)
        for _ in range(4):
            try:
                self._write_file(*snap)
            except (OSError, ValueError, TypeError):
                return
            with self._lock:
                if self._gen == gen:
                    return
                snap = self._snapshot_locked()
                gen = self._gen

    def _save(self):
        """Compat flush: snapshot under lock, write outside it.

        Safe to call with or without the lock held. Internal mutation sites
        snapshot inline and call _flush after releasing the lock instead.
        """
        with self._lock:
            path, payload = self._snapshot_locked()
            gen = self._gen
        self._flush(path, payload, gen)

    @staticmethod
    def _persist_view(rec):
        return {
            "id": rec["id"],
            "kind": rec["kind"],
            "tab_id": rec["tab_id"],
            "scope": rec["scope"],
            "family": rec["family"],
            "status": rec["status"],
            "done": rec["done"],
            "total": rec["total"],
            "error": rec["error"],
            "can_cancel": rec["can_cancel"],
            "created_at": rec.get("created_at") or 0,
            "updated_at": rec.get("updated_at") or 0,
        }

    # ---- views ----
    @staticmethod
    def _public(rec):
        return {
            "id": rec["id"],
            "kind": rec["kind"],
            "tab_id": rec["tab_id"],
            "scope": copy.deepcopy(rec["scope"]),
            "phase": rec["status"],
            "status": rec["status"],
            "done": rec["done"],
            "total": rec["total"],
            "error": rec["error"],
            "can_cancel": bool(rec["can_cancel"]),
        }

    @staticmethod
    def family_key(kind, scope):
        if kind == "content-index" and isinstance(scope, dict):
            proj = scope.get("project")
            if isinstance(proj, str) and proj:
                return "content-index:%s" % proj[:120]
        return kind

    # ---- core operations ----
    # Locking contract: identity (which id is active for a family) is always
    # decided while holding _lock. Foreign backend calls (start/status/stop)
    # always run WITHOUT the lock; afterwards the generation and family are
    # reconciled under the lock so a concurrent cancel/start is never
    # clobbered by a stale dispatch.
    @staticmethod
    def _call_status(fn):
        try:
            return fn()
        except Exception:
            return None

    def _collect_locked(self, fam=None):
        """Status targets for active jobs. Caller must hold _lock. No I/O, no backend calls."""
        fams = [fam] if fam is not None else list(self._active.keys())
        out = []
        for f in fams:
            if f in self._dispatching:
                # Backend not started yet for this family: any sample would
                # be pre-dispatch backend state. Skip, keep last known.
                continue
            jid = self._active.get(f)
            rec = self._jobs.get(jid) if jid else None
            if rec is None or rec["status"] not in ACTIVE:
                continue
            be = self._backends.get(f) or {}
            fn = be.get("status")
            if not callable(fn):
                continue
            norm = be.get("normalize") or NORMALIZERS.get(rec["kind"], normalize_content_index)
            out.append((f, jid, fn, norm))
        return out

    def _apply_one_locked(self, fam, jid, norm, raw):
        """Apply one polled payload. Returns True on state change.

        Never touches a record superseded while polling. Caller must hold _lock.
        """
        if self._active.get(fam) != jid:
            return False
        rec = self._jobs.get(jid)
        if rec is None or rec["status"] in TERMINAL or raw is None:
            return False
        try:
            st, done, total, err = norm(raw or {}, rec["cancel_requested"])
        except Exception:
            return False
        if st not in ACTIVE and st not in TERMINAL:
            return False
        # Never report completed when the backend signals an error string.
        if st == "completed" and err:
            st = "failed"
        new = (st, _as_int(done), total if total is None else _as_int(total), _trunc(err))
        if new == (rec["status"], rec["done"], rec["total"], rec["error"]):
            return False
        rec["status"], rec["done"], rec["total"], rec["error"] = new
        rec["updated_at"] = time.time()
        self._gen += 1
        if st in TERMINAL:
            self._retire_locked(fam)
        return True

    def _snap_locked(self):
        path, payload = self._snapshot_locked()
        return (path, payload, self._gen)

    def _flush_snap(self, snap):
        if snap is not None:
            self._flush(*snap)

    def _poll_and_apply(self, fam=None):
        """Poll backend status without the lock, apply under it.

        Returns a flush snapshot triple or None when nothing changed.
        """
        with self._lock:
            self._ensure_loaded_locked()
            targets = self._collect_locked(fam)
        raws = [(f, j, n, self._call_status(fn)) for (f, j, fn, n) in targets]
        with self._lock:
            changed = False
            for f, j, n, raw in raws:
                if self._apply_one_locked(f, j, n, raw):
                    changed = True
            if not changed:
                return None
            return self._snap_locked()

    def start(self, kind, scope=None, tab_id=None, can_cancel=True,
              backend_start=None, backend_status=None, backend_stop=None, normalize=None):
        if kind not in KINDS:
            raise ValueError("unknown kind")
        scope = _sanitize_scope(scope)
        tab = _sanitize_tab(tab_id)
        norm = normalize or NORMALIZERS[kind]
        with self._lock:
            self._ensure_loaded_locked()
            fam = self.family_key(kind, scope)
        # Busy check against fresh backend state; polling runs unlocked.
        self._flush_snap(self._poll_and_apply(fam))
        with self._lock:
            cur_id = self._active.get(fam)
            if cur_id is not None:
                cur = self._jobs.get(cur_id)
                if cur is not None and cur["status"] in ACTIVE:
                    return self._public(cur), False
            jid = uuid.uuid4().hex
            now = time.time()
            rec = {
                "id": jid, "kind": kind, "tab_id": tab, "scope": scope,
                "family": fam, "status": "running", "done": 0, "total": None,
                "error": "", "can_cancel": bool(can_cancel),
                "cancel_requested": False, "created_at": now, "updated_at": now,
            }
            self._jobs[jid] = rec
            self._active[fam] = jid
            self._backends[fam] = {"status": backend_status, "stop": backend_stop, "normalize": norm}
            self._dispatching.add(fam)
            self._gen += 1
            snap = self._snap_locked()
            stop_fn = backend_stop if callable(backend_stop) else None
        self._flush(*snap)
        # Dispatch while NOT holding the lock so a concurrent cancel cannot
        # stall on a slow backend start.
        try:
            ok = backend_start() if callable(backend_start) else True
        except Exception as exc:
            snap = None
            with self._lock:
                self._dispatching.discard(fam)
                if self._active.get(fam) == jid:
                    rec["status"] = "failed"
                    rec["error"] = _trunc(exc)
                    rec["updated_at"] = time.time()
                    self._gen += 1
                    self._retire_locked(fam)
                    snap = self._snap_locked()
                # Else superseded mid-dispatch: leave the new run alone.
            self._flush_snap(snap)
            raise
        with self._lock:
            self._dispatching.discard(fam)
            if self._active.get(fam) != jid:
                # Superseded (or retired by a concurrent refresh) while the
                # backend was starting: stop the orphan, report our record.
                orphan_stop = stop_fn
                out = self._public(rec)
                rest_snap = None
                need_reconcile_stop = False
            elif ok is False:
                # Backend reports busy without our record: drop phantom, report busy.
                self._jobs.pop(jid, None)
                if self._active.get(fam) == jid:
                    self._active.pop(fam, None)
                    self._backends.pop(fam, None)
                self._gen += 1
                rest_snap = self._snap_locked()
                out = None
                orphan_stop = None
                need_reconcile_stop = False
            else:
                orphan_stop = None
                rest_snap = None
                if rec["cancel_requested"]:
                    # A cancel arrived mid-dispatch: converge to stopping.
                    rec["status"] = "stopping"
                    rec["updated_at"] = time.time()
                    self._gen += 1
                    rest_snap = self._snap_locked()
                out = self._public(rec)
                need_reconcile_stop = bool(rec["cancel_requested"])
        if orphan_stop is not None:
            try:
                orphan_stop()
            except Exception:
                pass
            return out, True
        if out is None:
            self._flush_snap(rest_snap)
            return None, False
        self._flush_snap(rest_snap)
        if need_reconcile_stop and stop_fn is not None:
            # The backend started after cancel was requested: deliver the stop.
            try:
                stop_fn()
            except Exception as exc:
                with self._lock:
                    if self._active.get(fam) == jid and jid in self._jobs:
                        self._jobs[jid]["error"] = _trunc(exc)
                        self._jobs[jid]["updated_at"] = time.time()
                        self._gen += 1
                        self._flush_snap(self._snap_locked())
        self._flush_snap(self._poll_and_apply(fam))
        with self._lock:
            return self._public(self._jobs.get(jid, rec)), True

    def _retire_locked(self, fam):
        """Move an active id to recent. Caller must hold _lock."""
        jid = self._active.pop(fam, None)
        self._backends.pop(fam, None)
        if jid is None:
            return
        self._gen += 1
        if jid in self._jobs:
            if jid in self._recent:
                self._recent.remove(jid)
            self._recent.append(jid)
            self._recent = self._recent[-self._max_recent :]

    def refresh(self):
        self._flush_snap(self._poll_and_apply())

    def cancel(self, job_id):
        with self._lock:
            self._ensure_loaded_locked()
            rec = self._jobs.get(job_id)
            if rec is None:
                return None, "not-found"
            if rec["status"] in TERMINAL:
                return self._public(rec), "terminal-idempotent"
            fam = rec["family"]
            if self._active.get(fam) != job_id:
                # Old ID superseded by a newer run: never touch the new run.
                return self._public(rec), "superseded"
            if not rec["can_cancel"]:
                return self._public(rec), "not-cancellable"
            if rec["cancel_requested"] or rec["status"] == "stopping":
                return self._public(rec), "already-stopping"
            rec["cancel_requested"] = True
            if rec["status"] == "running":
                rec["status"] = "stopping"
            rec["updated_at"] = time.time()
            be = self._backends.get(fam) or {}
            stop = be.get("stop") if callable(be.get("stop")) else None
            self._gen += 1
            snap = self._snap_locked()
            out = self._public(rec)
        # Backend stop runs without the lock; exactly-once is guaranteed by
        # the cancel_requested flag set above.
        if stop is not None:
            try:
                stop()
            except Exception as exc:
                with self._lock:
                    if self._active.get(fam) == job_id and job_id in self._jobs:
                        self._jobs[job_id]["error"] = _trunc(exc)
                        self._jobs[job_id]["updated_at"] = time.time()
                        self._gen += 1
                        self._flush_snap(self._snap_locked())
        self._flush(*snap)
        return out, "stopping"

    def family_stop(self, kind, scope=None):
        """Legacy stop without ID: mark active job stopping and dispatch backend stop."""
        with self._lock:
            self._ensure_loaded_locked()
            fam = self.family_key(kind, _sanitize_scope(scope))
            jid = self._active.get(fam)
            if jid is None:
                return None
            rec = self._jobs.get(jid)
            if rec is None or rec["status"] in TERMINAL or not rec["can_cancel"]:
                return self._public(rec) if rec is not None else None
            rec["cancel_requested"] = True
            if rec["status"] == "running":
                rec["status"] = "stopping"
            rec["updated_at"] = time.time()
            be = self._backends.get(fam) or {}
            stop = be.get("stop") if callable(be.get("stop")) else None
            self._gen += 1
            snap = self._snap_locked()
            out = self._public(rec)
        if stop is not None:
            try:
                stop()
            except Exception:
                pass
        self._flush(*snap)
        return out

    def get(self, job_id):
        self._flush_snap(self._poll_and_apply())
        with self._lock:
            rec = self._jobs.get(job_id)
            return self._public(rec) if rec is not None else None

    def list(self):
        self._flush_snap(self._poll_and_apply())
        with self._lock:
            out = []
            for fam, jid in self._active.items():
                rec = self._jobs.get(jid)
                if rec is not None:
                    out.append(self._public(rec))
            for jid in reversed(self._recent):
                if jid in (self._active.get(f) for f in self._active):
                    continue
                rec = self._jobs.get(jid)
                if rec is not None:
                    out.append(self._public(rec))
            return out

    def reset(self):
        with self._lock:
            self._jobs.clear()
            self._active.clear()
            self._recent = []
            self._backends.clear()
            self._dispatching.clear()
            self._loaded = True
            self._gen += 1


REGISTRY = Registry()


def configure(path):
    REGISTRY.configure(path)


def start_family(kind, scope=None, tab_id=None, can_cancel=True,
                 backend_start=None, backend_status=None, backend_stop=None, normalize=None):
    return REGISTRY.start(kind, scope, tab_id, can_cancel,
                          backend_start, backend_status, backend_stop, normalize)


def cancel_job(job_id):
    return REGISTRY.cancel(job_id)


def family_stop(kind, scope=None):
    return REGISTRY.family_stop(kind, scope)


def get_job(job_id):
    return REGISTRY.get(job_id)


def list_jobs():
    return REGISTRY.list()


def handle(body, auth_ok, owner_ok):
    """Pure HTTP helper for POST /jobs. Returns (code, payload)."""
    if not auth_ok:
        return 403, {"error": "forbidden"}
    if not owner_ok:
        return 403, {"error": "owner-only"}
    if not isinstance(body, dict):
        return 400, {"error": "body-must-be-object"}
    action = body.get("action", "list")
    if action == "list":
        return 200, {"jobs": REGISTRY.list()}
    if action == "status":
        jid = body.get("job_id")
        if not isinstance(jid, str) or not jid:
            return 400, {"error": "job_id required"}
        rec = REGISTRY.get(jid)
        if rec is None:
            return 404, {"error": "unknown job"}
        return 200, {"job": rec}
    if action == "cancel":
        jid = body.get("job_id")
        if not isinstance(jid, str) or not jid:
            return 400, {"error": "job_id required"}
        rec, _reason = REGISTRY.cancel(jid)
        if rec is None:
            return 404, {"error": "unknown job"}
        return 200, {"job": rec}
    return 400, {"error": "unknown action"}


def _flush_at_exit():
    """Best-effort persist on interpreter shutdown (daemon threads are never
    joined by the registry itself; it spawns none — callers join their own)."""
    try:
        REGISTRY._save()
    except Exception:
        pass


_atexit.register(_flush_at_exit)
