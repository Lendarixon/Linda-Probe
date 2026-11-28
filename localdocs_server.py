"""LocalDocs product server (stdlib, CPU-only). Token + allowlist ACL."""
import json, os, sys
from project_search.resources import configure as _resource_configure
_SERVER_RESOURCE_PROFILE = _resource_configure()
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LOCALDOCS_WRITE_DIR") or ROOT  # writable data; the Store package is read-only
sys.path.insert(0, ROOT)
import localdocs_settings as _settings
import localdocs_jobs
TOKEN = os.environ.get("LOCALDOCS_TOKEN", "changeme")
TOKENS = {}  # token -> set(groups); empty = full access (legacy single-token mode)
PROJ_GROUPS = {}  # project -> set(groups); empty = any authenticated token
_CFG_FILE = os.environ.get("LOCALDOCS_CONFIG", os.path.join(ROOT, "localdocs_config.json"))
_AUDIT_FILE = os.path.join(DATA, "logs", "queries.log")
def _audit(project, query, hits):
    try:
        os.makedirs(os.path.dirname(_AUDIT_FILE), exist_ok=True)
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                                          "project": project, "query": query[:200], "hits": hits},
                                         ensure_ascii=False) + "\n")
    except OSError:
        pass
def _audit_click_rel(project, filename):
    """Project-relative rel path for click audit (rel only, never absolute)."""
    try:
        root = os.path.realpath(ALLOW[project])
        target = os.path.realpath(filename if os.path.isabs(filename) else os.path.join(root, filename))
        rel = os.path.relpath(target, root).replace(os.sep, "/")
        if rel == ".." or rel.startswith("../"):
            return "[outside-project]"
        return rel[:500] or "[unknown]"
    except Exception:
        return "[unknown]"
def _audit_click_project_for(abspath):
    """Map an absolute path to (project, rel); outside any project -> (None, marker)."""
    try:
        target = os.path.realpath(abspath)
        for name, root in ALLOW.items():
            try:
                rr = os.path.realpath(root)
                if os.path.commonpath([rr, target]) == rr:
                    return name, os.path.relpath(target, rr).replace(os.sep, "/")[:500]
            except ValueError:
                continue
    except Exception:
        pass
    return None, "[outside-project]"
def _audit_click(project, opened_path_rel, rank_pos=None, mission_id=None, query=""):
    """Click-audit event (query→click loop). Legacy query lines untouched (no 'ev' key there)."""
    try:
        import datetime as _dt
        import re as _re
        ok_mission = isinstance(mission_id, str) and bool(_re.fullmatch(r"[A-Za-z0-9-]{1,64}", mission_id))
        ev = "click" if ok_mission else "open-nomission"
        try:
            rank = int(rank_pos) if rank_pos is not None else None
        except (ValueError, TypeError):
            rank = None
        if rank is not None and not (0 <= rank <= 100000):
            rank = None
        rel = opened_path_rel if isinstance(opened_path_rel, str) else ""
        rel = rel.replace("\\", "/")[:500] or "[unknown]"
        if _re.match(r"[A-Za-z]:/", rel) or rel.startswith("\\\\"):
            rel = "[outside-project]"
        proj = project if isinstance(project, str) else None
        proj = (proj or "")[:120]
        q = query if isinstance(query, str) else ""
        os.makedirs(os.path.dirname(_AUDIT_FILE), exist_ok=True)
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _dt.datetime.now().isoformat(timespec="seconds"), "ev": ev,
                                          "project": proj, "query": q[:200], "opened_path_rel": rel,
                                          "rank_pos": rank, "mission_id": mission_id if ok_mission else None},
                                         ensure_ascii=False) + "\n")
    except OSError:
        pass
def _load_allow():
    default = {"demo": os.path.join(ROOT, "demo_docs")}
    try:
        with open(_CFG_FILE, encoding="utf-8-sig") as config_file:
            cfg = json.load(config_file)
    except FileNotFoundError:
        cfg = {}
    if not isinstance(cfg, dict) or not isinstance(cfg.get("projects", {}), dict) or not isinstance(cfg.get("users", {}), dict):
        raise ValueError("Invalid LocalDocs config: projects and users must be objects")
    allow = {}
    for k, v in cfg.get("projects", {}).items():
        if isinstance(v, dict):
            if not isinstance(v.get("path"), str) or not v['path']:
                raise ValueError("Invalid LocalDocs project path: " + str(k))
            allow[k] = v["path"] if os.path.isabs(v["path"]) else os.path.join(ROOT, v["path"])
            PROJ_GROUPS[k] = set(v.get("groups", []))
        else:
            if not isinstance(v, str) or not v:
                raise ValueError("Invalid LocalDocs project path: " + str(k))
            allow[k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    users = cfg.get("users", {})
    if users:
        for tok, u in users.items():
            TOKENS[tok] = set(u.get("groups", [])) if isinstance(u, dict) else set()
    tok = cfg.get("token") or os.environ.get("LOCALDOCS_TOKEN") or ""
    if not tok or tok == "changeme":
        import secrets
        tok = secrets.token_urlsafe(32)
        cfg["token"] = tok
        try:
            cfg.setdefault("projects", {"demo": "demo_docs"})
            with open(_CFG_FILE, "w", encoding="utf-8") as config_file:
                json.dump(cfg, config_file, indent=2)
        except OSError:
            pass
    global TOKEN
    TOKEN = tok
    global _OWNER_TOKEN
    _OWNER_TOKEN = tok
    global _DEFAULT_ONLY
    _DEFAULT_ONLY = not allow
    return allow or default
_OWNER_TOKEN = ""
_DEFAULT_ONLY = False  # True while only the built-in demo project exists (first run)
ALLOW = _load_allow()
import threading as _threading
SESSIONS = {}  # token -> {"groups": set, "login": str, "until": epoch}
_SESSIONS_LOCK = _threading.Lock()
_AUTH_LOCK = _threading.Lock()
_AUTH_FAILS = {}  # ip -> {"fails": [epoch...], "blocked_until": epoch}
_AUTH_WINDOW = 60.0
_AUTH_MAX = 10
_AUTH_BLOCK = 60.0
def _sessions_prune_locked(now):
    for tok in [t for t, s in SESSIONS.items() if not isinstance(s, dict) or s.get("until", 0) <= now]:
        SESSIONS.pop(tok, None)
def _auth_ip_blocked(ip):
    import time as _t
    now = _t.time()
    with _AUTH_LOCK:
        e = _AUTH_FAILS.get(ip)
        if not e:
            return False
        if e.get("blocked_until", 0) > now:
            return True
        fails = [t for t in e.get("fails", []) if now - t < _AUTH_WINDOW]
        if fails:
            e["fails"] = fails
            return False
        _AUTH_FAILS.pop(ip, None)
        return False
def _auth_note(ip, ok):
    import time as _t
    now = _t.time()
    with _AUTH_LOCK:
        if ok:
            _AUTH_FAILS.pop(ip, None)
            return
        e = _AUTH_FAILS.get(ip)
        if e is None:
            e = {"fails": [], "blocked_until": 0.0}
            _AUTH_FAILS[ip] = e
        fails = [t for t in e.get("fails", []) if now - t < _AUTH_WINDOW]
        fails.append(now)
        e["fails"] = fails
        if len(fails) >= _AUTH_MAX and e.get("blocked_until", 0) <= now:
            e["blocked_until"] = now + _AUTH_BLOCK
def _minutes_env(name, default):
    import math
    try:
        value = float(os.environ.get(name, str(default)) or default)
        if value < 0 or not math.isfinite(value):
            raise ValueError()
        return value
    except ValueError:
        return default

SESSION_TTL = _minutes_env("LOCALDOCS_SESSION_MIN", 480) * 60
START_TS = __import__("time").time()
SCHED_MIN = _minutes_env("LOCALDOCS_SCHEDULE_MIN", 0)
_SCHED_STARTED = False
try:
    WATCH_SEC = float(os.environ.get("LOCALDOCS_WATCH_SEC", "0") or 0)
except ValueError:
    WATCH_SEC = 0.0
_WATCHER_STATE = {"enabled": bool(WATCH_SEC > 0), "last_check": None, "last_changes": []}
_WATCHER_STARTED = False
_WATCHER_LOCK = _threading.Lock()
_INDEX_LOCKS = {}
_INDEX_LOCKS_GUARD = _threading.Lock()
def _watcher_log(obj):
    try:
        os.makedirs(os.path.join(DATA, "logs"), exist_ok=True)
        with open(os.path.join(DATA, "logs", "watcher.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except OSError:
        pass
def _with_matched(hits, query):
    import re as _re
    terms = [t.lower() for t in _re.findall(r"\w+", query, flags=_re.UNICODE) if len(t) > 2]
    for h in hits:
        sn = (h.get("snippet") or "").lower()
        h["matched"] = sorted({t for t in terms if t in sn})
    return hits

def _login(body):
    login = (body.get("login") or "")
    password = body.get("password") or ""
    try:
        cfg = json.load(open(_CFG_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    lc = cfg.get("ldap")
    if not lc or not lc.get("host"):
        return {"error": "ldap-not-configured"}
    from project_search.ldap_auth import LDAPAuthenticator
    import secrets as _secrets
    import time as _t
    groups = LDAPAuthenticator.from_config(lc).authenticate(login, password)
    if groups is None:
        return {"error": "bad-credentials"}
    tok = _secrets.token_hex(16)
    with _SESSIONS_LOCK:
        import time as _t2
        _sessions_prune_locked(_t2.time())
        SESSIONS[tok] = {"groups": set(groups), "login": login, "until": _t.time() + SESSION_TTL}
    return {"token": tok, "groups": sorted(groups)}

def _health():
    import time as _t
    try:
        from project_search.backend import list_projects
        projs = {p.get("folder", "?"): {"files": p.get("files"), "chunks": p.get("chunks")}
                 for p in list_projects()}
    except Exception:
        projs = {"error": "unavailable"}
    try:
        _ver = __import__("project_search").__version__
    except Exception:
        _ver = "0.0.0"
    if os.environ.get("LOCALDOCS_PREWARM", "1") == "0":
        _prewarm_state = "disabled"
    elif _PREWARM_DONE:
        _prewarm_state = "done"
    elif _PREWARM_INFLIGHT:
        _prewarm_state = "working"
    else:
        _prewarm_state = "idle"
    return {"version": _ver,
            "resources": {**_SERVER_RESOURCE_PROFILE,
                          "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                          "offline_models": os.environ.get("HF_HUB_OFFLINE") == '1'},
            "uptime_s": int(_t.time() - START_TS), "https": _USE_HTTPS,
            "scheduler_min": SCHED_MIN, "projects": projs,
            "prewarm": _prewarm_state,
            "watcher": {"enabled": bool(_WATCHER_STATE.get("enabled")),
                        "last_check": _WATCHER_STATE.get("last_check"),
                        "last_changes": _WATCHER_STATE.get("last_changes", [])}}

def _backup(project):
    import datetime as _dt
    import zipfile as _zf
    from project_search.backend import location, locked
    _, directory = location(ALLOW[project])
    outdir = os.environ.get("LOCALDOCS_BACKUP_DIR") or os.path.join(DATA, "logs", "backups")
    os.makedirs(outdir, exist_ok=True)
    # Project names are config labels, not safe filesystem paths.
    import secrets
    name = f"index_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}.zip"
    dest = os.path.join(outdir, name)
    temp = dest + ".tmp"
    try:
        # Share the writer lock so all files belong to one published generation.
        with locked(directory):
            if not (directory / "index_meta.json").exists() and not (directory / "shards.json").exists():
                raise ValueError("Project has no index to back up")
            with _zf.ZipFile(temp, "w", _zf.ZIP_DEFLATED) as z:
                for f in sorted(directory.rglob("*")):
                    rel = f.relative_to(directory)
                    if (not f.is_file() or f.is_symlink() or
                            any(p.startswith("pending") or p.endswith((".lock", ".tmp", ".journal"))
                                for p in rel.parts)):
                        continue
                    if f.name in {"meta.sqlite", "lexical.sqlite", "project.json", "shards.json"} or f.name.startswith("index"):
                        z.write(f, rel.as_posix())
        os.replace(temp, dest)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    return {"backup": dest}

def _scheduler_loop():
    import time as _t
    from project_search.backend import index_project
    while True:
        _t.sleep(max(SCHED_MIN, 0.1) * 60)
        for name, folder in list(ALLOW.items()):
            try:
                rep = index_project(folder)
                line = {"project": name, "files": rep["files"], "new": rep["new_embeddings"]}
            except Exception as e:
                line = {"project": name, "error": str(e)[:200]}
            try:
                with open(os.path.join(DATA, "logs", "scheduler.log"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            except OSError:
                pass

def _ensure_scheduler():
    global _SCHED_STARTED
    if SCHED_MIN > 0 and not _SCHED_STARTED:
        _SCHED_STARTED = True
        import threading as _th
        _th.Thread(target=_scheduler_loop, daemon=True).start()

def _fallback_search(folder, query, limit=6):
    """Direct txt scan used when project_search.backend is unavailable."""
    import re as _re
    terms = [t.lower() for t in _re.findall(r"\w+", query or "", flags=_re.UNICODE) if len(t) > 2]
    if not terms:
        return []
    hits = []
    try:
        from pathlib import Path as _P
        files = sorted(_P(str(folder)).rglob("*.txt"))
    except OSError:
        return []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8-sig", errors="replace")
        except (OSError, ValueError):
            continue
        low = text.lower()
        if any(t in low for t in terms):
            idx = min((low.find(t) for t in terms if t in low))
            hits.append({"file": str(p), "snippet": text[max(0, idx - 120):idx + 280],
                         "start_line": text.count("\n", 0, idx) + 1})
            if len(hits) >= limit:
                break
    return hits

def _watcher_reindex(project, folder):
    """Reindex one project via index_queue / index_project_chunked / index_project.
    Never runs two reindexes of the same project in parallel (per-project lock)."""
    lk = _project_index_lock(project)
    if not lk.acquire(blocking=False):
        _watcher_log({"project": project, "skipped": "already-indexing"})
        return {"skipped": "already-indexing"}
    try:
        try:
            from project_search.backend import index_queue as _q  # type: ignore
            put = getattr(_q, "put", None) or getattr(_q, "add", None) or getattr(_q, "enqueue", None)
            if callable(put):
                put((project, str(folder)))
                return {"queued": True}
            if callable(_q):
                _q(project, str(folder))
                return {"queued": True}
        except Exception as e:
            _watcher_log({"project": project, "index_queue_error": str(e)[:200]})
        try:
            from project_search.backend import index_project_chunked as _chunked  # type: ignore
            return _chunked(str(folder))
        except (ImportError, ModuleNotFoundError, AttributeError):
            pass
        except Exception as e:
            _watcher_log({"project": project, "error": str(e)[:300]})
            raise
        try:
            from project_search.backend import index_project as _idx  # type: ignore
            return _idx(str(folder))
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            _watcher_log({"project": project, "note": "no-backend-index: %s" % str(e)[:200]})
            return {"files": -1, "new_embeddings": 0, "note": "no-backend"}
    except Exception as e:
        _watcher_log({"project": project, "error": str(e)[:300]})
        return {"error": str(e)[:200]}
    finally:
        try:
            lk.release()
        except RuntimeError:
            pass

def _watcher_loop():
    import time as _t
    import datetime as _dt
    try:
        from project_search.localdocs import snapshot, watch_once  # noqa: F401 (both required)
    except Exception as e:
        _watcher_log({"error": "no-snapshot: %s" % str(e)[:200]})
        return
    prev = {}
    for _name, _folder in list(ALLOW.items()):
        try:
            prev[_name] = snapshot(str(_folder))
        except Exception as e:
            _watcher_log({"project": _name, "error": "snapshot-init: %s" % str(e)[:200]})
            prev[_name] = {}
    while True:
        _t.sleep(max(WATCH_SEC, 0.1))
        for _name, _folder in list(ALLOW.items()):
            try:
                cur = snapshot(str(_folder))
            except Exception as e:
                _watcher_log({"project": _name, "error": "snapshot: %s" % str(e)[:200]})
                continue
            old = prev.get(_name, {})
            try:
                changed = [k for k in set(cur) | set(old) if cur.get(k) != old.get(k)]
                _ = watch_once  # used for change detection contract (snapshot-based)
            except Exception as e:
                _watcher_log({"project": _name, "error": "diff: %s" % str(e)[:200]})
                continue
            with _WATCHER_LOCK:
                _WATCHER_STATE["last_check"] = _dt.datetime.now().isoformat(timespec="seconds")
            if changed:
                with _WATCHER_LOCK:
                    _WATCHER_STATE["last_changes"] = sorted(changed)[:50]
                try:
                    rep = _watcher_reindex(_name, str(_folder))
                    try:
                        if _settings.get().get("ocr_new_files", True) and _settings.get().get("ocr_when") != "manual":
                            _ocr_all_start([_name], auto=True)  # waits for user idle; the journal skips what was already read
                    except Exception:
                        pass
                    _watcher_log({"project": _name, "changed": len(changed), "rep": str(rep)[:300]})
                except Exception as e:
                    _watcher_log({"project": _name, "error": str(e)[:300]})
                prev[_name] = cur
            else:
                prev[_name] = cur

def _ensure_watcher():
    global _WATCHER_STARTED
    _WATCHER_STATE["enabled"] = bool(WATCH_SEC > 0)
    if WATCH_SEC > 0 and not _WATCHER_STARTED:
        _WATCHER_STARTED = True
        import threading as _th
        _th.Thread(target=_watcher_loop, daemon=True).start()

def _prewarm():
    """Cold-start fix: lazy warmup on first request (was N threads at boot).

    Boot only arms the lazy trigger; the first search/quick request warms the
    embedder and each project index in a bounded pool (max 4 threads).
    Disable with LOCALDOCS_PREWARM=0 (no threads, no work, as before)."""
    # Probe entitlement: _prewarm
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        return
    if os.environ.get("LOCALDOCS_PREWARM", "1") == "0":
        return
    # Lazy: real work happens in _ensure_prewarm() on the first request.
    return


_PREWARM_LOCK = _threading.Lock()
_PREWARM_DONE = False
_PREWARM_INFLIGHT = False


def _ensure_prewarm():
    """Run the deferred warmup once, bounded to 4 threads. No-op when disabled."""
    # Probe entitlement: _ensure_prewarm
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        return
    global _PREWARM_DONE, _PREWARM_INFLIGHT
    if os.environ.get("LOCALDOCS_PREWARM", "1") == "0":
        return
    with _PREWARM_LOCK:
        if _PREWARM_DONE or _PREWARM_INFLIGHT:
            return
        _PREWARM_INFLIGHT = True
    try:
        import concurrent.futures as _cf

        def _model():
            try:
                from project_search.neuro_vector_store import embed_texts
                embed_texts(["warmup"])
            except Exception:
                pass

        def _index(path):
            try:
                from project_search.backend import location, _load_cached
                from project_search.neuro_vector_store import embed_texts
                store = _load_cached(location(path)[1])
                embed_texts(["warmup"], "query", store.model)
            except Exception:
                pass

        paths = [p for p in ALLOW.values() if isinstance(p, str) and p]
        jobs = [_model] + [lambda p=p: _index(p) for p in paths]
        # Bounded pool: at most 4 warmup threads regardless of project count.
        with _cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="localdocs-prewarm") as _ex:
            futs = [_ex.submit(j) for j in jobs]
            for _f in futs:
                try:
                    _f.result()
                except Exception:
                    pass
    finally:
        with _PREWARM_LOCK:
            _PREWARM_INFLIGHT = False
            _PREWARM_DONE = True


_SR_REALPATH_CACHE: dict = {}
_SR_REALPATH_LOCK = _threading.Lock()


def _srv_realpath(path):
    """Cached os.path.realpath for server request paths (roots/folders)."""
    try:
        key = os.path.normcase(os.path.normpath(str(path)))
    except Exception:
        try:
            return os.path.realpath(path)
        except Exception:
            return ""
    with _SR_REALPATH_LOCK:
        hit = _SR_REALPATH_CACHE.get(key)
        if hit is not None:
            return hit
    try:
        real = os.path.realpath(path)
    except Exception:
        return ""
    if not real:
        return ""
    with _SR_REALPATH_LOCK:
        if len(_SR_REALPATH_CACHE) < 8192:
            _SR_REALPATH_CACHE[key] = real
        return _SR_REALPATH_CACHE.get(key, real)


def _srv_norm_real(path):
    try:
        return os.path.normcase(os.path.normpath(_srv_realpath(path)))
    except Exception:
        return ""

import threading as _threading
_INDEX_STATUS = {}
_INDEX_STATUS_LOCK = _threading.RLock()
_CI_STOP = {}
_CI_GEN = {}


def _ci_stop(project):
    """Cooperative stop: set per-project event; safe to call twice."""
    with _INDEX_STATUS_LOCK:
        ev = _CI_STOP.get(project)
        if ev is None:
            ev = _threading.Event()
            _CI_STOP[project] = ev
            _CI_GEN[project] = _CI_GEN.get(project, 0)
    ev.set()


def _jobs_tab(body):
    tab = body.get("tab_id") if isinstance(body, dict) else None
    return tab if isinstance(tab, str) and tab and len(tab) <= 64 else None


def _jobs_lookup(kind, scope=None):
    """Newest registry record for a family (active preferred), or None."""
    try:
        fam = localdocs_jobs.Registry.family_key(kind, scope or {})
    except Exception:
        fam = kind
    try:
        jobs = localdocs_jobs.list_jobs()
    except Exception:
        return None
    best = None
    for job in jobs:
        try:
            if job.get("kind") != kind:
                continue
            if kind == "content-index" and isinstance(scope, dict):
                if (job.get("scope") or {}).get("project") != scope.get("project"):
                    continue
            if localdocs_jobs.Registry.family_key(kind, job.get("scope") or {}) != fam:
                continue
        except Exception:
            continue
        if best is None:
            best = job
        elif job.get("id") == best.get("id"):
            best = job
    # Prefer active (non-terminal) entries which list() returns first.
    return best


def _jobs_legacy_extra(kind, scope=None):
    try:
        job = _jobs_lookup(kind, scope)
    except Exception:
        return {}
    if not job:
        return {}
    return {"job_id": job.get("id"), "job": job}

_QUICK = None
_QUICK_LOCK = _threading.Lock()

def _quick():
    """Lazy background file-name index (Desktop/Documents/Downloads/... + added folders)."""
    global _QUICK
    with _QUICK_LOCK:
        if _QUICK is None:
            import localdocs_quick
            def _roots():
                return localdocs_quick.default_roots([f for f in ALLOW.values() if isinstance(f, str)])
            engine = None
            if os.environ.get("LOCALDOCS_QUICK_ENGINE", "fast") != "scandir":
                try:
                    import localdocs_fastindex
                    engine = localdocs_fastindex.FastQuickIndex(
                        _roots, max_entries=4000000, workers=8,
                        cache_path=os.path.join(DATA, "indexes", "_quick_names.npz"))
                except Exception as exc:
                    _watcher_log({"error": "fast-quick-init: %s" % str(exc)[:200]})
            _QUICK = engine or localdocs_quick.QuickIndex(_roots)
            _QUICK.start()
        return _QUICK


def _filter_root_status(status_list, allowed_roots):
    """Keep only root_status entries at or under one of allowed_roots.

    Both sides are compared as os.path.normcase(os.path.realpath(...))
    with a path-boundary check (never a raw startswith), so a sibling
    such as C:\\data2 is not matched by an allowed root C:\\data.
    Returns [] when either side is empty/missing.
    """
    if not status_list or not allowed_roots:
        return []
    allowed = []
    for root in allowed_roots:
        if not isinstance(root, str) or not root:
            continue
        try:
            allowed.append(os.path.normcase(os.path.realpath(root)))
        except (OSError, ValueError):
            continue
    if not allowed:
        return []
    out = []
    for entry in status_list:
        if not isinstance(entry, dict):
            continue
        root = entry.get("root")
        if not isinstance(root, str) or not root:
            continue
        try:
            norm = os.path.normcase(os.path.realpath(root))
        except (OSError, ValueError):
            continue
        for base in allowed:
            if norm == base or norm.startswith(base.rstrip("\\/") + os.sep):
                out.append(entry)
                break
    return out

_CONFIG_LOCK = _threading.RLock()

_KNOWN_POST = {"/login", "/project/add", "/project/remove", "/pick-folder",
    "/index/start", "/open", "/parse-query", "/ocr-all", "/pick-files",
    "/reveal-path", "/ocr-batch", "/models", "/img-search", "/img-index",
    "/img-tags", "/ocr-journal", "/browse", "/random-file", "/dupes",
    "/jobs", "/file-meta", "/ocr-file", "/settings", "/quick-search",
    "/search_all", "/search", "/index", "/backup", "/file/ticket",
    "/api/prewarm", "/img-list", "/open-path", "/license", "/ask", "/ask/status"}

_FILE_TICKETS = {}
_FILE_TICKETS_LOCK = _threading.Lock()
_FILE_TICKET_TTL = 30.0


def _safe_detail(exc, limit=200):
    """Client-facing error detail without absolute server paths."""
    try:
        s = str(exc)[:limit]
    except Exception:
        s = "error"
    try:
        import re as _re
        s = _re.sub(r"[A-Za-z]:\\[^\s\"'<>]*", "<path>", s)
        s = _re.sub(r"\\\\[^\s\"'<>]+", "<path>", s)
        if ROOT and ROOT in s:
            s = s.replace(ROOT, "<root>")
        s = _re.sub(r"/(?:tmp|home|users|var|etc)/[^\s\"'<>]*", "<path>", s, flags=_re.I)
    except Exception:
        pass
    return s[:limit] or "error"


def _warn_compat(msg):
    try:
        os.makedirs(os.path.join(DATA, "logs"), exist_ok=True)
        import datetime as _dt
        with open(os.path.join(DATA, "logs", "deprecation.log"), "a", encoding="utf-8") as f:
            f.write("%s %s\n" % (_dt.datetime.now().isoformat(timespec="seconds"), msg[:200]))
    except OSError:
        pass


def _issue_file_ticket(token_value):
    import secrets as _secrets
    import time as _t
    now = _t.time()
    ticket = _secrets.token_urlsafe(24)
    with _FILE_TICKETS_LOCK:
        for k in [k for k, v in _FILE_TICKETS.items() if v[1] <= now]:
            _FILE_TICKETS.pop(k, None)
        _FILE_TICKETS[ticket] = (token_value, now + _FILE_TICKET_TTL)
    return ticket


def _redeem_file_ticket(ticket):
    import time as _t
    now = _t.time()
    with _FILE_TICKETS_LOCK:
        entry = _FILE_TICKETS.pop(ticket, None)
        if entry is None:
            return None
        token_value, expiry = entry
        if expiry <= now:
            return None
        return token_value


def _config_read():
    with _CONFIG_LOCK:
        try:
            with open(_CFG_FILE, encoding="utf-8-sig") as stream:
                value = json.load(stream)
        except FileNotFoundError:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("invalid configuration")
        value.setdefault("projects", {})
        return value

def _config_write(cfg):
    with _CONFIG_LOCK:
        temp = _CFG_FILE + ".tmp"
        with open(temp, "w", encoding="utf-8") as stream:
            json.dump(cfg, stream, ensure_ascii=False, indent=2)
            try:
                stream.flush()
                os.fsync(stream.fileno())
            except OSError:
                pass
        os.replace(temp, _CFG_FILE)
        try:
            os.chmod(_CFG_FILE, 0o600)
        except OSError:
            pass

def _add_project(path, name=None):
    import re as _re
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be an existing directory")
    real = os.path.realpath(os.path.expanduser(path.strip()))
    if not os.path.isdir(real):
        raise ValueError("path must be an existing directory")
    drive, tail = os.path.splitdrive(real)
    if not tail.strip("\\/"):
        raise ValueError("drive roots are not allowed")
    protected = [os.environ.get("ProgramFiles", r"C:\\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)"),
                 os.environ.get("WINDIR", r"C:\\Windows")]
    low = os.path.normcase(real)
    if any(low == os.path.normcase(os.path.realpath(p)) or
           low.startswith(os.path.normcase(os.path.realpath(p)).rstrip("\\/") + os.sep) for p in protected):
        raise ValueError("system folders are not allowed")
    label = name if isinstance(name, str) and name.strip() else os.path.basename(real.rstrip("\\/"))
    label = _re.sub(r"[^\w .()-]", "_", label, flags=_re.UNICODE).strip(" .")[:80]
    if not label:
        raise ValueError("invalid project name")
    with _CONFIG_LOCK:
        cfg = _config_read()
        projects = cfg["projects"]
        for existing, value in projects.items():
            old_path = value.get("path") if isinstance(value, dict) else value
            if isinstance(old_path, str):
                resolved = old_path if os.path.isabs(old_path) else os.path.join(ROOT, old_path)
                if os.path.normcase(os.path.realpath(resolved)) == os.path.normcase(real):
                    raise ValueError("folder is already added")
        candidate, i = label, 2
        while candidate in projects:
            candidate = "%s (%d)" % (label, i)
            i += 1
        projects[candidate] = real
        _config_write(cfg)
    global _DEFAULT_ONLY
    if _DEFAULT_ONLY:
        ALLOW.pop('demo', None)  # the built-in demo disappears once the user adds a real folder
        _DEFAULT_ONLY = False
    ALLOW[candidate] = real
    if _QUICK is not None:
        _QUICK.rescan_now()
    _events_sync()
    return candidate

def _remove_project(project):
    if not isinstance(project, str) or project not in ALLOW:
        raise ValueError("unknown project")
    folder = ALLOW[project]
    index_dir = None
    try:
        from project_search.backend import location
        _, index_dir = location(folder)
    except Exception:
        pass
    with _CONFIG_LOCK:
        cfg = _config_read()
        cfg["projects"].pop(project, None)
        _config_write(cfg)
    ALLOW.pop(project, None)
    PROJ_GROUPS.pop(project, None)
    _events_sync()
    try:
        root = os.path.realpath(os.path.join(DATA, "indexes"))
        target = os.path.realpath(str(index_dir))
        if os.path.commonpath([root, target]) == root and os.path.isdir(target):
            import shutil
            shutil.rmtree(target)
    except (ImportError, ValueError, OSError):
        pass

_PICK_CSHARP = r'''
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
[ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")] class FileOpenDialogCOM {}
[ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IShellItem {
 void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
 void GetParent(out IShellItem parent);
 void GetDisplayName(uint sigdn, out IntPtr name);
 void GetAttributes(uint mask, out uint attrs);
 void Compare(IShellItem other, uint hint, out int order);
}
[ComImport, Guid("B63EA76D-1F85-456F-A19C-48159EFA858B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IShellItemArray {
 void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
 void GetPropertyStore(int flags, ref Guid riid, out IntPtr ppv);
 void GetPropertyDescriptionList(IntPtr key, ref Guid riid, out IntPtr ppv);
 void GetAttributes(uint flags, uint mask, out uint attrs);
 void GetCount(out uint count);
 void GetItemAt(uint index, out IShellItem item);
 void EnumItems(out IntPtr items);
}
[ComImport, Guid("D57C7288-D4AD-4768-BE02-9D969532D960"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IFileOpenDialog {
 [PreserveSig] int Show(IntPtr owner);
 void SetFileTypes(uint count, IntPtr filters);
 void SetFileTypeIndex(uint index);
 void GetFileTypeIndex(out uint index);
 void Advise(IntPtr events, out uint cookie);
 void Unadvise(uint cookie);
 void SetOptions(uint options);
 void GetOptions(out uint options);
 void SetDefaultFolder(IShellItem folder);
 void SetFolder(IShellItem folder);
 void GetFolder(out IShellItem folder);
 void GetCurrentSelection(out IShellItem item);
 void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
 void GetFileName(out IntPtr name);
 void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
 void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
 void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
 void GetResult(out IShellItem item);
 void AddPlace(IShellItem item, int alignment);
 void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
 void Close(int hr);
 void SetClientGuid(ref Guid guid);
 void ClearClientData();
 void SetFilter(IntPtr filter);
 void GetResults(out IShellItemArray items);
 void GetSelectedItems(out IShellItemArray items);
}
public static class LocalDocsPicker {
 static string PathOf(IShellItem item) {
  IntPtr value = IntPtr.Zero;
  try { item.GetDisplayName(0x80058000, out value); return Marshal.PtrToStringUni(value); }
  finally { if(value != IntPtr.Zero) Marshal.FreeCoTaskMem(value); Marshal.ReleaseComObject(item); }
 }
 public static string[] Pick(IntPtr owner, bool folder) {
  IFileOpenDialog dialog = (IFileOpenDialog)new FileOpenDialogCOM();
  try {
   uint options; dialog.GetOptions(out options);
   dialog.SetOptions(options | 0x40u | (folder ? 0x20u : 0x200u));
   int hr = dialog.Show(owner);
   if(hr == unchecked((int)0x800704C7)) return new string[0];
   Marshal.ThrowExceptionForHR(hr);
   if(folder) { IShellItem item; dialog.GetResult(out item); return new string[] { PathOf(item) }; }
   IShellItemArray items; dialog.GetResults(out items);
   try {
    uint count; items.GetCount(out count); List<string> paths = new List<string>();
    for(uint i=0; i<count; i++) { IShellItem item; items.GetItemAt(i, out item); paths.Add(PathOf(item)); }
    return paths.ToArray();
   } finally { Marshal.ReleaseComObject(items); }
  } finally { Marshal.ReleaseComObject(dialog); }
 }
}
'''


def _pick_script(folder=True, compile_only=False):
    compile_code = "Add-Type -TypeDefinition @'\n" + _PICK_CSHARP + "\n'@ -ErrorAction Stop\n"
    if compile_only:
        return "$ErrorActionPreference='Stop'\n" + compile_code
    fallback = ("$d=New-Object System.Windows.Forms.FolderBrowserDialog; "
                "if($d.ShowDialog($o) -eq 'OK'){[Console]::Write($d.SelectedPath)}" if folder else
                "$d=New-Object System.Windows.Forms.OpenFileDialog; $d.Multiselect=$true; "
                "$d.Filter='PDF and images|*.pdf;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.webp|All files|*.*'; "
                "if($d.ShowDialog($o) -eq 'OK'){[Console]::Write(($d.FileNames -join [char]10))}")
    return ("$ErrorActionPreference='Stop'; [Console]::OutputEncoding=New-Object System.Text.UTF8Encoding; "
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
            "$o=New-Object System.Windows.Forms.Form; $o.TopMost=$true; $o.ShowInTaskbar=$false; "
            "$o.Opacity=0; $o.Size=New-Object System.Drawing.Size(1,1); $o.Show(); $o.Activate();\n"
            "try { try {\n" + compile_code +
            "$paths=[LocalDocsPicker]::Pick($o.Handle,$" + ('true' if folder else 'false') + "); "
            "[Console]::Write(($paths -join [char]10))\n} catch { " + fallback +
            " } finally { if($d){$d.Dispose()} } } finally { $o.Dispose() }")


class PickerBusy(RuntimeError):
    """A picker dialog is already open; repeated clicks must not stack more dialogs."""


_PICKER_LOCK = _threading.Lock()


def _run_picker(folder):
    if os.name != 'nt':
        raise NotImplementedError()
    import subprocess
    if not _PICKER_LOCK.acquire(blocking=False):
        raise PickerBusy('picker already open')
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-STA', '-NonInteractive',
                                 '-ExecutionPolicy', 'Bypass', '-Command', _pick_script(folder)],
                                capture_output=True, text=True, encoding='utf-8', timeout=300,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
    finally:
        _PICKER_LOCK.release()
    if result.returncode:
        raise RuntimeError('picker dialog failed')
    return [p for p in result.stdout.splitlines() if p]


def _pick_folder():
    paths = _run_picker(True)
    return paths[0] if paths else None

def _project_index_lock(project):
    """One non-reentrant lock per project, shared by every content-index
    entry path (guarded start, watcher reindex, inline OCR run, sync /index).
    Serializes backend writes; the visible phase still lives in exactly one
    place (_try_begin_index / _index_worker completion under _INDEX_STATUS_LOCK)."""
    with _INDEX_LOCKS_GUARD:
        lk = _INDEX_LOCKS.get(project)
        if lk is None:
            lk = _threading.Lock()
            _INDEX_LOCKS[project] = lk
        return lk


def _try_begin_index(project):
    """Single place where a content-index run claims the running phase.

    Returns (gen, ev) on success, None when another run holds the project.
    Holds no lock on return; the caller must hand gen to _index_worker.
    """
    import datetime as _dt
    with _INDEX_STATUS_LOCK:
        if _INDEX_STATUS.get(project, {}).get("phase") == "running":
            return None
        _INDEX_STATUS[project] = {"phase": "running", "files": 0, "chunks": 0, "current": "",
            "error": "", "started_at": _dt.datetime.now().isoformat(timespec="seconds"), "finished_at": None}
        gen = _CI_GEN.get(project, 0) + 1
        _CI_GEN[project] = gen
        ev = _threading.Event()
        _CI_STOP[project] = ev
        return (gen, ev)


def _index_worker(project, folder, gen=None):
    import datetime as _dt
    if os.name == "nt":
        try:  # background indexing must never starve search or the UI
            import ctypes as _ct
            _k = _ct.windll.kernel32
            _k.GetCurrentThread.restype = _ct.c_void_p
            _k.SetThreadPriority.argtypes = [_ct.c_void_p, _ct.c_int]
            _k.SetThreadPriority(_k.GetCurrentThread(), -1)
        except Exception:
            pass
    def set_status(**fields):
        with _INDEX_STATUS_LOCK:
            if _CI_GEN.get(project, 0) != gen:
                return  # superseded: never overwrite the newer run's phase
            _INDEX_STATUS[project].update(fields)
    # Serialize backend writes for this project (watcher/sync/inline paths
    # share this lock; the watcher uses try-acquire and skips when held).
    with _project_index_lock(project):
        with _INDEX_STATUS_LOCK:
            ev = _CI_STOP.get(project)
            if ev is None:
                ev = _threading.Event()
                _CI_STOP[project] = ev
            if gen is None:
                # Legacy direct call without a claimed begin: adopt current gen.
                gen = _CI_GEN.get(project, 0)
        def _stopped():
            if ev.is_set():
                return True
            with _INDEX_STATUS_LOCK:
                return _CI_GEN.get(project, 0) != gen
        try:
            from project_search.backend import index_project_chunked
            result = index_project_chunked(folder, group_files=2000, should_stop=_stopped)
            if isinstance(result, dict) and result.get("stopped"):
                set_status(phase="interrupted", files=result.get("files", 0),
                           chunks=result.get("chunks", result.get("new_embeddings", 0)),
                           current="", finished_at=_dt.datetime.now().isoformat(timespec="seconds"), error="")
            else:
                if _stopped():
                    set_status(phase="interrupted", files=result.get("files", 0), chunks=result.get("chunks", result.get("new_embeddings", 0)),
                               current="", finished_at=_dt.datetime.now().isoformat(timespec="seconds"), error="")
                else:
                    set_status(phase="done", files=result.get("files", 0), chunks=result.get("chunks", result.get("new_embeddings", 0)),
                               current="", finished_at=_dt.datetime.now().isoformat(timespec="seconds"), error="")
        except Exception:
            if _stopped():
                set_status(phase="interrupted", current="", error="",
                           finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
            else:
                set_status(phase="error", current="", error="indexing failed",
                           finished_at=_dt.datetime.now().isoformat(timespec="seconds"))

def _start_index(project):
    if project not in ALLOW:
        return False
    beg = _try_begin_index(project)
    if beg is None:
        return False
    gen, _ev = beg
    _threading.Thread(target=_index_worker, args=(project, ALLOW[project], gen), daemon=True).start()
    return True

_EVENTS = None
_EVENTS_LOCK = _threading.Lock()

def _events_sync():
    """Event-driven indexing: watch every project folder and re-index only after real changes (no polling)."""
    global _EVENTS
    if os.environ.get("LOCALDOCS_EVENT_WATCH", "1") == "0":
        return None
    with _EVENTS_LOCK:
        if _EVENTS is None:
            try:
                import localdocs_events
                from project_search.backend import EXT as _EXT, SKIP as _SKIP
                _EVENTS = localdocs_events.EventIndexer(
                    lambda: dict(ALLOW), _start_index, _EXT, skip_dirs=_SKIP,
                    ignore_prefixes=[os.path.join(DATA, "indexes"), os.path.join(DATA, "logs")],
                    quiet_s=float(os.environ.get("LOCALDOCS_EVENT_QUIET_S", "20") or 20),
                    log=_watcher_log)
            except Exception as exc:
                _watcher_log({"error": "events-init: %s" % str(exc)[:200]})
                return None
        events = _EVENTS
    return events.sync()

def _pick_files():
    return _run_picker(False)


def _uploads_dir():
    d = os.environ.get("LOCALDOCS_UPLOADS_DIR") or os.path.join(DATA, "indexes", "_uploads")
    os.makedirs(d, exist_ok=True)
    return d


def _reveal_path(path):
    import subprocess as _sp
    if os.name != "nt":
        raise NotImplementedError()
    target = os.path.realpath(path)
    if not os.path.exists(target):
        raise ValueError("not found")
    _sp.Popen(['explorer.exe', '/select,"%s"' % target], close_fds=True)
    return {"ok": True}


_IMG_JOB = {"phase": "idle", "done": 0, "total": 0, "current": "", "error": None}
_IMG_STOP = _threading.Event()
_IMG_ENGINE = {"obj": None, "device": None}
_IMG_LOCK = _threading.Lock()
# Generation: incremented on every start while holding _IMG_LOCK. The worker
# captures its gen; completion updates apply only when the gen is still
# current, so a stale worker can never overwrite a newer run. The stop event
# is cleared on start and set on stop, both under _IMG_LOCK, so a stop only
# ever targets the generation that is current at that moment (_CI_GEN pattern).
_IMG_GEN = 0
_IMG_THREAD = None


def _img_store(name):
    d = os.path.join(DATA, "indexes", "_img")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "".join(c if c.isalnum() or c in "-_" else "_" for c in name) + ".npz")


def _img_project_store(name):
    return _img_store('_project_custom' if name == 'custom' else name)


def _img_engine():
    import localdocs_imgsearch
    device = _settings.get().get("image_device", "cpu")
    with _IMG_LOCK:
        if _IMG_ENGINE["obj"] is None or _IMG_ENGINE["device"] != device:
            _IMG_ENGINE["obj"] = localdocs_imgsearch.ImageSearch(device=device)
            _IMG_ENGINE["device"] = device
        return _IMG_ENGINE["obj"]


def _img_gen_stale(gen):
    with _IMG_LOCK:
        return gen is not None and gen != _IMG_GEN


def _img_update(gen, **fields):
    """Locked phase/progress update; applies only for the current generation."""
    with _IMG_LOCK:
        if gen is not None and gen != _IMG_GEN:
            return False
        _IMG_JOB.update(fields)
        return True


def _img_index_worker(paths=None, gen=None):
    try:
        eng = _img_engine()
        if paths is not None:
            def custom_progress(info):
                _img_update(gen, done=int(info.get('done') or 0), total=int(info.get('total') or 0), current='custom')
            eng.index_paths(paths, _img_store('custom'),
                            should_stop=lambda: _IMG_STOP.is_set() or _img_gen_stale(gen),
                            progress=custom_progress)
            _img_update(gen, phase='done')
            return
        projects = list(ALLOW.items())
        for n, (name, folder) in enumerate(projects):
            if _IMG_STOP.is_set() or _img_gen_stale(gen) or not os.path.isdir(folder):
                continue

            def prog(info):
                _img_update(gen, done=int(info.get("done") or 0), total=int(info.get("total") or 0), current=name)
            eng.index_folder(folder, _img_project_store(name),
                             should_stop=lambda: _IMG_STOP.is_set() or _img_gen_stale(gen),
                             progress=prog)
        _img_update(gen, phase="done")
    except Exception as e:
        _img_update(gen, phase="idle" if _IMG_STOP.is_set() else "error", error=None if _IMG_STOP.is_set() else str(e)[:200])


def _img_index(action, paths=None):
    global _IMG_GEN, _IMG_THREAD
    if action == "start":
        if paths is not None:
            paths = _image_selection(paths)
        with _IMG_LOCK:
            if _IMG_JOB["phase"] == "running":
                return dict(_IMG_JOB)
            _IMG_GEN += 1
            gen = _IMG_GEN
            _IMG_STOP.clear()
            _IMG_JOB.update(phase="running", done=0, total=0, current="", error=None)
            _t = _threading.Thread(target=_img_index_worker, args=(paths, gen), name="localdocs-img-index", daemon=True)
            _IMG_THREAD = _t
        _t.start()
        return dict(_IMG_JOB)
    elif action == "stop":
        with _IMG_LOCK:
            if _IMG_JOB.get("phase") == "running":
                _IMG_STOP.set()
        return dict(_IMG_JOB)
    return dict(_IMG_JOB)


def _jobs_join_workers(timeout=2.0):
    """Best-effort join of tracked job worker threads (daemon fallback stays)."""
    try:
        th = _IMG_THREAD
    except NameError:
        th = None
    if th is not None and th.is_alive():
        try:
            th.join(timeout=timeout)
        except Exception:
            pass


try:
    import atexit as _atexit

    _atexit.register(_jobs_join_workers)
except Exception:
    pass


def _image_selection(paths, limit=None):
    import localdocs_imgsearch as I
    if not isinstance(paths, list) or not paths:
        raise ValueError('paths must be a nonempty list of absolute files or folders')
    checked = [_system_path(p) for p in paths]
    # A start job validates roots only; recursion belongs to the background worker.
    return [str(p) for p in I.image_paths(checked, limit)] if limit is not None else checked


def _img_stores():
    # Reserve custom.npz for selections even if a project happens to be named custom.
    return [(n, _img_project_store(n)) for n in ALLOW if os.path.isfile(_img_project_store(n))] + (
        [('', _img_store('custom'))] if os.path.isfile(_img_store('custom')) else [])


def _img_result(name, hit):
    full = os.path.abspath(hit['path'])
    st = os.stat(full)
    if not name:
        _system_preview(full)
        # Custom selections inside a project retain its relative UI contract.
        candidates = []
        for project, directory in ALLOW.items():
            root = os.path.realpath(directory)
            try:
                if os.path.commonpath([root, os.path.realpath(full)]) == root:
                    candidates.append((len(root), project))
            except ValueError:
                continue
        if candidates:
            name = max(candidates)[1]
    if name:
        root = os.path.realpath(ALLOW[name])
        if os.path.commonpath([root, os.path.realpath(full)]) != root:
            return None
        path = os.path.relpath(full, root).replace(os.sep, '/')
    else:
        path = full
    out = dict(hit, path=path, project=name, name=os.path.basename(full), size=st.st_size, mtime=st.st_mtime)
    if not name:
        out['abs'] = full
    return out


def _img_search(body):
    import localdocs_imgsearch
    _ensure_prewarm()
    stores = _img_stores()
    if not stores:
        return {"ready": True, "reason": "no-index", "results": [], "device_used": ""}
    limit = max(1, min(int(body.get("limit", 60)), 200))
    eng = _img_engine()
    _warn = ''
    # Single query vector for all stores (was one embed per store in the loop).
    import concurrent.futures as _cf
    if body.get("mode") == "image":
        _rawq = str(body.get("path", ""))
        _proj = str(body.get("project", ""))
        if _proj:
            if _proj not in ALLOW:
                raise PermissionError("unknown project")
            _root = os.path.realpath(ALLOW[_proj])
            _full = os.path.realpath(os.path.join(_root, _rawq))
            try:
                if os.path.commonpath([_root, _full]) != _root:
                    raise PermissionError("outside project")
            except ValueError:
                raise PermissionError("outside project")
            qpath = _system_path(_full)
        else:
            qpath = _system_path(_rawq)
        try:
            if os.path.getsize(qpath) > 200 * 1024 * 1024:
                raise OverflowError("query image exceeds 200 MB")
        except OSError:
            pass
        ready = localdocs_imgsearch.available()
        if not ready.get("ready"):
            return {"ready": False, "reason": ready.get("reason", ""), "results": [], "device_used": ""}
        _vecs, _mask = eng.embed_images([qpath])
        if not _mask[0]:
            raise ValueError("Cannot decode query image: " + qpath)
        _qvec = _vecs[0]
        _exclude = os.path.normcase(os.path.abspath(qpath))
        _jobs = [(n, s, None) for n, s in stores]
        def _one(job):
            n, s, _ = job
            return (n, eng._search(s, _qvec, limit, None, _exclude))
    else:
        _query = str(body.get("query", ""))
        try:
            _text, _tags = localdocs_imgsearch.parse_photo_query(_query)
        except Exception:
            _text, _tags = _query, []
        _tag_mode = "all" if str(body.get("tag_mode", body.get("mode") if body.get("mode") in ("any", "all") else "any")) == "all" else "any"
        _tag_filter = str(body.get("tag_filter", "all"))
        if _tag_filter not in ("all", "with", "without"):
            raise ValueError("tag_filter must be all/with/without")
        if _tags or _tag_filter != "all":
            # Reuse the tested per-store orchestration (parse/merge/reasons live
            # in localdocs_imgsearch.search_combined); the server only fans out
            # over stores and maps absolute hits to project-relative results.
            # Read-only: search_combined never creates indexes.
            _per, _reasons = [], []
            for _n, _s in stores:
                try:
                    _r = localdocs_imgsearch.search_combined(
                        eng, _s, _query, limit, _tag_mode, _tag_filter,
                        available_fn=localdocs_imgsearch.available)
                except (OSError, ValueError) as _e:
                    _reasons.append("no-match")
                    continue
                if _r.get("reason") == "no-model":
                    return {"ready": False, "reason": _r.get("detail", ""),
                            "results": [], "device_used": ""}
                _reasons.append(_r.get("reason", ""))
                for _h in _r.get("results", []) or []:
                    _per.append((_n, _h))
            _per.sort(key=lambda nh: -float(nh[1].get("score", 0)))
            _seen, _out = set(), []
            for _n, _h in _per:
                if len(_out) >= limit:
                    break
                _k = os.path.normcase(os.path.normpath(_h.get("path", "")))
                if not _k or _k in _seen:
                    continue
                try:
                    _rr = _img_result(_n, {"path": _h.get("path", ""), "score": _h.get("score", 0)})
                except (OSError, ValueError, OverflowError):
                    continue
                if not _rr:
                    continue
                _seen.add(_k)
                _rr["source"], _rr["tags"] = _h.get("source", "tag"), _h.get("tags", [])
                _out.append(_rr)
            _tw = ''
            try:
                _ti = localdocs_imgsearch.active_text_info()
                if localdocs_imgsearch.needs_multilingual_text(_query) and not _ti.get("multilingual"):
                    _tw = 'multilingual-suggested'
            except Exception:
                _tw = ''
            if _out:
                return {"ready": True, "reason": "", "device_used": eng.device_used or "", "results": _out, "text_warning": _tw}
            for _cand in ("filtered", "no-tag", "no-photos", "empty-query"):
                if _cand in _reasons:
                    _detail = ", ".join(_tags) if _cand == "no-tag" else ""
                    return {"ready": True, "reason": _cand, "detail": _detail,
                            "device_used": eng.device_used or "", "results": [], "text_warning": _tw}
            return {"ready": True, "reason": "no-match", "device_used": eng.device_used or "", "results": [], "text_warning": _tw}
        if not _text and not _tags:
            return {"ready": True, "reason": "empty-query", "device_used": eng.device_used or "", "results": []}
        ready = localdocs_imgsearch.available()
        if not ready.get("ready"):
            return {"ready": False, "reason": ready.get("reason", ""), "results": [], "device_used": ""}
        _info = localdocs_imgsearch.active_text_info()
        _warn = 'multilingual-suggested' if (localdocs_imgsearch.needs_multilingual_text(_text or _query) and not _info.get("multilingual")) else ''
        try:
            _qvec = eng.embed_text([_text or _query])[0]
        except ValueError as _e:
            return {"ready": False, "reason": "bad-text-dim", "detail": str(_e),
                    "device_used": eng.device_used or "", "results": [], "text_warning": _warn}
        def _one(job):
            n, s = job
            return (n, eng._search(s, _qvec, limit, None))
        _jobs = list(stores)
    # Fan-out over stores (max 8 threads), results merged in deterministic order.
    _raw = []
    _first_err = None
    with _cf.ThreadPoolExecutor(max_workers=min(8, len(_jobs))) as _ex:
        _futs_ordered = [_ex.submit(_one, job) for job in _jobs]
        for job, _f in zip(_jobs, _futs_ordered):
            try:
                n, hits = _f.result()
            except (OSError, ValueError) as _e:
                if _first_err is None:
                    _first_err = _e
                continue
            for h in hits:
                _raw.append((n, h))
    if not _raw and _first_err is not None:
        raise _first_err
    _raw.sort(key=lambda nh: -nh[1].get("score", 0))
    # os.stat only for the rows we will actually return (was stat per store-hit).
    out = []
    for n, h in _raw:
        if len(out) >= limit:
            break
        try:
            result = _img_result(n, h)
        except (OSError, ValueError, OverflowError):
            continue
        if result:
            out.append(result)
    return {"ready": True, "reason": "", "device_used": eng.device_used or "", "results": out, "text_warning": _warn}


def _img_tags_api(body):
    import localdocs_tags as T
    a = body.get("action", "counts")
    stores = _img_stores()
    if a == 'import':
        return T.fetch_tags(body, top=body.get('top', 1000))
    if a == "vocab":
        return {"builtin": list(T.BUILTIN)}
    if a == "parse":
        parsed = T.parse_tags(str(body.get("text", "")), top=None)
        return {"tags": parsed[:5000], "truncated": len(parsed) > 5000}
    if a in ('run', 'run_paths'):
        selected = _image_selection(body.get('paths')) if a == 'run_paths' else None
        import localdocs_imgsearch
        if not localdocs_imgsearch.available().get("ready"):
            return {"ready": False, "error": "model"}
        if not stores and a == 'run':
            return {"ready": True, "error": "no-index"}
        tags = T.parse_tags(str(body.get("text", "")), top=None) if body.get("text") else [str(x) for x in (body.get("tags") or [])]
        if body.get("use") == "builtin":
            tags = list(T.BUILTIN) + [t for t in tags if t not in T.BUILTIN]
        if len(tags) > 5000:
            raise ValueError("The tag list exceeds 5000 entries; choose a smaller list")
        per = max(1, min(int(body.get("per_image", 3)), 20))
        mn = max(0.0, min(float(body.get("min_score", 0.2)), 0.9))
        eng = _img_engine()
        total = {"images": 0, "tagged": 0, "tags": len(tags)}
        text_mods, text_warns = set(), set()
        if a == 'run_paths':
            if not tags:
                raise ValueError('no tags')
            if selected:
                state = eng.index_paths(selected, _img_store('custom'))
                r_paths = T.tag_store(eng, _img_store('custom'), tags, per, mn, only_paths=selected)
                total.update({k: r_paths.get(k, total.get(k)) for k in ("images", "tagged", "tags")})
                text_mods.add(str(r_paths.get("text_module", "")))
                if r_paths.get("text_warning"):
                    text_warns.add(str(r_paths.get("text_warning")))
                total['indexed'] = state['indexed'] - state['reused']
            else:
                total['indexed'] = 0
            total["text_module"] = sorted(m for m in text_mods if m)[:4]
            total["text_warning"] = sorted(text_warns)[:4]
            return dict(total, ready=True)
        only = body.get("project")
        for name, store in stores:
            if only is not None and name != only:
                continue
            folder = body.get("folder") or None
            if folder and name:
                folder = os.path.realpath(str(folder))
                try:
                    if os.path.commonpath([folder, os.path.realpath(ALLOW[name])]) != os.path.realpath(ALLOW[name]):
                        continue
                except ValueError:
                    continue
            r = T.tag_store(eng, store, tags, per, mn, folder)
            total["images"] += r["images"]
            total["tagged"] += r["tagged"]
            text_mods.add(str(r.get("text_module", "")))
            if r.get("text_warning"):
                text_warns.add(str(r.get("text_warning")))
        total["text_module"] = sorted(m for m in text_mods if m)[:4]
        total["text_warning"] = sorted(text_warns)[:4]
        return dict(total, ready=True)
    if a == "counts":
        merged = {}
        for name, store in stores:
            for t, c in T.tag_counts(store):
                merged[t] = merged.get(t, 0) + c
        return {"tags": sorted(([t, c] for t, c in merged.items()), key=lambda x: -x[1])}
    if a == "search":
        wanted = [str(x) for x in (body.get("tags") or [])]
        out = []
        for name, store in stores:
            for h in T.search_tags(store, wanted, "all" if body.get("mode") == "all" else "any", 300):
                try:
                    result = _img_result(name, h)
                except (OSError, ValueError, OverflowError):
                    continue
                if result:
                    out.append(result)
        out.sort(key=lambda r: -r["score"])
        return {"ready": True, "results": out[:300], "device_used": ""}
    if a == "of":
        name = str(body.get("project", ""))
        if name == '':
            full = os.path.normcase(_system_path(body.get('path', '')))
        elif name in ALLOW:
            full = os.path.normcase(os.path.realpath(os.path.join(ALLOW[name], str(body.get("path", "")))))
            root = os.path.normcase(os.path.realpath(ALLOW[name]))
            if os.path.commonpath([root, full]) != root:
                raise PermissionError('outside project')
        else:
            return {'tags': []}
        picked_tags = []
        for project, store in stores:
            if project not in ('', name):
                continue
            for p, picked in T.load(store).get('assign', {}).items():
                if os.path.normcase(os.path.realpath(p)) == full:
                    picked_tags.extend(t for t, _ in picked)
        return {'tags': list(dict.fromkeys(picked_tags))}
    raise ValueError("unknown action")


def _img_list(body):
    """Indexed photos for the gallery (newest first); paging keeps the payload small."""
    import numpy as _np
    import localdocs_tags as T
    flt = str(body.get("filter", "all"))
    if flt not in ("all", "with", "without"):
        raise ValueError("filter must be all/with/without")
    offset = max(0, int(body.get("offset", 0)))
    limit = max(1, min(int(body.get("limit", 120)), 400))
    rows, seen = [], set()
    for name, store in _img_stores():
        try:
            with _np.load(store, allow_pickle=False) as data:
                paths = data['paths'].astype(str).tolist()
        except (OSError, ValueError, KeyError):
            continue
        assign = {os.path.normcase(os.path.realpath(p)): [t for t, _ in picked]
                  for p, picked in T.load(store).get("assign", {}).items()}
        for p in paths:
            key = os.path.normcase(os.path.realpath(p))
            if key in seen:
                continue
            tags = assign.get(key, [])
            if (flt == "with" and not tags) or (flt == "without" and tags):
                continue
            # Only photos that still exist inside their search folder count (stale entries are skipped).
            try:
                result = _img_result(name, {"path": p, "score": 0.0})
            except (OSError, ValueError, OverflowError):
                continue
            if not result:
                continue
            seen.add(key)
            result["tags"] = tags
            rows.append(result)
    rows.sort(key=lambda r: -float(r.get("mtime") or 0))
    return {"total": len(rows), "offset": offset, "results": rows[offset:offset + limit]}


_THUMB_LOCK = _threading.Lock()


def _thumb_bytes(full, width):
    """JPEG thumbnail cached on disk by path+size+mtime, so galleries never ship originals."""
    import hashlib
    st = os.stat(full)
    if st.st_size > 200 * 1024 * 1024:
        raise OverflowError('image exceeds 200 MB')
    key = hashlib.sha1(("%s|%d|%d|%d" % (os.path.normcase(full), st.st_size,
                                         int(st.st_mtime), width)).encode("utf-8")).hexdigest()
    folder = os.path.join(DATA, "indexes", "_thumbs", key[:2])
    target = os.path.join(folder, key + ".jpg")
    try:
        with open(target, "rb") as cached:
            return cached.read()
    except OSError:
        pass
    from PIL import Image, ImageOps
    with Image.open(full) as image:
        image.draft("RGB", (width * 2, width * 2))
        image = ImageOps.exif_transpose(image)
        image.thumbnail((width, width))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        import io as _io
        buffer = _io.BytesIO()
        image.save(buffer, "JPEG", quality=82, optimize=True)
        data = buffer.getvalue()
    with _THUMB_LOCK:
        try:
            os.makedirs(folder, exist_ok=True)
            tmp = target + ".%d.tmp" % _threading.get_ident()
            with open(tmp, "wb") as out:
                out.write(data)
            os.replace(tmp, target)
        except OSError:
            pass
    return data


_RESTORE_PS = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$want = [Console]::In.ReadToEnd() | ConvertFrom-Json
$bin = (New-Object -ComObject Shell.Application).Namespace(10)
$items = @($bin.Items())
$done = @()
foreach ($w in $want) {
  $best = $null; $bestDate = [datetime]::MinValue
  foreach ($i in $items) {
    $from = [string]$i.ExtendedProperty('System.Recycle.DeletedFrom')
    if ($from.TrimEnd('\') -ine $w.dir.TrimEnd('\')) { continue }
    $nm = [string]$i.Name
    $leaf = [System.IO.Path]::GetFileName([string]$i.Path)
    if ($nm -ine $w.name -and $nm -ine [System.IO.Path]::GetFileNameWithoutExtension($w.name)) { continue }
    $when = $i.ExtendedProperty('System.Recycle.DateDeleted')
    if ($when -ne $null -and [datetime]$when -gt $bestDate) { $best = $i; $bestDate = [datetime]$when }
    elseif ($best -eq $null) { $best = $i }
  }
  if ($best -ne $null) { $best.InvokeVerb('undelete'); $done += $w.path }
}
[Console]::Write(($done | ConvertTo-Json -Compress))
'''


def _restore_recycled(paths):
    """Bring items back from the Windows Recycle Bin to their original place (best effort, verified)."""
    if os.name != "nt":
        raise NotImplementedError()
    import subprocess
    want = [{"path": p, "dir": os.path.dirname(p), "name": os.path.basename(p)} for p in paths if not os.path.exists(p)]
    if want:
        subprocess.run(['powershell.exe', '-NoProfile', '-STA', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                        '-Command', _RESTORE_PS], input=json.dumps(want), capture_output=True, text=True,
                       encoding='utf-8', timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
    return [p for p in paths if os.path.exists(p)]


def _open_system_path(path, reveal=False):
    """Open (or reveal) a local file found by the name index; same path rules as system browsing."""
    if os.name != "nt":
        raise NotImplementedError()
    target = _system_path(str(path))
    if not os.path.exists(target):
        raise ValueError("not found")
    if reveal:
        return _reveal_path(target)
    os.startfile(target)
    return {"ok": True}


def _ocr_journal_api(body):
    import localdocs_ocr_journal as J
    a = body.get("action", "stats")
    if a == "recent":
        st = body.get("status")
        return {"rows": J.recent(max(1, min(int(body.get("limit", 200)), 1000)), st if st in ("ok", "empty", "error", "skipped", "uncertain") else None,
                                 str(body.get("query") or "") or None)}
    if a == "stats":
        return J.stats()
    if a == "forget":
        J.forget(str(body["path"]) if body.get("path") else None)
        return J.stats()
    if a == "cleanup":
        return {"removed": J.cleanup_missing(int(body.get("limit", 5000))), "stats": J.stats()}
    raise ValueError("unknown action")


def _models_api(body):
    import localdocs_models as M
    a = body.get("action", "list")
    if a == "list":
        rows = []
        for e in M.list_models():
            if e.get('internal'): continue
            st = e.get("status", {})
            rows.append({"id": e["id"], "kind": e.get("kind"), "supported_kinds": e.get("supported_kinds", [e.get("kind")]),
                         "title": e.get("title"), "description": e.get("description"), "license": e.get("license"),
                         "commercial_ok": e.get("commercial_ok") is True, "size_mb": round(e.get("size_mb") or st.get("size_mb", 0)),
                         "source": e.get("source"), "installed": bool(st.get("installed")), "active": bool(e.get("active")),
                         "device": e.get("device", ["cpu", "gpu"]), "verified": bool(e.get("verified"))})
        return {"models": rows, "download": M.download_status()}
    if a == "download":
        return {"started": M.start_download(str(body.get("id", ""))), "download": M.download_status()}
    if a == "download_status":
        return M.download_status()
    if a == "set_active":
        M.set_active(str(body.get("kind", "")), str(body.get("id", "")))
        return {"ok": True}
    if a == "add_custom":
        return {"id": M.add_custom(str(body.get("kind", "")), str(body.get("path", "")), str(body.get("title", "") or "custom"))}
    if a == "remove":
        M.remove_custom(str(body.get("id", "")))
        return {"ok": True}
    if a == "update_state":
        return M.update_state()
    if a == "check_update":
        return M.check_updates(str(body.get("id", "")))
    raise ValueError("unknown action")


def _file_meta(project, filename):
    """Detailed metadata of one file inside an allowed project (same containment rule as _open_file)."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("file is required")
    if not _settings.get().get("detailed_metadata", False):
        return {"enabled": False}
    root = os.path.realpath(ALLOW[project])
    target = os.path.realpath(filename if os.path.isabs(filename) else os.path.join(root, filename))
    try:
        if os.path.commonpath([root, target]) != root:
            raise PermissionError()
    except ValueError:
        raise PermissionError()
    if not os.path.isfile(target):
        raise ValueError("file not found")
    import localdocs_meta
    out = localdocs_meta.file_meta(target, detailed=True)
    out["enabled"] = True
    return out

def _ocr_file(project, filename, mode):
    """OCR one file inside an allowed project. mode 'text' returns the text; 'pdf' writes <name>.ocr.pdf next to it
    (original untouched) with an invisible text layer, so the PDF becomes searchable."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("file is required")
    if mode not in ("text", "pdf"):
        raise ValueError("mode must be text or pdf")
    root = os.path.realpath(ALLOW[project])
    target = os.path.realpath(filename if os.path.isabs(filename) else os.path.join(root, filename))
    try:
        if os.path.commonpath([root, target]) != root:
            raise PermissionError()
    except ValueError:
        raise PermissionError()
    if not os.path.isfile(target):
        raise ValueError("file not found")
    cfg = _settings.get()
    engine = "own" if cfg.get("ocr_engine") == "own" else "auto"
    device = "gpu" if cfg.get("ocr_gpu") else "cpu"
    lang = cfg.get("ocr_lang") or "rus+eng+pol"
    try:
        rotation = int(cfg.get("ocr_rotation") or 0) % 360
    except (TypeError, ValueError):
        rotation = 0
    if rotation not in (0, 90, 180, 270):
        rotation = 0
    spreads = bool(cfg.get("ocr_spreads"))
    ext = os.path.splitext(target)[1].lower()
    import localdocs_ocr
    if mode == "pdf":
        if ext != ".pdf":
            raise ValueError("searchable PDF needs a .pdf file")
        import localdocs_pdf_ocr
        dst = os.path.splitext(target)[0] + ".ocr.pdf"
        n = 1
        while os.path.exists(dst):
            n += 1
            dst = os.path.splitext(target)[0] + ".ocr (%d).pdf" % n
        info = localdocs_pdf_ocr.make_searchable_pdf(target, dst, lang=lang, device=device,
                                                      engine=engine, rotation=rotation or None,
                                                      spreads=spreads)
        info["output"] = dst
        return info
    if ext == ".pdf":
        import pypdfium2 as pdfium
        texts = []
        doc = pdfium.PdfDocument(target)
        try:
            for i in range(min(len(doc), 50)):
                page = doc[i]
                try:
                    bitmap = page.render(scale=150 / 72)
                    image = bitmap.to_pil()
                    texts.append(localdocs_ocr.ocr_image(image, lang, device, engine, rotation=rotation or None, spreads=spreads)["text"])
                finally:
                    page.close()
        finally:
            doc.close()
        return {"text": (chr(10) * 2).join(texts), "pages": len(texts), "engine": engine}
    res = localdocs_ocr.ocr_image(target, lang, device, engine,
                                  rotation=rotation or None, spreads=spreads)
    return {"text": res.get("text", ""), "engine": res.get("engine", engine), "seconds": res.get("seconds")}

# ---- on-demand OCR of everything (first content search, or a button/installer choice); results are cached by file hash ----
_OCR_ALL = {"phase": "idle", "project": "", "done": 0, "total": 0, "started_at": None, "finished_at": None, "stop": False, "auto": False, "error": ""}
_OCR_ALL_LOCK = _threading.Lock()
# Cooperative stop event: the single source of truth for should_stop checks.
# The dict ["stop"] flag is mirrored under _OCR_ALL_LOCK for status compat.
_OCR_ALL_STOP = _threading.Event()


def _ocr_marker():
    return os.path.join(DATA, "indexes", "_ocr_all.json")


def _ocr_all_status():
    with _OCR_ALL_LOCK:
        return dict(_OCR_ALL)


def _user_idle_seconds():
    """Seconds since the last keyboard/mouse input (Windows); 0 if unknown."""
    if os.name != "nt":
        return 0.0
    try:
        import ctypes as _ct
        class _LII(_ct.Structure):
            _fields_ = [("cbSize", _ct.c_uint), ("dwTime", _ct.c_uint)]
        info = _LII(_ct.sizeof(_LII), 0)
        if not _ct.windll.user32.GetLastInputInfo(_ct.byref(info)):
            return 0.0
        return max(0.0, (_ct.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0)
    except Exception:
        return 0.0


def _ocr_all_worker(names):
    import datetime as _dt
    import time as _time
    import localdocs_settings as _st
    _st.FORCE_OCR.set()
    _failed = ""
    try:
        for n in names:
            if _OCR_ALL_STOP.is_set() or n not in ALLOW:
                break
            while True:  # automatic runs only while the user is away from keyboard and mouse
                with _OCR_ALL_LOCK:
                    auto = _OCR_ALL["auto"]
                if not auto or _OCR_ALL_STOP.is_set() or _user_idle_seconds() >= 30:
                    break
                _time.sleep(5)
            beg = None
            while True:  # never overlap an indexing run of the same project
                if _OCR_ALL_STOP.is_set():
                    break
                beg = _try_begin_index(n)
                if beg is not None:
                    break
                _time.sleep(1)
            if beg is None:
                break
            with _OCR_ALL_LOCK:
                _OCR_ALL["project"] = n
            _index_worker(n, ALLOW[n], beg[0])
            with _OCR_ALL_LOCK:
                _OCR_ALL["done"] += 1
    except Exception as exc:
        try:
            _failed = str(exc)[:200] or "error"
        except Exception:
            _failed = "error"
    finally:
        _st.FORCE_OCR.clear()
        with _OCR_ALL_LOCK:
            if _failed and not _OCR_ALL_STOP.is_set():
                _OCR_ALL.update(phase="error", project="", error=_failed,
                                finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
            else:
                _OCR_ALL.update(phase="idle", project="", error="",
                                finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
        try:
            os.makedirs(os.path.dirname(_ocr_marker()), exist_ok=True)
            with open(_ocr_marker(), "w", encoding="utf-8") as f:
                json.dump({"finished_at": _OCR_ALL["finished_at"], "stopped": bool(_OCR_ALL["stop"])}, f)
        except OSError:
            pass


def _ocr_all_start(names=None, auto=False):
    import datetime as _dt
    names = [n for n in (names or list(ALLOW)) if n in ALLOW]
    if not names:
        return False
    with _OCR_ALL_LOCK:
        if _OCR_ALL["phase"] == "running":
            return False
        _OCR_ALL.update(phase="running", project="", done=0, total=len(names), stop=False, auto=bool(auto), error="",
                        started_at=_dt.datetime.now().isoformat(timespec="seconds"), finished_at=None)
        _OCR_ALL_STOP.clear()
    _threading.Thread(target=_ocr_all_worker, args=(names,), daemon=True, name="localdocs-ocr-all").start()
    return True


def _ocr_all_stop():
    import localdocs_settings as _st
    with _OCR_ALL_LOCK:
        _OCR_ALL["stop"] = True
    _OCR_ALL_STOP.set()
    _st.FORCE_OCR.clear()  # files still queued in the current pass skip OCR and finish quickly


def _maybe_auto_ocr():
    """First content search of a session (and at most once a day): OCR images/scans in the background, cached."""
    # Probe entitlement: _maybe_auto_ocr
    import probe_licensing as _pl
    if not _pl.allowed('ocr'):
        return
    try:
        cfg = _settings.get()
        if cfg.get("ocr_when") != "search" or not cfg.get("content_search") or not (cfg.get("ocr_images") or cfg.get("ocr_pdf_scans")):
            return
        with _OCR_ALL_LOCK:
            if _OCR_ALL["phase"] == "running":
                return
        try:
            import time as _time
            if _time.time() - os.path.getmtime(_ocr_marker()) < 86400 and not _OCR_ALL.get("auto_pending"):
                return
        except OSError:
            pass
        _ocr_all_start(auto=True)
    except Exception:
        pass


# ---- file system view, random file, duplicates, file download (drag-out) ----
def _system_path(path):
    """Local drive absolute paths only; reject every reparse ancestor (including junctions)."""
    import ntpath
    import re
    if not isinstance(path, str) or '\x00' in path:
        raise ValueError('invalid path')
    path = path.replace('/', '\\')
    if not re.match(r'^[A-Za-z]:\\', path) or path.startswith('\\\\'):
        raise PermissionError('local absolute drive path required')
    parts = path[3:].split('\\')
    for part in parts:
        if ':' in part or (part not in ('', '.', '..') and (part.endswith((' ', '.')) or
                re.match(r'^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\.|$)', part, re.I))):
            raise PermissionError('device paths are forbidden')
    path = ntpath.normpath(path)
    cur = path
    while True:
        st = os.lstat(cur)
        if os.path.islink(cur) or getattr(st, 'st_file_attributes', 0) & 0x400:
            raise PermissionError('reparse points are forbidden')
        parent = ntpath.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return path


_KNOWN_FOLDERS = {
    # FOLDERID GUIDs; real locations honour OneDrive / user redirection.
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}
_KNOWN_CACHE = None


def _known_folders():
    global _KNOWN_CACHE
    if _KNOWN_CACHE is not None:
        return dict(_KNOWN_CACHE)
    out = {}
    if os.name == "nt":
        import ctypes as ct
        import uuid as _uuid

        class _GUIDK(ct.Structure):
            _fields_ = [("data", ct.c_ubyte * 16)]
        try:
            shell = ct.WinDLL("shell32")
            ole = ct.WinDLL("ole32")
            fn = shell.SHGetKnownFolderPath
            fn.argtypes = [ct.POINTER(_GUIDK), ct.c_uint32, ct.c_void_p, ct.POINTER(ct.c_wchar_p)]
            fn.restype = ct.c_long
            for key, text in _KNOWN_FOLDERS.items():
                g = _GUIDK()
                ct.memmove(g.data, _uuid.UUID(text).bytes_le, 16)
                p = ct.c_wchar_p()
                if fn(ct.byref(g), 0, None, ct.byref(p)) == 0 and p.value:
                    if os.path.isdir(p.value):
                        out[key] = p.value
                    ole.CoTaskMemFree(p)
        except Exception:
            pass
    _KNOWN_CACHE = out
    return dict(out)


def _system_drives():
    import ctypes as ct
    from ctypes import wintypes as wt
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.GetLogicalDrives.argtypes, kernel.GetLogicalDrives.restype = [], wt.DWORD
    kernel.GetDriveTypeW.argtypes, kernel.GetDriveTypeW.restype = [wt.LPCWSTR], wt.UINT
    kernel.GetDiskFreeSpaceExW.argtypes = [wt.LPCWSTR] + [ct.POINTER(ct.c_ulonglong)] * 3
    kernel.GetDiskFreeSpaceExW.restype = wt.BOOL
    mask = kernel.GetLogicalDrives()
    if not mask:
        raise ct.WinError(ct.get_last_error())
    drives = []
    for i in range(26):
        if not mask & (1 << i):
            continue
        path = chr(65 + i) + ':\\'
        kind = {2: 'removable', 3: 'fixed', 4: 'network', 5: 'cdrom'}.get(kernel.GetDriveTypeW(path))
        if kind is None:
            continue
        available, total, free = ct.c_ulonglong(), ct.c_ulonglong(), ct.c_ulonglong()
        kernel.GetDiskFreeSpaceExW(path, ct.byref(available), ct.byref(total), ct.byref(free))
        drives.append({'name': path, 'path': path, 'type': kind, 'free': available.value, 'total': total.value})
    return drives


def _browse_system(path=''):
    import ntpath
    import stat
    if path == '':
        return {'root': 'system', 'path': '', 'parent': None, 'drives': _system_drives()}
    full = _system_path(path)
    if not os.path.isdir(full):
        raise ValueError('directory not found')
    entries = []
    with os.scandir(full) as iterator:
        for entry in iterator:
            try:
                st = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(st, 'st_file_attributes', 0) & 0x400:
                    continue
                # Hidden/system items (desktop.ini, thumbs.db, $Recycle.Bin…) stay hidden, as in Windows Explorer.
                if getattr(st, 'st_file_attributes', 0) & 0x6:
                    continue
                is_dir = stat.S_ISDIR(st.st_mode)
                entries.append({'name': entry.name, 'path': ntpath.join(full, entry.name),
                                'is_dir': is_dir, 'size': 0 if is_dir else st.st_size,
                                'mtime': st.st_mtime, 'ext': '' if is_dir else ntpath.splitext(entry.name)[1][1:].lower()})
            except OSError:
                continue
            if len(entries) >= 5000:
                break
    entries.sort(key=lambda e: (not e['is_dir'], e['name'].casefold()))
    parent = ntpath.dirname(full)
    return {'root': 'system', 'path': full, 'rel': full, 'parent': '' if parent == full else parent,
            'entries': entries}


def _system_preview(path):
    import stat
    target = _system_path(path)
    st = os.lstat(target)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError('preview requires an ordinary file')
    if st.st_size > 200 * 1024 * 1024:
        raise OverflowError('preview file exceeds 200 MB')
    return target


def _browse(project, rel):
    import localdocs_browse
    root = ALLOW[project]
    out = localdocs_browse.list_dir(root, rel or "")
    out["breadcrumbs"] = localdocs_browse.breadcrumbs(root, rel or "")
    return out


def _random_file(names):
    import random
    names = [n for n in names if n in ALLOW]
    if not names:
        raise ValueError("no folders")
    for _ in range(60):
        n = random.choice(names)
        cur = ALLOW[n]
        for _depth in range(12):
            try:
                entries = [e for e in os.scandir(cur) if not e.name.startswith(".")]
            except OSError:
                break
            if not entries:
                break
            e = random.choice(entries)
            try:
                if e.is_dir(follow_symlinks=False):
                    cur = e.path
                    continue
                st = e.stat(follow_symlinks=False)
            except OSError:
                break
            if st.st_size < 1:
                break
            return {"project": n, "path": e.path, "name": e.name, "size": st.st_size, "mtime": st.st_mtime}
    raise ValueError("nothing found")


_DUPES = {"phase": "idle", "scanned": 0, "groups": [], "wasted_bytes": 0, "stop": False, "started_at": None, "finished_at": None, "error": ""}
_DUPES_LOCK = _threading.Lock()
# Cooperative stop event: the single source of truth for should_stop checks.
# The dict ["stop"] flag is mirrored under _DUPES_LOCK for status compat.
_DUPES_STOP = _threading.Event()


def _dupes_worker(roots):
    import datetime as _dt
    _failed = ""
    try:
        import localdocs_dupes

        def progress(d):
            with _DUPES_LOCK:
                _DUPES["scanned"] = int(d.get("scanned", 0))

        res = localdocs_dupes.find_duplicates(roots, should_stop=_DUPES_STOP.is_set, progress=progress)
        with _DUPES_LOCK:
            _DUPES.update(groups=res.get("groups", [])[:200], wasted_bytes=int(res.get("wasted_bytes", 0)), scanned=int(res.get("scanned", 0)))
    except Exception as exc:
        try:
            _failed = str(exc)[:200] or "error"
        except Exception:
            _failed = "error"
    finally:
        with _DUPES_LOCK:
            if _failed and not _DUPES_STOP.is_set():
                _DUPES.update(phase="error", error=_failed,
                              finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
            else:
                _DUPES.update(phase="idle", error="",
                              finished_at=_dt.datetime.now().isoformat(timespec="seconds"))


def _dupes_start(names):
    import datetime as _dt
    roots = [ALLOW[n] for n in names if n in ALLOW]
    if not roots:
        return False
    with _DUPES_LOCK:
        if _DUPES["phase"] == "running":
            return False
        _DUPES.update(phase="running", scanned=0, groups=[], wasted_bytes=0, stop=False, error="",
                      started_at=_dt.datetime.now().isoformat(timespec="seconds"), finished_at=None)
        _DUPES_STOP.clear()
    _threading.Thread(target=_dupes_worker, args=(roots,), daemon=True, name="localdocs-dupes").start()
    return True



def _auto_index_first_run():
    """First launch: build the content index of every folder that has none yet, one by one, right away at low priority
    (OCR of images/scans is a separate, idle-gated job). Disable with LOCALDOCS_AUTO_INDEX=0."""
    import time as _t
    if os.environ.get("LOCALDOCS_AUTO_INDEX", "1") == "0":
        return
    _t.sleep(3)
    try:
        from project_search.backend import list_projects
        done = {os.path.normcase(os.path.realpath(str(p.get("folder", "")))) for p in list_projects()}
    except Exception:
        done = set()
    for n, f in list(ALLOW.items()):
        try:
            if os.path.normcase(os.path.realpath(f)) in done or not os.path.isdir(f):
                continue
        except OSError:
            continue
        if _start_index(n):
            while True:
                _t.sleep(2)
                with _INDEX_STATUS_LOCK:
                    if _INDEX_STATUS.get(n, {}).get("phase") != "running":
                        break


def _open_file(project, filename, reveal=False):
    import subprocess as _sp
    if os.name != "nt":
        raise NotImplementedError()
    if not isinstance(filename, str) or not filename:
        raise ValueError("file is required")
    root = os.path.realpath(ALLOW[project])
    target = os.path.realpath(filename if os.path.isabs(filename) else os.path.join(root, filename))
    try:
        if os.path.commonpath([root, target]) != root:
            raise PermissionError()
    except ValueError:
        raise PermissionError()
    if not os.path.isfile(target):
        raise ValueError("file not found")
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags)
    except OSError:
        raise ValueError("file not found")
    try:
        import stat as _stat
        if not _stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("file not found")
        fresh = os.path.realpath(target)
        try:
            if os.path.commonpath([root, fresh]) != root:
                raise PermissionError()
        except ValueError:
            raise PermissionError()
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if reveal:
        _sp.Popen(['explorer.exe', '/select,"%s"' % target], close_fds=True)
    else:
        os.startfile(target)
    return {"ok": True}

with open(os.path.join(ROOT, "app.html"), encoding="utf-8") as _app:
    APP_HTML = _app.read()
try:
    _APP_MTIME = os.stat(os.path.join(ROOT, "app.html")).st_mtime
except OSError:
    _APP_MTIME = 0.0


def _current_app_html():
    """Serve the interface file as it is on disk (an update or fix needs no server restart)."""
    global APP_HTML, _APP_MTIME
    path = os.path.join(ROOT, "app.html")
    try:
        mtime = os.stat(path).st_mtime
        if mtime != _APP_MTIME:
            with open(path, encoding="utf-8") as fresh:
                APP_HTML = fresh.read()
            _APP_MTIME = mtime
    except OSError:
        pass
    return APP_HTML


def _send_html(self):
    html = _current_app_html()
    host = self.headers.get("Host", "")
    origin = self.headers.get("Origin")
    peer = self.client_address[0]
    import ipaddress as _ipaddress
    try:
        loopback = _ipaddress.ip_address(peer).is_loopback
    except ValueError:
        loopback = False
    same_origin = origin is None or origin in ("http://" + host, "https://" + host)
    if (loopback and host in ("127.0.0.1:%d" % self.server.server_port,
                              "localhost:%d" % self.server.server_port) and same_origin):
        html = html.replace("/*__LOCALDOCS_SESSION__*/",
                            "window.LD_TOKEN=" + json.dumps(_OWNER_TOKEN) + ";")
    b = html.encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(b)))
    self.send_header("X-Content-Type-Options", "nosniff")
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(b)

def _match_token(tok):
    """Constant-time match against ALL known tokens (TOKENS keys + TOKEN).
    No early exit: every known token is compared, the first match is remembered."""
    import hmac as _hmac
    if tok is None:
        return None
    try:
        raw = tok.encode("utf-8")
    except AttributeError:
        return None
    found = None
    # Owner-lockout (accepted P4 contract): in users mode (TOKENS non-empty)
    # the legacy shared TOKEN no longer authenticates anywhere. Single-token
    # deployments (TOKENS empty) are unaffected.
    candidates = list(TOKENS) if TOKENS else [TOKEN]
    for known in candidates:
        try:
            eq = _hmac.compare_digest(raw, str(known).encode("utf-8"))
        except Exception:
            eq = False
        if eq and found is None:
            found = known
    return found

def _internal_error(e):
    """500 without leaking exception text to the client; full repr goes to logs/server_errors.log."""
    import datetime as _dt
    import secrets as _secrets
    eid = _secrets.token_hex(4)
    try:
        os.makedirs(os.path.join(DATA, "logs"), exist_ok=True)
        with open(os.path.join(DATA, "logs", "server_errors.log"), "a", encoding="utf-8") as f:
            f.write("%s %s %r\n" % (_dt.datetime.now().isoformat(timespec="seconds"), eid, e))
    except OSError:
        pass
    return {"error": "internal error", "code": "internal-error", "id": eid}

class H(BaseHTTPRequestHandler):
    def _ocr_upload(self):
        try:
            _ip = self.client_address[0]
        except Exception:
            _ip = "?"
        if _auth_ip_blocked(_ip):
            return self._send({"error": "too many attempts"}, 429)
        if not self._auth() or (TOKENS and not self._is_owner()):
            return self._send({"error": "forbidden"}, 403)
        try:
            n = int(self.headers.get("Content-Length"))
        except (TypeError, ValueError):
            return self._send({"error": "bad-content-length"}, 400)
        limit = int(os.environ.get("LOCALDOCS_MAX_UPLOAD", str(500 * 1024 * 1024)))
        if n < 0 or n > limit:
            return self._send({"error": "payload too large"}, 413)
        import probe_licensing as _pl
        if not _pl.allowed('ocr'):
            return self._send({"error": "license_required", "code": "license_required", "feature": "ocr"}, 403)
        from urllib.parse import unquote
        import uuid as _uuid
        raw_name = unquote(self.headers.get("X-Filename", "upload.bin"))
        name = os.path.basename(raw_name.replace(chr(92), "/")) or "upload.bin"
        folder = os.path.join(_uploads_dir(), _uuid.uuid4().hex[:12])
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, name)
        left = n
        with open(dest, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left > 0:
            try:
                os.remove(dest)
            except OSError:
                pass
            return self._send({"error": "incomplete-body"}, 400)
        return self._send({"id": os.path.basename(folder), "path": dest})

    def _serve_thumb(self, query):
        try:
            width = int((query.get("w") or ["320"])[0])
        except ValueError:
            width = 320
        width = max(64, min(width, 1024))
        try:
            if (query.get("system") or [""])[0] == "1":
                if TOKENS and not self._is_owner():
                    return self._send({"error": "owner-only"}, 403)
                full = _system_preview((query.get("path") or [""])[0])
            else:
                project = (query.get("project") or [""])[0]
                if not project or project not in ALLOW or not self._may(project):
                    return self._send({"error": "forbidden-project"}, 403)
                root = os.path.realpath(ALLOW[project])
                path = (query.get("path") or [""])[0]
                full = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
                if os.path.commonpath([root, full]) != root:
                    return self._send({"error": "file outside project"}, 403)
                if not os.path.isfile(full):
                    return self._send({"error": "file not found"}, 404)
            data = _thumb_bytes(full, width)
        except PermissionError:
            return self._send({"error": "access denied"}, 403)
        except FileNotFoundError:
            return self._send({"error": "file not found"}, 404)
        except OverflowError:
            return self._send({"error": "too-large"}, 413)
        except Exception:
            return self._send({"error": "thumb-failed"}, 415)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, query):
        if (query.get('system') or [''])[0] == '1':
            if TOKENS and not self._is_owner():
                return self._send({'error': 'owner-only'}, 403)
            try:
                target = _system_preview((query.get('path') or [''])[0])
                # Open before sending headers; access failures remain JSON responses.
                # Re-verify after open: regular file, still passes _system_path (TOCTOU).
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                try:
                    _fd = os.open(target, flags)
                except OSError:
                    return self._send({'error': 'file not found'}, 404)
                try:
                    import stat as _stat
                    if not _stat.S_ISREG(os.fstat(_fd).st_mode):
                        return self._send({'error': 'preview-error', 'code': 'preview-error', 'detail': 'not-a-file'}, 400)
                    try:
                        _system_path(target)
                    except (PermissionError, ValueError):
                        return self._send({'error': 'access denied'}, 403)
                    import mimetypes
                    size = os.fstat(_fd).st_size
                    if size > 200 * 1024 * 1024:
                        raise OverflowError('preview file exceeds 200 MB')
                    self.send_response(200)
                    self.send_header('Content-Type', mimetypes.guess_type(target)[0] or 'application/octet-stream')
                    self.send_header('Content-Length', str(size))
                    self.send_header('Content-Disposition', 'inline')
                    self.send_header('Content-Security-Policy', "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:")
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    with os.fdopen(_fd, 'rb', closefd=True) as stream:
                        _fd = None
                        remaining = size
                        while remaining:
                            chunk = stream.read(min(1 << 20, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                finally:
                    if _fd is not None:
                        try:
                            os.close(_fd)
                        except OSError:
                            pass
                return
            except PermissionError:
                return self._send({'error': 'access denied'}, 403)
            except FileNotFoundError:
                return self._send({'error': 'file not found'}, 404)
            except OverflowError as e:
                return self._send({'error': 'preview-too-large', 'code': 'preview-too-large', 'detail': _safe_detail(e)}, 413)
            except (OSError, ValueError) as e:
                return self._send({'error': 'preview-error', 'code': 'preview-error', 'detail': _safe_detail(e)}, 400)
        project = (query.get("project") or [""])[0]
        path = (query.get("path") or [""])[0]
        if not project or not self._may(project):
            return self._send({"error": "forbidden-project"}, 403)
        root = os.path.realpath(ALLOW[project])
        target = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
        try:
            if os.path.commonpath([root, target]) != root:
                return self._send({"error": "file outside project"}, 403)
        except ValueError:
            return self._send({"error": "file outside project"}, 403)
        if not os.path.isfile(target):
            return self._send({"error": "file not found"}, 404)
        import mimetypes
        from urllib.parse import quote
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            _fd = os.open(target, flags)
        except OSError:
            return self._send({"error": "file not found"}, 404)
        try:
            import stat as _stat2
            try:
                if not _stat2.S_ISREG(os.fstat(_fd).st_mode):
                    return self._send({"error": "file not found"}, 404)
                fresh = os.path.realpath(target)
                try:
                    if os.path.commonpath([root, fresh]) != root:
                        return self._send({"error": "file outside project"}, 403)
                except ValueError:
                    return self._send({"error": "file outside project"}, 403)
                size = os.fstat(_fd).st_size
            except OSError:
                return self._send({"error": "file not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(os.path.basename(target)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with os.fdopen(_fd, "rb", closefd=True) as f:
                _fd = None
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        finally:
            if _fd is not None:
                try:
                    os.close(_fd)
                except OSError:
                    pass

    def _send(self, o, code=200):
        if isinstance(o, dict) and "error" in o and "code" not in o:
            import re as _re
            _err = o.get("error")
            if isinstance(_err, str) and _re.match(r"^[a-z0-9][a-z0-9\-_]*$", _err):
                o["code"] = _err
            else:
                _slug = _re.sub(r"[^a-z0-9]+", "-", str(_err).lower()).strip("-")[:40].strip("-")
                o["code"] = _slug or "error"
        b = json.dumps(o, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(b)
    def _auth(self):
        tok = self.headers.get("X-Token")
        import time as _t
        with _SESSIONS_LOCK:
            _sessions_prune_locked(_t.time())
            if tok in SESSIONS:
                if SESSIONS[tok]["until"] > _t.time():
                    return True
                del SESSIONS[tok]
                return False
        return _match_token(tok) is not None
    def _groups(self):
        tok = self.headers.get("X-Token")
        import time as _t
        with _SESSIONS_LOCK:
            _sessions_prune_locked(_t.time())
            if tok in SESSIONS:
                return set(SESSIONS[tok]["groups"])
        return TOKENS.get(tok, set())
    def _may(self, project):
        if project not in ALLOW:
            return False
        need = PROJ_GROUPS.get(project, set())
        if not need:
            return True
        import time as _t
        tok = self.headers.get("X-Token")
        with _SESSIONS_LOCK:
            _sessions_prune_locked(_t.time())
            if tok in SESSIONS:
                return bool(need & SESSIONS[tok]["groups"])
        if not TOKENS:
            return True  # legacy single-token owner
        return bool(need & self._groups())
    def _quick_roots(self, body=None):
        names = [n for n in ALLOW if self._may(n)]
        body = body or {}
        only = body.get("only_projects")
        if only is not None:
            if not isinstance(only, list) or not all(isinstance(n, str) for n in only):
                raise ValueError("invalid-only-projects")
            names = [n for n in names if n in only]
        roots = None if self._is_owner() and only is None else [ALLOW[n] for n in names]
        requested = body.get("only_roots")
        if requested is not None:
            if not isinstance(requested, list) or not all(isinstance(r, str) and r for r in requested):
                raise ValueError("invalid-only-roots")
            if len(requested) > 50:
                raise ValueError("invalid-only-roots")
            for r in requested:
                if len(r) > 400:
                    raise ValueError("invalid-only-roots")
            requested = [_srv_norm_real(r) for r in requested]
            if roots is None:
                return requested
            narrowed = []
            for r in requested:
                for root in roots:
                    root = _srv_norm_real(root)
                    try:
                        common = os.path.commonpath([r, root])
                    except ValueError:
                        continue
                    if common == root:
                        narrowed.append(r)
                    elif common == r:
                        narrowed.append(root)
            roots = narrowed
        return roots
    def do_GET(self):
        if self.path == "/app":
            return _send_html(self)
        if self.path == "/ping":  # unauthenticated liveness probe for the launcher (no data, not counted as an auth failure)
            return self._send({"ok": True})
        if self.path == "/icon.png":
            try:
                with open(os.path.join(ROOT, "brand", "icon.png"), "rb") as icon:
                    data = icon.read()
            except OSError:
                return self._send({"error": "unknown"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return
        try:
            _ip = self.client_address[0]
        except Exception:
            _ip = "?"
        if _auth_ip_blocked(_ip):
            return self._send({"error": "too many attempts"}, 429)
        from urllib.parse import urlsplit, parse_qs
        route = urlsplit(self.path)
        if route.path == "/file" and not self.headers.get("X-Token"):
            # Browser drag-out cannot send headers: one-time ticket ?ticket= (TTL 30s).
            # Legacy ?t= (long-lived token in URL) still works for one version with a server-log warning.
            _qs = parse_qs(route.query)
            _tick = (_qs.get("ticket") or [""])[0]
            if _tick:
                _tok = _redeem_file_ticket(_tick)
                if _tok:
                    self.headers["X-Token"] = _tok
            else:
                _t = (_qs.get("t") or [""])[0]
                if _t:
                    _warn_compat("GET /file?t= deprecated; use POST /file/ticket")
                    self.headers["X-Token"] = _t
        if not self._auth():
            _auth_note(_ip, False)
            return self._send({"error": "forbidden"}, 403)
        _auth_note(_ip, True)
        if route.path == "/file":
            return self._serve_file(parse_qs(route.query))
        if route.path == "/thumb":
            return self._serve_thumb(parse_qs(route.query))
        if route.path == "/fs/preview":
            # Read-only preview (panel in the Files tab): same auth as
            # other /fs/* reads (X-Token header, never a token in query).
            # POST /fs/preview is served by the /fs/ branch in do_POST.
            import localdocs_fsapi as _fsapi
            q = parse_qs(route.query)
            _body = {"project": (q.get("project") or [""])[0],
                     "rel": (q.get("rel") or q.get("path") or [""])[0]}
            code, payload = _fsapi.handle("/fs/preview", ALLOW, self._auth(),
                                          self._is_owner(),
                                          lambda p: self._may(p), _body)
            return self._send(payload, code)
        if route.path.startswith("/api/update/"):
            # Authenticated remote clients remain supported, but updater operations are local-only.
            if not self._is_loopback_peer():
                return self._send({"error": "loopback only"}, 403)
            if route.path == "/api/update/backup_status":
                import localdocs_updater
                _code, _res = localdocs_updater.handle(self.server, route.path, {})
                return self._send(_res, _code)
            if route.path != "/api/update/status":
                return self._send({"error": "method not allowed"}, 405)
            import localdocs_updater
            # status() already carries rollback_available/backup_version/backup_date (app_backup_status).
            return self._send(localdocs_updater.status(self.server))
        if route.path == "/api/app-info":
            try:
                import project_search
                version = getattr(project_search, "__version__", "0.0.0")
            except Exception:
                version = "0.0.0"
            model = os.environ.get("LOCALDOCS_MODEL_TIER", "balanced").lower()
            if model not in ("balanced", "fast"):
                model = "balanced"
            try:
                import pytesseract
                ocr = bool(pytesseract)
            except Exception:
                ocr = False
            return self._send({"name": "LocalDocs", "version": version,
                               "model": model, "ocr_available": ocr, "first_run": bool(_DEFAULT_ONLY) or not bool(ALLOW), "live_updates": bool(_EVENTS is not None and _EVENTS.status().get("available")),
                               "site_url": "https://lendarixon.github.io/LocalDocs/",
                               "releases_url": "https://github.com/Lendarixon/LocalDocs/releases"})
        if route.path == "/index/status":
            project = parse_qs(route.query).get("project", [""])[0]
            if not self._may(project):
                return self._send({"error": "forbidden-project"}, 403)
            with _INDEX_STATUS_LOCK:
                state = dict(_INDEX_STATUS.get(project, {"phase": "idle", "files": 0, "chunks": 0,
                    "current": "", "error": "", "started_at": None, "finished_at": None}))
            out = dict(state)
            out.update(_jobs_legacy_extra("content-index", {"project": project}))
            return self._send(out)
        if self.path == "/version":
            from project_search import __version__
            names = sorted(p for p in ALLOW if self._may(p))
            out = {"version": __version__, "projects": names, "https": _USE_HTTPS}
            if self._is_owner():
                # The owner sees where each search folder lives (labels alone are ambiguous in the UI).
                out["folders"] = {n: ALLOW[n] for n in names}
                out["demo"] = bool(_DEFAULT_ONLY)
                out["home"] = os.path.expanduser("~")
                out["known"] = _known_folders()
            return self._send(out)
        if self.path == "/health":
            health = _health()
            folder_to_name = {f: n for n, f in ALLOW.items() if self._may(n)}
            filtered = {}
            for folder, info in (health.get("projects") or {}).items():
                if folder in folder_to_name:
                    filtered[folder_to_name[folder]] = info
            if "error" in (health.get("projects") or {}):
                filtered["error"] = "unavailable"
            health['projects'] = filtered
            # Watcher changes contain paths from projects outside this user's groups.
            try:
                health['watcher'].pop('last_changes', None)
            except (AttributeError, TypeError):
                pass
            return self._send(health)
        if self.path == "/quick/status":
            try:
                roots = self._quick_roots()
                if roots is None:
                    return self._send(_quick().status())
                # Global roots and file counts describe other users' folders.
                full = _quick().status()
                phase = full.get("phase", "idle") if roots else "ready"
                return self._send({"phase": phase, "files": None if roots else 0, "roots": roots,
                                   "root_status": _filter_root_status(full.get("root_status"), roots)})
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path == "/list":
            from project_search.backend import list_projects
            folder_to_name = {f: n for n, f in ALLOW.items() if self._may(n)}
            out = []
            for p in list_projects():
                if not isinstance(p, dict):
                    continue
                folder = p.get("folder")
                if folder in folder_to_name:
                    q = {k: v for k, v in p.items() if k != "folder"}
                    q["project"] = folder_to_name[folder]
                    out.append(q)
            return self._send(out)
        if route.path == "/settings":
            try:
                return self._send(_settings.get())
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if route.path == "/doctor":
            if not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            try:
                import project_search
                from project_search import doctor as _doctor
                from project_search.backend import list_projects
                checks = _doctor.run_checks()
                allow_folders = set(ALLOW.values())
                indexed = [
                    p for p in list_projects()
                    if isinstance(p, dict) and p.get("folder") in allow_folders
                ]
                tess_ok = any(
                    c.get("name") == "tesseract" and c.get("ok") is True
                    for c in checks
                )
                return self._send({
                    "ok": all(c.get("ok") is not False for c in checks),
                    "version": getattr(project_search, "__version__", "0.0.0"),
                    "python": sys.version.split()[0],
                    "projects_count": len(ALLOW),
                    "indexed_projects_count": len(indexed),
                    "model": _doctor._model_name(),
                    "ocr_available": tess_ok,
                    "checks": checks,
                })
            except Exception as e:
                return self._send(_internal_error(e), 500)
        return self._send({"error": "unknown"}, 404)
    def do_POST(self):
        if self.path == "/api/prewarm":
            try:
                _n = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                _n = 0
            try:
                if _n > 0:
                    self.rfile.read(min(_n, 1048576))
            except Exception:
                pass
            import threading as _th
            _th.Thread(target=_ensure_prewarm, daemon=True).start()
            return self._send({"ok": True}, 202)
        if self.path == "/ocr-upload":
            return self._ocr_upload()
        try:
            _ip = self.client_address[0]
        except Exception:
            _ip = "?"
        if _auth_ip_blocked(_ip):
            return self._send({"error": "too many attempts"}, 429)
        # Auth by header BEFORE reading the body (DoS: unauthed must not force n-byte reads).
        # /login authenticates via the body itself, so it skips this pre-check.
        if self.path != "/login" and not self._auth():
            _auth_note(_ip, False)
            return self._send({"error": "forbidden"}, 403)
        # (1) Body-size guard: validate Content-Length before reading anything.
        try:
            n = int(self.headers.get("Content-Length"))
        except (TypeError, ValueError):
            n = -1
        if n < 0:
            return self._send({"error": "bad-content-length"}, 400)
        try:
            max_body = int(os.environ.get("LOCALDOCS_MAX_BODY", "1048576"))
        except ValueError:
            max_body = 1048576
        if n > max_body:
            return self._send({"error": "payload too large"}, 413)
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except Exception: return self._send({"error": "bad-json"}, 400)
        if not isinstance(body, dict):
            return self._send({"error": "body-must-be-object"}, 400)
        if self.path == "/login":
            try:
                result = _login(body)
                _auth_note(_ip, "token" in result)
                return self._send(result, 200 if "token" in result else 403)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if not self._auth():
            _auth_note(_ip, False)
            return self._send({"error": "forbidden"}, 403)
        _auth_note(_ip, True)
        if (self.path not in _KNOWN_POST and not self.path.startswith("/fs/")
                and not self.path.startswith("/api/update/")):
            return self._send({"error": "unknown", "code": "unknown"}, 404)
        if self.path in ("/ask", "/ask/status"):
            # Linda Sejf: odpowiedź Bielika wyłącznie z folderów dozwolonych dla tego tokenu.
            import hashlib
            import localdocs_sejf as _sejf
            if self.path == "/ask/status":
                return self._send(_sejf.status())
            question = body.get("question")
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                return self._send({"error": "bad-question"}, 400)
            names = [n for n in ALLOW if self._may(n)]
            wanted = body.get("projects")
            if isinstance(wanted, list):
                wanted = {w for w in wanted if isinstance(w, str)}
                names = [n for n in names if n in wanted]
            if not names:
                return self._send({"error": "no-allowed-projects"}, 403)
            tok = self.headers.get("X-Token") or ""
            user = hashlib.sha256(tok.encode("utf-8")).hexdigest()[:12]
            try:
                return self._send(_sejf.ask(question.strip(), [ALLOW[n] for n in names], user=user,
                                            mode=str(body.get("mode", "flag")), k=body.get("k", 5)))
            except RuntimeError as e:
                return self._send({"error": str(e)}, 503)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        import probe_licensing as _pl
        if self.path == "/license":
            if not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            try:
                action = body.get("action", "status")
                import probe_packs as _packs
                if action == "start_trial": return self._send(_pl.start_trial())
                if action == "activate":
                    state = _pl.activate(body.get("key", ""))
                    return self._send(state)
                if action == "remove":
                    _packs.stop()
                    return self._send(_pl.deactivate())
                if action == "prepare":
                    return self._send(_packs.start(body.get("packs")))
                if action == "pack_status": return self._send(_packs.status())
                if action == "pack_stop": return self._send(_packs.stop())
                if action == "status": return self._send(_pl.public_state())
                return self._send({"error": "unknown"}, 400)
            except _pl.LicenseError as exc:
                return self._send({"error": exc.code, "code": exc.code}, 403)
            except ValueError:
                return self._send({"error": "invalid-packs"}, 400)
        feature = _pl.route_feature(self.path, body)
        if feature and not _pl.allowed(feature):
            return self._send({"error": "license_required", "code": "license_required", "feature": feature}, 403)
        if self.path == "/file/ticket":
            try:
                return self._send({"ticket": _issue_file_ticket(self.headers.get("X-Token"))})
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path in ("/project/add", "/project/remove", "/pick-folder") and not self._is_owner():
            return self._send({"error": "owner-only"}, 403)
        if self.path == "/project/add":
            if not isinstance(body.get("path"), str) or ("name" in body and not isinstance(body.get("name"), str)):
                return self._send({"error": "invalid path or name"}, 400)
            try:
                return self._send({"project": _add_project(body.get("path"), body.get("name"))})
            except ValueError as e:
                return self._send({"error": "invalid-project-path", "code": "invalid-project-path", "detail": _safe_detail(e)}, 400)
        if self.path == "/project/remove":
            if not isinstance(body.get("project"), str):
                return self._send({"error": "invalid project"}, 400)
            try:
                _remove_project(body.get("project"))
                return self._send({"ok": True})
            except ValueError as e:
                return self._send({"error": "unknown-project", "code": "unknown-project", "detail": _safe_detail(e)}, 400)
        if self.path == "/pick-folder":
            try:
                return self._send({"path": _pick_folder()})
            except PickerBusy:
                return self._send({"error": "picker-busy"}, 409)
            except Exception:
                return self._send({"error": "folder picker failed"}, 500)
        if self.path == "/index/start":
            project = body.get("project")
            if not isinstance(project, str):
                return self._send({"error": "invalid project"}, 400)
            if not self._may(project):
                return self._send({"error": "forbidden-project"}, 403)
            scope = {"project": project}
            tab = _jobs_tab(body)
            def _ci_start(p=project):
                return _start_index(p)
            def _ci_status(p=project):
                with _INDEX_STATUS_LOCK:
                    return dict(_INDEX_STATUS.get(p, {"phase": "idle", "files": 0, "chunks": 0,
                        "current": "", "error": "", "started_at": None, "finished_at": None}))
            def _ci_stop_fn(p=project):
                return _ci_stop(p)
            try:
                rec, started = localdocs_jobs.start_family(
                    "content-index", scope, tab, True, _ci_start, _ci_status, _ci_stop_fn,
                    localdocs_jobs.normalize_content_index)
            except Exception as e:
                return self._send(_internal_error(e), 500)
            if rec is None and not started:
                return self._send(dict(_ci_status(), started=False,
                    **_jobs_legacy_extra("content-index", scope)), 200)
            if not started:
                with _INDEX_STATUS_LOCK:
                    legacy = dict(_INDEX_STATUS.get(project, {"phase": "running"}))
                return self._send(dict(legacy, started=False,
                    **_jobs_legacy_extra("content-index", scope)), 200)
            return self._send(dict(_ci_status(), started=True,
                **_jobs_legacy_extra("content-index", scope)), 202)
        if self.path == "/open":
            project = body.get("project")
            if not isinstance(project, str):
                return self._send({"error": "invalid project"}, 400)
            if not self._may(project):
                return self._send({"error": "forbidden-project"}, 403)
            try:
                _open_res = _open_file(project, body.get("file"), bool(body.get("reveal")))
            except PermissionError:
                return self._send({"error": "file outside project"}, 403)
            except ValueError as e:
                return self._send({"error": "invalid-file", "code": "invalid-file", "detail": _safe_detail(e)}, 400)
            except NotImplementedError:
                return self._send({"error": "Windows only"}, 501)
            except Exception as e:
                return self._send(_internal_error(e), 500)
            try:
                _audit_click(project, _audit_click_rel(project, body.get("file") if isinstance(body.get("file"), str) else ""),
                             body.get("rank_pos", body.get("rank")), body.get("mission_id", body.get("mission")), body.get("query", ""))
            except Exception:
                pass
            return self._send(_open_res)
        if self.path == "/parse-query":
            text = body.get("text", "")
            if not isinstance(text, str) or len(text) > 500:
                return self._send({"error": "invalid text"}, 400)
            try:
                import localdocs_nlq
                return self._send(localdocs_nlq.parse(text))
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path == "/ocr-all":
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            action = body.get("action", "status")
            if action == "start":
                names = body.get("projects")
                wanted = list(names) if isinstance(names, list) else None
                tab = _jobs_tab(body)
                # Resolve names without storing paths: labels only, bounded.
                resolved = [n for n in (wanted if wanted is not None else list(ALLOW)) if n in ALLOW]
                scope = {"projects": len(resolved)}
                def _oa_start(names_in=resolved):
                    return _ocr_all_start(list(names_in))
                try:
                    rec, started = localdocs_jobs.start_family(
                        "ocr-all", scope, tab, True, _oa_start,
                        _ocr_all_status, _ocr_all_stop, localdocs_jobs.normalize_ocr_all)
                except Exception as e:
                    return self._send(_internal_error(e), 500)
                if rec is None and not started:
                    return self._send(dict(_ocr_all_status(), started=False,
                        **_jobs_legacy_extra("ocr-all", scope)), 200)
                if not started:
                    return self._send(dict(_ocr_all_status(), started=False,
                        **_jobs_legacy_extra("ocr-all", scope)), 200)
                return self._send(dict(_ocr_all_status(), started=True,
                    **_jobs_legacy_extra("ocr-all", scope)), 202)
            if action == "stop":
                localdocs_jobs.family_stop("ocr-all", {"projects": 0})
                _ocr_all_stop()
            out = dict(_ocr_all_status())
            out.update(_jobs_legacy_extra("ocr-all", {"projects": 0}))
            return self._send(out)
        if self.path == "/pick-files":
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            try:
                return self._send({"paths": _pick_files()})
            except PickerBusy:
                return self._send({"error": "picker-busy"}, 409)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path == "/reveal-path":
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            try:
                _reveal_res = _reveal_path(str(body.get("path", "")))
            except ValueError as e:
                return self._send({"error": "not-found", "code": "not-found", "detail": _safe_detail(e)}, 404)
            except Exception as e:
                return self._send(_internal_error(e), 500)
            try:
                _rp, _rr = _audit_click_project_for(str(body.get("path", "")))
                _audit_click(_rp, _rr, body.get("rank_pos", body.get("rank")),
                             body.get("mission_id", body.get("mission")), body.get("query", ""))
            except Exception:
                pass
            return self._send(_reveal_res)
        if self.path == "/open-path":
            if (TOKENS and not self._is_owner()) or not self._is_loopback_peer():
                return self._send({"error": "owner-only"}, 403)
            try:
                return self._send(_open_system_path(body.get("path", ""), bool(body.get("reveal"))))
            except PermissionError:
                return self._send({"error": "access denied"}, 403)
            except (ValueError, OSError) as e:
                return self._send({"error": "not-found", "code": "not-found", "detail": _safe_detail(e)}, 404)
            except NotImplementedError:
                return self._send({"error": "Windows only"}, 501)
        if self.path == "/img-list":
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            try:
                return self._send(_img_list(body))
            except (ValueError, TypeError) as e:
                return self._send({"error": "invalid-request", "detail": _safe_detail(e)}, 400)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path == "/ocr-batch":
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            import localdocs_ocr_batch
            job = localdocs_ocr_batch.get_job()
            action = body.get("action", "status")
            if action == "start":
                items = body.get("items")
                if not isinstance(items, list) or not items or len(items) > 500:
                    return self._send({"error": "items required (max 500)"}, 400)
                mode = body.get("mode", "txt")
                if mode not in ("pdf_layer", "pdf_new", "txt", "md", "text"):
                    return self._send({"error": "invalid mode"}, 400)
                cfg = _settings.get()
                engine = body.get("engine") or ("own" if cfg.get("ocr_engine") == "own" else "auto")
                device = body.get("device") or ("gpu" if cfg.get("ocr_gpu") else "cpu")
                if engine not in ("auto", "own", "rapid", "tesseract") or device not in ("cpu", "gpu"):
                    return self._send({"error": "invalid engine/device"}, 400)
                try:
                    str_items = [str(x) for x in items]
                    str_outdir = str(body.get("outdir") or "")
                    str_lang = str(body.get("lang") or cfg.get("ocr_lang") or "rus+eng+pol")
                    dpi = int(body.get("dpi") or 200)
                    skip = bool(body.get("skip_done", True))
                    try:
                        rotation = int(body.get("rotation") or 0) % 360
                    except (TypeError, ValueError):
                        return self._send({"error": "rotation must be 0/90/180/270"}, 400)
                    if rotation not in (0, 90, 180, 270):
                        return self._send({"error": "rotation must be 0/90/180/270"}, 400)
                    spreads = body.get("spreads", False)
                    if not isinstance(spreads, bool):
                        return self._send({"error": "spreads must be bool"}, 400)
                    tab = _jobs_tab(body)
                    scope = {"mode": mode, "items": len(str_items)}
                    def _ob_start():
                        try:
                            job.start(str_items, mode, str_outdir, str_lang, device, engine, dpi, skip_done=skip,
                                      rotation=rotation, spreads=spreads)
                            return True
                        except RuntimeError:
                            return False
                    rec, started = localdocs_jobs.start_family(
                        "ocr-batch", scope, tab, True, _ob_start,
                        job.status, job.stop, localdocs_jobs.normalize_ocr_batch)
                except Exception as e:
                    return self._send(_internal_error(e), 500)
                if rec is None and not started:
                    return self._send(dict(job.status(), started=False,
                        **_jobs_legacy_extra("ocr-batch", scope)), 200)
                if not started:
                    return self._send(dict(job.status(), started=False,
                        **_jobs_legacy_extra("ocr-batch", scope)), 200)
                return self._send(dict(job.status(), started=True,
                    **_jobs_legacy_extra("ocr-batch", scope)), 202)
            if action == "stop":
                localdocs_jobs.family_stop("ocr-batch", {"mode": "x", "items": 0} if False else None)
                try:
                    localdocs_jobs.REGISTRY.refresh()
                except Exception:
                    pass
                job.stop()
            out = dict(job.status())
            out.update(_jobs_legacy_extra("ocr-batch", None))
            return self._send(out)
        if self.path in ("/models", "/img-search", "/img-index", "/img-tags", "/ocr-journal"):
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            try:
                if self.path == "/models":
                    return self._send(_models_api(body))
                if self.path == "/ocr-journal":
                    return self._send(_ocr_journal_api(body))
                if self.path == "/img-tags":
                    return self._send(_img_tags_api(body))
                if self.path == "/img-index":
                    iaction = str(body.get("action", "status"))
                    if iaction == "start":
                        paths = body.get("paths")
                        tab = _jobs_tab(body)
                        scope = {"selection": "custom" if isinstance(paths, list) else "projects",
                                 "items": len(paths) if isinstance(paths, list) else 0}
                        def _ii_start(paths_in=paths):
                            with _IMG_LOCK:
                                if _IMG_JOB.get("phase") == "running":
                                    return False
                            _img_index("start", paths_in)
                            return True
                        def _ii_status():
                            return dict(_IMG_JOB)
                        def _ii_stop():
                            _img_index("stop")
                        try:
                            rec, started = localdocs_jobs.start_family(
                                "image-index", scope, tab, True, _ii_start,
                                _ii_status, _ii_stop, localdocs_jobs.normalize_image)
                        except Exception as e:
                            return self._send(_internal_error(e), 500)
                        if rec is None and not started:
                            existing = _jobs_lookup("image-index", scope)
                            if existing is None and _ii_status().get("phase") == "running":
                                try:
                                    rec2, _s2 = localdocs_jobs.start_family(
                                        "image-index", scope, tab, True, lambda: True,
                                        _ii_status, _ii_stop, localdocs_jobs.normalize_image)
                                    existing = {"job_id": rec2["id"], "job": rec2} if rec2 else None
                                except Exception:
                                    existing = None
                            base = dict(_ii_status(), started=False)
                            if existing:
                                base.update(existing)
                            return self._send(base, 200)
                        if not started:
                            return self._send(dict(_ii_status(), started=False,
                                **_jobs_legacy_extra("image-index", scope)), 200)
                        return self._send(dict(_ii_status(), started=True,
                            **_jobs_legacy_extra("image-index", scope)), 202)
                    if iaction == "stop":
                        localdocs_jobs.family_stop("image-index", None)
                        _img_index("stop")
                    out = dict(_IMG_JOB)
                    out.update(_jobs_legacy_extra("image-index", None))
                    return self._send(out)
                return self._send(_img_search(body))
            except OverflowError as e:
                return self._send({'error': 'too-large', 'code': 'too-large', 'detail': _safe_detail(e)}, 413)
            except PermissionError as e:
                return self._send({'error': 'access denied', 'code': 'access-denied', 'detail': _safe_detail(e)}, 403)
            except OSError as e:
                if self.path == '/img-tags' and str(body.get("action", "")) == 'import':
                    return self._send({'error': 'network', 'code': 'network', 'detail': _safe_detail(e)}, 502)
                return self._send({'error': 'io-error', 'code': 'io-error', 'detail': _safe_detail(e)}, 400)
            except (ValueError, KeyError, TypeError) as e:
                return self._send({"error": "invalid-request", "code": "invalid-request", "detail": _safe_detail(e) or "invalid"}, 400)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path == "/fs/restore":
            # Undo for "move to Recycle Bin": restore the given project items from the bin.
            if (TOKENS and not self._is_owner()) or not self._is_loopback_peer():
                return self._send({"error": "owner-only"}, 403)
            project = body.get("project")
            rels = body.get("items")
            if not isinstance(project, str) or project not in ALLOW or not self._may(project):
                return self._send({"error": "forbidden-project"}, 403)
            if not isinstance(rels, list) or not rels or len(rels) > 500 or not all(isinstance(r, str) and r for r in rels):
                return self._send({"error": "invalid-request"}, 400)
            root = os.path.realpath(ALLOW[project])
            targets = []
            for rel in rels:
                full = os.path.normpath(os.path.join(root, rel.replace("/", os.sep)))
                try:
                    if os.path.commonpath([root, full]) != root or full == root:
                        return self._send({"error": "outside project"}, 403)
                except ValueError:
                    return self._send({"error": "outside project"}, 403)
                targets.append(full)
            try:
                restored = _restore_recycled(targets)
            except Exception as e:
                return self._send(_internal_error(e), 500)
            return self._send({"restored": len(restored), "requested": len(targets)}, 200 if restored else 404)
        if self.path.startswith("/fs/"):
            import localdocs_fsapi as _fsapi
            code, payload = _fsapi.handle(self.path, ALLOW, self._auth(), self._is_owner(), lambda p: self._may(p), body)
            if self.path == "/fs/preview":
                try:
                    if int(code) < 400 and isinstance(body.get("project"), str):
                        _audit_click(body.get("project"), body.get("rel") if isinstance(body.get("rel"), str) else "",
                                     body.get("rank_pos", body.get("rank")), body.get("mission_id", body.get("mission")), body.get("query", ""))
                except Exception:
                    pass
            return self._send(payload, code)
        if self.path == "/browse":
            if body.get('system') is True:
                if TOKENS and not self._is_owner():
                    return self._send({'error': 'owner-only'}, 403)
                try:
                    return self._send(_browse_system(body.get('path', '')))
                except PermissionError:
                    return self._send({'error': 'access denied'}, 403)
                except (OSError, ValueError, TypeError) as e:
                    return self._send({'error': 'browse-error', 'code': 'browse-error', 'detail': _safe_detail(e)}, 400)
            project = body.get("project")
            if not isinstance(project, str):
                return self._send({"error": "invalid project"}, 400)
            if not self._may(project):
                return self._send({"error": "forbidden-project"}, 403)
            try:
                return self._send(_browse(project, body.get("rel", "")))
            except PermissionError:
                return self._send({"error": "outside project"}, 403)
            except (ValueError, OSError) as e:
                return self._send({"error": "browse-error", "code": "browse-error", "detail": _safe_detail(e)}, 400)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path == "/random-file":
            names = [n for n in ALLOW if self._may(n)]
            wanted = body.get("projects")
            if isinstance(wanted, list):
                names = [n for n in names if n in wanted]
            try:
                return self._send(_random_file(names))
            except ValueError as e:
                return self._send({"error": "not-found", "code": "not-found", "detail": _safe_detail(e)}, 404)
        if self.path == "/dupes":
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            action = body.get("action", "status")
            if action == "start":
                names = [n for n in ALLOW if self._may(n)]
                wanted = body.get("projects")
                if isinstance(wanted, list):
                    names = [n for n in names if n in wanted]
                tab = _jobs_tab(body)
                scope = {"projects": len(names)}
                def _du_start(names_in=list(names)):
                    return _dupes_start(list(names_in))
                def _du_status():
                    with _DUPES_LOCK:
                        return dict(_DUPES)
                def _du_stop():
                    with _DUPES_LOCK:
                        _DUPES["stop"] = True
                    _DUPES_STOP.set()
                try:
                    rec, started = localdocs_jobs.start_family(
                        "duplicate", scope, tab, True, _du_start,
                        _du_status, _du_stop, localdocs_jobs.normalize_duplicate)
                except Exception as e:
                    return self._send(_internal_error(e), 500)
                if rec is None and not started:
                    with _DUPES_LOCK:
                        legacy = dict(_DUPES)
                    return self._send(dict(legacy, started=False,
                        **_jobs_legacy_extra("duplicate", scope)), 200)
                if not started:
                    with _DUPES_LOCK:
                        legacy = dict(_DUPES)
                    return self._send(dict(legacy, started=False,
                        **_jobs_legacy_extra("duplicate", scope)), 200)
                with _DUPES_LOCK:
                    legacy = dict(_DUPES)
                return self._send(dict(legacy, started=True,
                    **_jobs_legacy_extra("duplicate", scope)), 202)
            if action == "stop":
                localdocs_jobs.family_stop("duplicate", None)
                with _DUPES_LOCK:
                    _DUPES["stop"] = True
                _DUPES_STOP.set()
            with _DUPES_LOCK:
                out = dict(_DUPES)
            out.update(_jobs_legacy_extra("duplicate", None))
            return self._send(out)
        if self.path == "/jobs":
            owner_ok = (not TOKENS) or self._is_owner()
            code, payload = localdocs_jobs.handle(body, True, owner_ok)
            return self._send(payload, code)
        if self.path == "/file-meta":
            project = body.get("project")
            if not isinstance(project, str):
                return self._send({"error": "invalid project"}, 400)
            if not self._may(project):
                return self._send({"error": "forbidden-project"}, 403)
            try:
                _meta_res = _file_meta(project, body.get("file"))
            except PermissionError:
                return self._send({"error": "file outside project"}, 403)
            except ValueError as e:
                return self._send({"error": "invalid-file", "code": "invalid-file", "detail": _safe_detail(e)}, 400)
            except Exception as e:
                return self._send(_internal_error(e), 500)
            try:
                _audit_click(project, _audit_click_rel(project, body.get("file") if isinstance(body.get("file"), str) else ""),
                             body.get("rank_pos", body.get("rank")), body.get("mission_id", body.get("mission")), body.get("query", ""))
            except Exception:
                pass
            return self._send(_meta_res)
        if self.path == "/ocr-file":
            project = body.get("project")
            if not isinstance(project, str):
                return self._send({"error": "invalid project"}, 400)
            if not self._may(project):
                return self._send({"error": "forbidden-project"}, 403)
            try:
                return self._send(_ocr_file(project, body.get("file"), body.get("mode", "text")))
            except PermissionError:
                return self._send({"error": "file outside project"}, 403)
            except ValueError as e:
                return self._send({"error": "invalid-file", "code": "invalid-file", "detail": _safe_detail(e)}, 400)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if self.path.startswith("/api/update/"):
            if not self._is_loopback_peer():
                return self._send({"error": "loopback only"}, 403)
            import localdocs_updater
            # /api/update/rollback + /api/update/backup_status handled by updater.handle (single-active, atomic).
            status, result = localdocs_updater.handle(self.server, self.path, body)
            return self._send(result, status)
        if self.path == "/settings":
            if TOKENS and not self._is_owner():
                return self._send({"error": "owner-only"}, 403)
            try:
                return self._send(_settings.update(body))
            except (ValueError, KeyError, TypeError) as e:
                return self._send({"error": _safe_detail(e) or "invalid-settings"}, 400)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        if not isinstance(body.get("query", ""), str) or not isinstance(body.get("project", ""), str):
            return self._send({"error": "invalid-query-or-project"}, 400)
        try:
            limit = int(body.get("limit", 6))
            if not 1 <= limit <= 100:
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            return self._send({"error": "limit-must-be-1-to-100"}, 400)
        if self.path == "/quick-search":
            _ensure_prewarm()
            query = body.get("query", "")
            try:
                roots = self._quick_roots(body)
            except ValueError as e:
                return self._send({"error": str(e)}, 400)
            if roots == []:
                return self._send([])
            try:
                res = _quick().search(query, limit, only_roots=roots)
            except Exception as e:
                return self._send(_internal_error(e), 500)
            try:
                _corrected = getattr(res, "corrected_from", None)
            except Exception:
                _corrected = None
            try:
                _hints = list(getattr(res, "hints", None) or [])
            except Exception:
                _hints = []
            hits = list(res)
            try:
                hits = hits[:int(limit)]
            except Exception:
                pass
            by_folder = {_srv_norm_real(f): n
                         for n, f in ALLOW.items() if self._may(n)}
            import stat as _stat
            for h in hits:
                # Mark folders and fill size/date only for returned rows
                if h.get("path"):
                    try:
                        _st = os.stat(h["path"])
                        h["is_dir"] = _stat.S_ISDIR(_st.st_mode)
                        if not h["is_dir"] and not h.get("size"):
                            h["size"], h["mtime"] = _st.st_size, _st.st_mtime
                        if h["is_dir"]:
                            h["size"] = 0
                    except OSError:
                        pass
            for h in hits:
                h["project"] = None
                folder = _srv_norm_real(str(h.get("folder", "")))
                for root, n in by_folder.items():
                    if folder == root or folder.startswith(root.rstrip("\\/") + os.sep):
                        h["project"] = n
                        break
            if _corrected or _hints:
                return self._send({"hits": hits, "corrected_from": _corrected, "hints": _hints})
            return self._send(hits)
        if self.path == "/search_all":
            _ensure_prewarm()
            _maybe_auto_ocr()
            from project_search.backend import search_all
            try:
                names = [n for n in ALLOW if self._may(n)]
                wanted = body.get("projects")
                if isinstance(wanted, list):
                    wanted = {w for w in wanted if isinstance(w, str)}
                    names = [n for n in names if n in wanted]
                if not names:
                    return self._send([])
                from project_search.backend import search_project
                merged = []
                skipped = []
                per = min(limit, 40)
                from project_search import rerank as _rr
                precise = bool(body.get("precise"))
                # Operators (ext:, date:, size:, folder:, path:) filter content hits too, instead of being searched as words.
                from localdocs_fastindex import _parse_search_query
                content_query, spec = _parse_search_query(str(body.get("query", "")))
                if spec is None:
                    content_query = body.get("query", "")
                else:
                    per = min(limit * 5, 200)
                    if not content_query.strip():
                        if body.get("envelope"):
                            _raw_q = str(body.get("query", ""))
                            if not _raw_q.strip():
                                _st, _det = "bad_query", "empty query"
                            else:
                                _st, _det = "ok", "operator-only query; no content terms"
                            return self._send({"hits": [], "searched": len(names), "skipped": [],
                                               "status": _st, "status_detail": _det})
                        return self._send([])
                import concurrent.futures as _cf2
                import threading as _th2
                import project_search.neuro_vector_store as _nvs_mod
                _orig_embed = _nvs_mod.embed_texts
                _embed_cache = {}
                _embed_lock = _th2.Lock()

                def _once_embed(texts, kind="passage", model_name=None):
                    try:
                        is_target = (kind == "query" and isinstance(texts, list) and len(texts) == 1 and texts[0] == content_query)
                    except Exception:
                        is_target = False
                    if not is_target:
                        return _orig_embed(texts, kind, model_name)
                    key = (texts[0], kind, model_name)
                    with _embed_lock:
                        if key in _embed_cache:
                            return _embed_cache[key]
                        res = _orig_embed(texts, kind, model_name)
                        _embed_cache[key] = res
                        return res

                _nvs_mod.embed_texts = _once_embed
                try:
                    def _one_proj(_pn):
                        try:
                            with _rr.precise(precise):
                                _found = search_project(ALLOW[_pn], content_query, per)
                        except (ValueError, RuntimeError, OSError, IndexError, KeyError) as _exc:
                            return (_pn, None, _exc)
                        return (_pn, _found, None)

                    with _cf2.ThreadPoolExecutor(max_workers=min(8, len(names))) as _ex:
                        _futs = [_ex.submit(_one_proj, _n) for _n in names]
                        for _n, _f in zip(names, _futs):
                            try:
                                _pn, _found, _exc = _f.result()
                            except (ValueError, RuntimeError, OSError, IndexError, KeyError) as _exc:
                                skipped.append({"project": _n, "reason": "not-indexed" if isinstance(_exc, ValueError) else "error"})
                                continue
                            except Exception as _exc:
                                skipped.append({"project": _n, "reason": "error"})
                                continue
                            if _exc is not None:
                                skipped.append({"project": _n, "reason": "not-indexed" if isinstance(_exc, ValueError) else "error"})
                                continue
                            for h in _found:
                                h["project"] = ALLOW[_n]
                                h["project_name"] = _n
                                merged.append(h)
                finally:
                    _nvs_mod.embed_texts = _orig_embed
                merged.sort(key=lambda h: h.get("score", 0), reverse=True)
                # one-line page leftovers ("ch", "lo od ...") are not readable evidence: drop them when
                # a real passage of the same file (or any other file) is available
                solid = [h for h in merged if len((h.get("snippet") or "").strip()) >= 30]
                if solid:
                    merged = solid
                _audit("*", body.get("query", ""), len(merged))
                if spec is None:
                    out = _with_matched(merged[:limit], content_query)
                    for h in out:
                        try:
                            st = os.stat(h["file"])
                            h["size"], h["mtime"] = st.st_size, st.st_mtime
                        except (OSError, KeyError):
                            pass
                else:
                    from localdocs_filters import matches as _spec_matches
                    cands = _with_matched(merged[:limit * 5], content_query)
                    out = []
                    for h in cands:
                        try:
                            st = os.stat(h["file"])
                            h["size"], h["mtime"] = st.st_size, st.st_mtime
                        except (OSError, KeyError):
                            pass
                        try:
                            ok = _spec_matches(spec, {"path": h.get("file", ""), "name": os.path.basename(h.get("file", "")),
                                                      "size": int(h.get("size", -1)), "mtime": float(h.get("mtime", -1))})
                        except Exception:
                            ok = False
                        if ok:
                            out.append(h)
                            if len(out) >= limit:
                                break
                if body.get("envelope"):
                    from project_search import backend as _search_be
                    if out:
                        try:
                            _rr = _search_be.route(content_query)
                        except Exception:
                            _rr = "semantic"
                        _st, _det = "ok", "route=%s, %d hit(s)" % (_rr, len(out))
                    else:
                        _ok_folders = [ALLOW[n] for n in names
                                       if all(s.get("project") != n for s in skipped)]
                        _st, _det = _search_be.federation_search_status(_ok_folders, content_query)
                    return self._send({"hits": out, "searched": len(names) - len(skipped),
                                       "skipped": skipped, "status": _st, "status_detail": _det})
                return self._send(out)
            except Exception as e:
                return self._send(_internal_error(e), 500)
        proj = body.get("project")
        if not self._may(proj): return self._send({"error": "forbidden-project"}, 403)
        try:
            if self.path == "/search":
                from project_search.backend import search_project, search_status
                try:
                    hits = _with_matched(search_project(ALLOW[proj], body.get("query", ""), limit), body.get("query", ""))
                except ValueError as e:
                    return self._send({"error": "not-indexed", "code": "not-indexed",
                                       "detail": _safe_detail(e), "project": proj,
                                       "status": "engine_unavailable"}, 404)
                except (RuntimeError, OSError, IndexError, KeyError) as e:
                    # Engine/lock failures must not leak internal text (paths,
                    # quotas, private markers): log full repr server-side,
                    # return a generic client detail.
                    _eid = _internal_error(e).get("id")
                    return self._send({"error": "search-error", "code": "search-error",
                                       "detail": "search failed", "id": _eid, "project": proj,
                                       "status": "engine_unavailable"}, 400)
                _audit(proj, body.get("query", ""), len(hits))
                if body.get("envelope"):
                    st, det = search_status(ALLOW[proj], body.get("query", ""), hits)
                    return self._send({"hits": hits, "status": st, "status_detail": det})
                return self._send(hits)
            if self.path == "/index":
                from project_search.backend import index_project_chunked, IndexLimitError
                # Fourth content-index path (sync, blocking): serialize with
                # the same per-project lock and phase, never beside a run.
                lk = _project_index_lock(proj)
                if not lk.acquire(blocking=False):
                    return self._send({"error": "busy", "code": "busy",
                                       "detail": "indexing already running"}, 409)
                beg = _try_begin_index(proj)
                if beg is None:
                    try:
                        lk.release()
                    except RuntimeError:
                        pass
                    return self._send({"error": "busy", "code": "busy",
                                       "detail": "indexing already running"}, 409)
                gen, _ev = beg
                failed = ""
                try:
                    return self._send(index_project_chunked(ALLOW[proj], group_files=2000))
                except IndexLimitError as e:
                    failed = _safe_detail(e) or "Достигнут настроенный лимит индекса. Выберите меньшую папку или измените LOCALDOCS_MAX_FILES / LOCALDOCS_MAX_CHUNKS."
                    with _INDEX_STATUS_LOCK:
                        if _CI_GEN.get(proj, 0) == gen and _INDEX_STATUS.get(proj, {}).get("phase") == "running":
                            _INDEX_STATUS[proj].update(phase="error", current="", error="index-limit",
                                finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
                    return self._send({"error": "index-limit", "code": "index-limit",
                                       "detail": failed}, 400)
                finally:
                    import datetime as _dt
                    with _INDEX_STATUS_LOCK:
                        if _CI_GEN.get(proj, 0) == gen and _INDEX_STATUS.get(proj, {}).get("phase") == "running":
                            _INDEX_STATUS[proj].update(phase="done", current="",
                                finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
                    try:
                        lk.release()
                    except RuntimeError:
                        pass
            if self.path == "/backup":
                try:
                    return self._send(_backup(proj))
                except ValueError as e:
                    return self._send({"error": "nothing-to-backup", "code": "nothing-to-backup",
                                       "detail": _safe_detail(e)}, 404)
        except Exception as e: return self._send(_internal_error(e), 500)
        return self._send({"error": "unknown", "code": "unknown"}, 404)
    def log_message(self, *a): pass
    def _is_owner(self):
        return self.headers.get("X-Token") == _OWNER_TOKEN and self._auth()
    def _is_loopback_peer(self):
        import ipaddress as _ipaddress
        try:
            return _ipaddress.ip_address(self.client_address[0]).is_loopback
        except ValueError:
            return False
_USE_HTTPS = os.environ.get("LOCALDOCS_HTTPS") == "1"
if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8798
    if os.name == "nt" and os.environ.get("LOCALDOCS_PRIORITY", "low") != "normal":
        try:  # a background search service must yield to the user's foreground apps (indexing/OCR can be heavy)
            import ctypes as _ct
            _k = _ct.windll.kernel32
            _k.GetCurrentProcess.restype = _ct.c_void_p
            _k.SetPriorityClass.argtypes = [_ct.c_void_p, _ct.c_uint32]
            _k.SetPriorityClass(_k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    if _USE_HTTPS:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(os.path.join(ROOT, "localdocs_cert.pem"),
                            os.path.join(ROOT, "localdocs_key.pem"))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(json.dumps({"port": port, "https": _USE_HTTPS, "projects": list(ALLOW)}), flush=True)
    _ensure_scheduler()
    _ensure_watcher()
    _events_sync()
    try:
        # Move undo journals of older versions out of the user's folders.
        import localdocs_fsapi as _fsapi_boot
        _fsapi_boot.prune_journals()
        for _root in list(ALLOW.values()):
            if isinstance(_root, str) and os.path.isdir(_root):
                _fsapi_boot.journal_path_for(os.path.realpath(_root))
    except Exception:
        pass
    import probe_licensing as _pl_boot
    _threading.Thread(target=_pl_boot.startup, name='probe-license-check', daemon=True).start()
    _prewarm()
    if "LOCALDOCS_WATCH_SEC" not in os.environ and _settings.get().get("watch_new_files", True):
        WATCH_SEC = 60.0  # pick up new files in added folders (cheap snapshot diff), then index/OCR them
        _ensure_watcher()
    try:
        import localdocs_models as _lm
        _lm.start_auto_update_check()
    except Exception:
        pass
    _threading.Thread(target=_auto_index_first_run, name="localdocs-first-index", daemon=True).start()
    srv.serve_forever()
