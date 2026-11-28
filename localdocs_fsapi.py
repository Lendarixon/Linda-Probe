"""LocalDocs owner-only file API over the fsops engine (P5 sandbox).

Scope: C:/Users/ninja/Desktop/LocalDocs/work/product_recovery_20261022/P5_api/dev
No integration in main. No GPU/indexes. No jobs: long copy/move are plain
synchronous engine calls.

Contract (rel-based, no absolute foreign paths, no tokens in payloads):
  POST /fs/list    authed + _may(project)      -> directory listing
  POST /fs/mkdir   owner-only                  -> create_folder
  POST /fs/rename  owner-only                  -> rename (same dir; use move to relocate)
  POST /fs/copy    owner-only                  -> copy
  POST /fs/move    owner-only                  -> move
  POST /fs/recycle owner-only                  -> delete_recycle (recycle-only)
  POST /fs/undo    owner-only                  -> undo via fsops journal

Auth gates live here in handle(): unauth -> 403 forbidden;
non-owner mutation -> 403 owner-only (same shape as /jobs).
Project containment uses localdocs_browse helpers (_check_root/_norm_rel/
_resolve/_check_inside); every mutation additionally goes through
FsEngine._resolve (allowlist, system-path, reparse, root-mutation) so a
lost race turns into ITEM_EXISTS, never an overwrite.

Collision: engine has NO overwrite policy. Default collision=fail ->
HTTP 409 ITEM_EXISTS. Explicit "collision": "skip" | "rename_new" is
honoured. Any "overwrite": true or "collision": "overwrite" is rejected
with 400 OVERWRITE_FORBIDDEN. overwrite=false (default) is accepted.

Bulk: {"project":..., "items":[...]} up to MAX_ITEMS. Mixed outcomes ->
HTTP 207 PARTIAL_SUCCESS with per-file items. Single-item failures map to
409/404/403/400 (see _code_to_http). Repeated undo is safe: second undo
reports NOT_FOUND/MODIFIED/NOT_EMPTY/DEST_EXISTS, never overwrites.

Recycle-only: only FsEngine delete_recycle is exposed; there is no
permanent-delete path. Undo of recycle returns UNSUPPORTED (manual
Restore from the OS bin), repeat is equally safe.

Responses never contain tokens or absolute foreign paths: src/dst are
project-relative ("/" separators), error strings are scrubbed of
drive-absolute segments ("[path]").
"""
from __future__ import annotations

import os
import re
import shutil
import threading

MAX_ITEMS = 100
MAX_LIST_LIMIT = 2000

_ROUTES = ("/fs/list", "/fs/mkdir", "/fs/rename", "/fs/copy",
           "/fs/move", "/fs/recycle", "/fs/undo", "/fs/preview")

_ENGINES = {}
_ENGINES_LOCK = threading.RLock()
_SHELL_FACTORY = None  # tests may inject: lambda root_real -> backend


def set_shell_factory(fn):
    """Inject shell backend factory for tests. Production leaves None."""
    global _SHELL_FACTORY
    _SHELL_FACTORY = fn


class FsApiError(Exception):
    def __init__(self, code, message, http=400):
        super().__init__(message)
        self.code = code
        self.http = http


def _scrub(msg, root_real=None, project=None):
    try:
        s = str(msg or "")
    except Exception:
        s = ""
    # Never leak tokens: strip anything that looks like a token field.
    # Absolute drive paths -> [path]; project root -> [project]/rel.
    if root_real:
        try:
            s = s.replace(root_real, "[project]")
            # normcase variant (lowercased drive) for Windows.
            s = s.replace(os.path.normcase(root_real), "[project]")
        except Exception:
            pass
    s = re.sub(r"[A-Za-z]:\\[^\s\"']*", "[path]", s)
    s = re.sub(r"[A-Za-z]:/[^\s\"']*", "[path]", s)
    # Absolute POSIX leftovers that escape the project.
    if "/tmp/" in s or "/var/" in s:
        s = re.sub(r"/[^\s\"']*(tmp|var)/[^\s\"']*", "[path]", s)
    return s[:500]


def _code_to_http(code):
    c = str(code or "")
    if c in ("ITEM_EXISTS", "DEST_EXISTS", "MODIFIED", "NOT_EMPTY"):
        return 409
    if c in ("NOT_FOUND", "UNKNOWN_OP"):
        return 404
    if c in ("OUTSIDE_ROOT", "ROOT_MUTATION", "SYSTEM_PATH",
             "REPARSE_POINT", "OWNER_DENIED", "RECYCLE_UNSUPPORTED",
             "ACCESS_DENIED", "FORBIDDEN", "FORBIDDEN_PROJECT",
             "OWNER_ONLY"):
        return 403
    if c in ("UNSUPPORTED", "UNSUPPORTED_UNDO"):
        return 400
    return 400


def _project_root(allow, project):
    import localdocs_browse as _b
    if not isinstance(project, str) or not project:
        raise FsApiError("INVALID_REQUEST", "project is required", 400)
    if not isinstance(allow, dict) or project not in allow:
        raise FsApiError("FORBIDDEN_PROJECT", "unknown project", 403)
    raw_root = allow[project]
    if not isinstance(raw_root, str) or not raw_root:
        raise FsApiError("INVALID_REQUEST", "bad project root", 400)
    try:
        root_real = _b._check_root(raw_root)
    except FileNotFoundError:
        raise FsApiError("NOT_FOUND", "project root not found", 404)
    except ValueError as exc:
        raise FsApiError("INVALID_REQUEST", "bad project root", 400)
    # System-root guard: refuse to serve a project rooted in the OS.
    try:
        from localdocs_fsops import _is_system_path as _issys
        if _issys(root_real):
            raise FsApiError("SYSTEM_PATH",
                             "project targets a protected system location", 403)
    except FsApiError:
        raise
    except Exception:
        pass
    # Drive-root guard: never serve C:\ itself.
    try:
        drv, tail = os.path.splitdrive(root_real)
        if not tail.strip("\\/"):
            raise FsApiError("SYSTEM_PATH", "drive roots are not allowed", 403)
    except FsApiError:
        raise
    except Exception:
        pass
    return root_real


def _canon_rel(root_real, rel_raw, what="rel"):
    """Canonicalize project-relative ref via browse helpers + reparse guard."""
    import localdocs_browse as _b
    rel_in = "" if rel_raw is None else rel_raw
    if not isinstance(rel_in, str):
        raise FsApiError("INVALID_REQUEST", "%s must be a string" % what, 400)
    try:
        rel_norm = _b._norm_rel(rel_in)
    except PermissionError as exc:
        raise FsApiError("OUTSIDE_ROOT", "path escapes project root", 403)
    except Exception as exc:
        raise FsApiError("INVALID_REQUEST", "bad %s" % what, 400)
    try:
        full = _b._resolve(root_real, rel_norm)
    except Exception as exc:
        raise FsApiError("OUTSIDE_ROOT", "path escapes project root", 403)
    try:
        _b._check_inside(root_real, full)
    except PermissionError:
        raise FsApiError("OUTSIDE_ROOT", "path escapes project root", 403)
    # System-path guard on the resolved target.
    try:
        from localdocs_fsops import _is_system_path as _issys
        if _issys(full):
            raise FsApiError("SYSTEM_PATH",
                             "target is a protected system location", 403)
    except FsApiError:
        raise
    except Exception:
        pass
    # Reparse guard on existing ancestors (clear error, no stacktrace).
    # Check the logical path first: realpath() would hide a symlink
    # component (link -> real becomes "real"), so "link" alone must be
    # rejected here. Then check the resolved target as well.
    logical = root_real if not rel_norm else os.path.normpath(
        os.path.join(root_real, rel_norm))
    _reject_reparse_ancestors(root_real, logical)
    _reject_reparse_ancestors(root_real, full)
    rel_out = _b._to_rel_str(rel_norm)
    return (full, rel_out)


def _reject_reparse_ancestors(root_real, full):
    """Reject symlink/reparse on existing ancestors root..full."""
    try:
        from localdocs_fsops import _is_reparse as _isrep
    except Exception:
        _isrep = None
    # Walk from root down to full over the existing prefix.
    try:
        rel = os.path.relpath(full, root_real)
    except Exception:
        raise FsApiError("OUTSIDE_ROOT", "path escapes project root", 403)
    if rel.startswith(".."):
        raise FsApiError("OUTSIDE_ROOT", "path escapes project root", 403)
    parts = [] if rel in (".", "") else rel.split(os.sep)
    cur = root_real
    # Root itself: if the project root is a link, refuse.
    try:
        if os.path.islink(cur) or (_isrep(cur) if _isrep else False):
            raise FsApiError("REPARSE_POINT",
                             "path crosses a symlink/reparse point", 403)
    except FsApiError:
        raise
    except Exception:
        pass
    for comp in parts:
        cur = os.path.join(cur, comp)
        try:
            exists = os.path.lexists(cur)
        except Exception:
            return
        if not exists:
            return
        try:
            if os.path.islink(cur) or (_isrep(cur) if _isrep else False):
                raise FsApiError("REPARSE_POINT",
                                 "path crosses a symlink/reparse point", 403)
        except FsApiError:
            raise
        except Exception:
            continue


def _abs_to_rel(root_real, abs_path):
    try:
        rel = os.path.relpath(abs_path, root_real)
    except Exception:
        return None
    if rel.startswith(".."):
        return None
    return rel.replace(os.sep, "/")


LEGACY_JOURNAL_NAME = ".localdocs-fs-journal.json"


def journal_path_for(root_real):
    """Undo journal lives in the app's data folder, never inside the user's folder
    (a file there showed up in listings and search results)."""
    import hashlib
    base = _journal_dir()
    os.makedirs(base, exist_ok=True)
    key = hashlib.sha1(os.path.normcase(root_real).encode("utf-8")).hexdigest()[:20]
    target = os.path.join(base, key + ".json")
    marker = os.path.join(base, key + ".root")
    try:
        if not os.path.exists(marker):
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(root_real)
    except OSError:
        pass
    legacy = os.path.join(root_real, LEGACY_JOURNAL_NAME)
    try:
        if os.path.isfile(legacy) and not os.path.lexists(target):
            shutil.move(legacy, target)
        elif os.path.isfile(legacy):
            os.remove(legacy)
        if os.path.isfile(legacy + ".lock"):
            os.remove(legacy + ".lock")
    except OSError:
        pass
    return target


def _journal_dir():
    return os.environ.get("LOCALDOCS_FSJOURNAL_DIR") or os.path.join(
        os.environ.get("LOCALDOCS_WRITE_DIR") or os.path.dirname(os.path.abspath(__file__)), "indexes", "_fsjournal")


def prune_journals():
    """Drop journals whose folder no longer exists (removed search folders, temporary test roots)."""
    base = _journal_dir()
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    removed = 0
    for name in names:
        key, ext = os.path.splitext(name)
        if ext != ".json":
            continue
        marker = os.path.join(base, key + ".root")
        try:
            with open(marker, encoding="utf-8") as fh:
                root = fh.read().strip()
        except OSError:
            root = ""
        if root and os.path.isdir(root):
            continue
        for leftover in (name, key + ".json.lock", key + ".root"):
            try:
                os.remove(os.path.join(base, leftover))
            except OSError:
                pass
        removed += 1
    return removed


def get_engine(root_real):
    """Cached FsEngine per project root (journal in the app data folder)."""
    from localdocs_fsops import FsEngine
    with _ENGINES_LOCK:
        eng = _ENGINES.get(root_real)
        if eng is not None:
            return eng
        journal = journal_path_for(root_real)
        if _SHELL_FACTORY is not None:
            try:
                backend = _SHELL_FACTORY(root_real)
            except Exception:
                backend = None
        else:
            backend = None
        kw = {"journal_path": journal, "owner_check": lambda p: True}
        if backend is not None:
            kw["shell_backend"] = backend
        eng = FsEngine([root_real], **kw)
        _ENGINES[root_real] = eng
        return eng


def reset_engines():
    with _ENGINES_LOCK:
        _ENGINES.clear()


def _check_collision_option(body):
    if not isinstance(body, dict):
        return "fail"
    if body.get("overwrite") is True:
        raise FsApiError("OVERWRITE_FORBIDDEN",
                         "overwrite is not supported; use collision=skip|rename_new",
                         400)
    coll = body.get("collision", "fail")
    if coll is None:
        return "fail"
    if coll == "overwrite":
        raise FsApiError("OVERWRITE_FORBIDDEN",
                         "overwrite is not supported; use collision=skip|rename_new",
                         400)
    if coll not in ("fail", "skip", "rename_new"):
        raise FsApiError("INVALID_REQUEST",
                         "unknown collision policy %r" % (coll,), 400)
    return coll


def _items_or_single(body, kind):
    """Normalize POST body to (items_list, collision). Raises FsApiError."""
    if not isinstance(body, dict):
        raise FsApiError("INVALID_REQUEST", "body must be an object", 400)
    collision = _check_collision_option(body)
    raw_items = body.get("items")
    if raw_items is not None:
        if not isinstance(raw_items, list) or not raw_items:
            raise FsApiError("INVALID_REQUEST",
                             "items must be a non-empty list", 400)
        if len(raw_items) > MAX_ITEMS:
            raise FsApiError("INVALID_REQUEST",
                             "too many items (max %d)" % MAX_ITEMS, 400)
        return (raw_items, collision)
    # Single-item shorthand per kind.
    if kind == "mkdir":
        rel = body.get("rel", body.get("path"))
        if not isinstance(rel, str) or not rel:
            raise FsApiError("INVALID_REQUEST",
                             "rel is required for mkdir", 400)
        return ([{"rel": rel}], collision)
    if kind == "rename":
        if not isinstance(body.get("src"), str):
            raise FsApiError("INVALID_REQUEST",
                             "src is required for rename", 400)
        return ([{k: body.get(k) for k in
                  ("src", "dst", "new_name") if body.get(k) is not None}],
                collision)
    if kind in ("copy", "move"):
        if not isinstance(body.get("src"), str) or \
                not isinstance(body.get("dst"), str):
            raise FsApiError("INVALID_REQUEST",
                             "src and dst are required", 400)
        return ([{"src": body.get("src"), "dst": body.get("dst")}],
                collision)
    if kind == "recycle":
        rel = body.get("rel", body.get("src", body.get("path")))
        if not isinstance(rel, str) or not rel:
            raise FsApiError("INVALID_REQUEST",
                             "rel is required for recycle", 400)
        return ([{"src": rel}], collision)
    raise FsApiError("INVALID_REQUEST", "bad request", 400)


def _sanitize_result(root_real, project, res):
    """Strip absolute paths/tokens from an engine result."""
    out_items = []
    for it in (res.get("items") or []):
        src = it.get("src")
        dst = it.get("dst")
        # src/dst may be absolute engine paths; convert when inside.
        if isinstance(src, str) and os.path.isabs(src):
            r = _abs_to_rel(root_real, src)
            src = r if r is not None else "[path]"
        if isinstance(dst, str) and os.path.isabs(dst):
            r = _abs_to_rel(root_real, dst)
            dst = r if r is not None else "[path]"
        err = it.get("error")
        if isinstance(err, str):
            err = _scrub(err, root_real, project)
        out_items.append({"src": src, "dst": dst,
                          "status": it.get("status"),
                          "code": it.get("code"), "error": err})
    out = {"op_id": res.get("op_id"), "action": res.get("action"),
           "status": res.get("status"), "items": out_items}
    if res.get("error"):
        out["error"] = _scrub(res.get("error"), root_real, project)
    for k in ("journal_error", "journal_pending"):
        if res.get(k):
            out[k] = _scrub(str(res.get(k))[:200], root_real, project)
    return out


def _http_for_result(res):
    status = res.get("status")
    if status == "SUCCESS":
        return 200
    if status == "PARTIAL_SUCCESS":
        return 207
    if status == "UNSUPPORTED":
        return 400
    items = res.get("items") or []
    if len(items) == 1:
        return _code_to_http(items[0].get("code"))
    if items:
        codes = {i.get("code") for i in items}
        if len(codes) == 1:
            return _code_to_http(next(iter(codes)))
    return 400


# ---- route handlers (pure; auth flags come from the server) ----

def handle_list(allow, body):
    project = body.get("project") if isinstance(body, dict) else None
    root_real = _project_root(allow, project)
    rel_raw = body.get("rel", "")
    show_hidden = bool(body.get("show_hidden", False))
    try:
        limit = int(body.get("limit", MAX_LIST_LIMIT))
    except Exception:
        raise FsApiError("INVALID_REQUEST", "bad limit", 400)
    limit = max(1, min(limit, MAX_LIST_LIMIT))
    full, rel_out = _canon_rel(root_real, rel_raw, "rel")
    import localdocs_browse as _b
    try:
        out = _b.list_dir(root_real, rel_out, limit=limit,
                          show_hidden=show_hidden)
    except FileNotFoundError:
        raise FsApiError("NOT_FOUND", "directory not found", 404)
    except PermissionError:
        raise FsApiError("OUTSIDE_ROOT", "path escapes project root", 403)
    except (ValueError, OSError) as exc:
        raise FsApiError("INVALID_REQUEST",
                         _scrub(str(exc)[:200], root_real, project), 400)
    try:
        crumbs = _b.breadcrumbs(root_real, rel_out)
    except PermissionError:
        raise FsApiError("OUTSIDE_ROOT", "path escapes project root", 403)
    except Exception as exc:
        raise FsApiError("INVALID_REQUEST", "bad rel", 400)
    # Drop the absolute "path"; keep only rel-scoped fields.
    entries = []
    for e in (out.get("entries") or []):
        if e.get("name") == LEGACY_JOURNAL_NAME:
            continue  # our own undo journal from older versions is not a user file
        entries.append({k: e.get(k) for k in
                        ("name", "path", "is_dir", "size", "mtime", "ext")})
    return (200, {"project": project, "rel": out.get("rel", rel_out),
                  "parent": out.get("parent"), "entries": entries,
                  "breadcrumbs": crumbs})


def _run_mutation(allow, body, action):
    project = body.get("project") if isinstance(body, dict) else None
    root_real = _project_root(allow, project)
    raw_list, collision = _items_or_single(body, action)
    eng_items = []
    # Keep rel mapping for sanitized responses.
    rel_pairs = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise FsApiError("INVALID_REQUEST",
                             "each item must be an object", 400)
        if action == "mkdir":
            rel = entry.get("rel", entry.get("path"))
            if not isinstance(rel, str) or not rel:
                raise FsApiError("INVALID_REQUEST",
                                 "each item needs rel", 400)
            full, rel_out = _canon_rel(root_real, rel, "rel")
            eng_items.append({"path": full})
            rel_pairs.append((rel, None))
        elif action == "rename":
            src_rel = entry.get("src")
            if not isinstance(src_rel, str) or not src_rel:
                raise FsApiError("INVALID_REQUEST",
                                 "each item needs src", 400)
            src_full, src_out = _canon_rel(root_real, src_rel, "src")
            if entry.get("dst") is not None:
                if not isinstance(entry.get("dst"), str):
                    raise FsApiError("INVALID_REQUEST", "bad dst", 400)
                dst_full, dst_out = _canon_rel(root_real, entry.get("dst"),
                                               "dst")
                eng_items.append({"src": src_full, "dst": dst_full})
                rel_pairs.append((src_rel, entry.get("dst")))
            elif entry.get("new_name") is not None:
                nm = entry.get("new_name")
                if not isinstance(nm, str) or not nm or \
                        any(c in nm for c in "\\/:"):
                    raise FsApiError("INVALID_REQUEST",
                                     "new_name must be a plain file name",
                                     400)
                parent = os.path.dirname(src_full)
                dst_full = os.path.join(parent, nm)
                # Validate the composed dst through the same canon gate.
                dst_rel_guess = _abs_to_rel(root_real, dst_full)
                if dst_rel_guess is None:
                    raise FsApiError("OUTSIDE_ROOT",
                                     "path escapes project root", 403)
                _canon_rel(root_real, dst_rel_guess, "dst")
                eng_items.append({"src": src_full, "new_name": nm})
                rel_pairs.append((src_rel, nm))
            else:
                raise FsApiError("INVALID_REQUEST",
                                 "rename needs dst or new_name", 400)
        elif action in ("copy", "move"):
            s = entry.get("src")
            d = entry.get("dst")
            if not isinstance(s, str) or not isinstance(d, str):
                raise FsApiError("INVALID_REQUEST",
                                 "each item needs src and dst", 400)
            s_full, _ = _canon_rel(root_real, s, "src")
            d_full, _ = _canon_rel(root_real, d, "dst")
            eng_items.append({"src": s_full, "dst": d_full})
            rel_pairs.append((s, d))
        elif action == "recycle":
            s = entry.get("src", entry.get("rel", entry.get("path")))
            if not isinstance(s, str) or not s:
                raise FsApiError("INVALID_REQUEST",
                                 "each item needs src", 400)
            s_full, _ = _canon_rel(root_real, s, "src")
            eng_items.append({"src": s_full})
            rel_pairs.append((s, None))
        else:
            raise FsApiError("INVALID_REQUEST", "bad action", 400)
    eng = get_engine(root_real)
    engine_action = {"mkdir": "create_folder"}.get(action, action)
    try:
        res = eng.execute(engine_action, eng_items,
                          {"collision": collision})
    except Exception as exc:
        raise FsApiError("INVALID_REQUEST", "engine rejected batch", 400)
    clean = _sanitize_result(root_real, project, res)
    # Replace absolute-derived src/dst with the original rels where the
    # engine echoed raw absolute inputs (keeps responses rel-only).
    # For rename_new successes the engine dst differs: keep resolved rel.
    fixed = []
    for idx, it in enumerate(clean.get("items") or []):
        o_src, o_dst = rel_pairs[idx] if idx < len(rel_pairs) else (None, None)
        code = it.get("code")
        if it.get("status") == "OK" and action in ("copy", "move", "rename",
                                                   "mkdir"):
            # Success: dst is the canonical outcome; re-derive rel from
            # the engine record when possible, else keep requested rel.
            try:
                rec_items = None
                # engine journal dst is absolute; clean already converted.
                pass
            except Exception:
                pass
            if o_src is not None and it.get("src") in (None, "[path]"):
                it["src"] = o_src
            # mkdir success: dst holds the created rel already.
            # rename_new success: keep engine-resolved dst rel.
            if it.get("dst") in (None, "[path]") and o_dst is not None:
                it["dst"] = o_dst
        else:
            # Failure/skip: echo requested rels, never absolutes.
            if o_src is not None:
                it["src"] = o_src
            if action in ("copy", "move", "rename") and o_dst is not None:
                if it.get("dst") in (None, "[path]") or os.path.isabs(
                        str(it.get("dst") or "")):
                    it["dst"] = o_dst
        fixed.append(it)
    clean["items"] = fixed
    clean["project"] = project
    return (_http_for_result(clean), clean)


def handle_undo(allow, body):
    project = body.get("project") if isinstance(body, dict) else None
    root_real = _project_root(allow, project)
    op_id = body.get("op_id") if isinstance(body, dict) else None
    if not isinstance(op_id, str) or not op_id or len(op_id) > 128:
        raise FsApiError("INVALID_REQUEST", "op_id is required", 400)
    # op_id charset guard (journal hex ids; be lenient but bounded).
    if not re.match(r"^[A-Za-z0-9_.-]+$", op_id):
        raise FsApiError("INVALID_REQUEST", "bad op_id", 400)
    eng = get_engine(root_real)
    try:
        res = eng.undo(op_id)
    except Exception:
        raise FsApiError("INVALID_REQUEST", "undo failed", 400)
    clean = _sanitize_result(root_real, project, res)
    clean["project"] = project
    code = _http_for_result(clean)
    # Unknown op -> 404 (not a generic 400).
    items = clean.get("items") or []
    if res.get("status") == "FAILED" and not items and \
            "UNKNOWN_OP" in str(res.get("error") or ""):
        clean["code"] = "UNKNOWN_OP"
        return (404, clean)
    return (code, clean)


def handle(path, allow, auth_ok, owner_ok, may_fn, body):
    """Single dispatch for /fs/* used by localdocs_server (thin wrapper).

    may_fn(project) -> bool mirrors H._may. No tokens are read or echoed.
    Never raises: unexpected errors become 400/500 JSON without traceback
    or absolute paths.
    """
    try:
        if path not in _ROUTES:
            return (404, {"error": "unknown"})
        if not isinstance(body, dict):
            return (400, {"error": "body-must-be-object"})
        if path == "/fs/list":
            if not auth_ok:
                return (403, {"error": "forbidden"})
            proj = body.get("project")
            try:
                ok = bool(may_fn(proj)) if callable(may_fn) else True
            except Exception:
                ok = False
            if not ok:
                return (403, {"error": "forbidden-project"})
            try:
                return handle_list(allow, body)
            except FsApiError as exc:
                return (exc.http, {"error": _scrub(exc.args[0]),
                                  "code": exc.code})
            except PermissionError:
                return (403, {"error": "outside project", "code": "OUTSIDE_ROOT"})
            except FileNotFoundError:
                return (404, {"error": "not found", "code": "NOT_FOUND"})
            except Exception:
                return (400, {"error": "bad request", "code": "INVALID_REQUEST"})
        if path == "/fs/preview":
            if not auth_ok:
                return (403, {"error": "forbidden"})
            proj = body.get("project")
            try:
                ok = bool(may_fn(proj)) if callable(may_fn) else True
            except Exception:
                ok = False
            if not ok:
                return (403, {"error": "forbidden-project"})
            try:
                return handle_preview(allow, body)
            except FsApiError as exc:
                return (exc.http, {"error": _scrub(exc.args[0]),
                                   "code": exc.code})
            except PermissionError:
                return (403, {"error": "outside project", "code": "OUTSIDE_ROOT"})
            except FileNotFoundError:
                return (404, {"error": "not found", "code": "NOT_FOUND"})
            except Exception:
                return (400, {"error": "bad request", "code": "INVALID_REQUEST"})
        # Mutations: owner-only (same shape as /jobs).
        if not auth_ok:
            return (403, {"error": "forbidden"})
        if not owner_ok:
            return (403, {"error": "owner-only"})
        try:
            if path == "/fs/undo":
                return handle_undo(allow, body)
            action = {"/fs/mkdir": "mkdir"}.get(path, path.rsplit("/", 1)[-1])
            return _run_mutation(allow, body, action)
        except FsApiError as exc:
            return (exc.http, {"error": _scrub(exc.args[0]),
                              "code": exc.code})
        except PermissionError:
            return (403, {"error": "outside project", "code": "OUTSIDE_ROOT"})
        except FileNotFoundError:
            return (404, {"error": "not found", "code": "NOT_FOUND"})
        except Exception:
            return (400, {"error": "bad request", "code": "INVALID_REQUEST"})
    except Exception:
        return (400, {"error": "bad request"})


# ---- preview (read-only panel for the Files tab) ----
#
# New branch only: kind detection by extension + magic signature,
# size limits, truncation for text/code. No mutations, no open-file,
# no tokens or absolute paths in responses. Text/code travel as plain
# JSON strings; the UI must insert them via textContent (never HTML).

PV_LIMIT_IMAGE = 25 * 1024 * 1024
PV_LIMIT_TEXT = 1 * 1024 * 1024
PV_LIMIT_PDF = 50 * 1024 * 1024
PV_LIMIT_MEDIA = 200 * 1024 * 1024

PV_IMAGE_EXT = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "gif": "image/gif",
                "webp": "image/webp", "bmp": "image/bmp"}
PV_PDF_EXT = {"pdf": "application/pdf"}
PV_TEXT_EXT = {"txt", "md", "log", "csv", "json", "xml"}
PV_CODE_EXT = {"py", "js", "c", "h", "css"}
PV_VIDEO_EXT = {"mp4": "video/mp4", "webm": "video/webm"}
PV_AUDIO_EXT = {"mp3": "audio/mpeg", "wav": "audio/wav",
                "ogg": "audio/ogg"}
PV_OFFICE_EXT = {"docx", "xlsx", "pptx"}
# Never rendered inline: active content or executables go straight to
# kind=none with a human reason + an "open in program" button.
PV_DENY_EXT = {"svg", "html", "htm", "exe", "bat", "ps1", "psm1",
               "cmd", "com", "scr", "msi", "vbs", "vbe", "jar",
               "dll", "lnk"}


def _pv_none(project, rel_out, name, size, code, reason):
    return (200, {"project": project, "rel": rel_out, "kind": "none",
                  "name": name, "size": size, "truncated": False,
                  "code": code, "reason": reason})


def _pv_sig_ok(ext, head):
    try:
        if ext == "png":
            return head[:8] == b"\x89PNG\r\n\x1a\n"
        if ext in ("jpg", "jpeg"):
            return head[:3] == b"\xff\xd8\xff"
        if ext == "gif":
            return head[:6] in (b"GIF87a", b"GIF89a")
        if ext == "webp":
            return head[:4] == b"RIFF" and head[8:12] == b"WEBP"
        if ext == "bmp":
            return head[:2] == b"BM"
        if ext == "pdf":
            return head[:5] == b"%PDF-"
    except Exception:
        return False
    return False


def _pv_png_size(data):
    try:
        if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
            import struct as _st
            w, h = _st.unpack(">II", data[16:24])
            if 0 < w <= 100000 and 0 < h <= 100000:
                return (w, h)
    except Exception:
        pass
    return (None, None)


def handle_preview(allow, body):
    """POST /fs/preview {project, rel} -> preview payload (auth+may gated)."""
    import base64 as _b64
    project = body.get("project") if isinstance(body, dict) else None
    root_real = _project_root(allow, project)
    rel_raw = body.get("rel", body.get("path", "")) \
        if isinstance(body, dict) else ""
    if not isinstance(rel_raw, str) or not rel_raw:
        raise FsApiError("INVALID_REQUEST", "rel is required", 400)
    full, rel_out = _canon_rel(root_real, rel_raw, "rel")
    name = os.path.basename(full)
    try:
        is_dir = os.path.isdir(full)
    except Exception:
        raise FsApiError("INVALID_REQUEST", "bad rel", 400)
    if is_dir:
        return _pv_none(project, rel_out, name, 0, "PREVIEW_IS_DIR",
                         "Это папка. Файлы внутри видны в списке выше.")
    try:
        size = os.path.getsize(full)
    except OSError:
        raise FsApiError("NOT_FOUND", "file not found", 404)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in PV_DENY_EXT:
        if ext in ("svg", "html", "htm"):
            reason = ("Этот файл не показываем прямо здесь: внутри "
                      "может быть активный код. Нажмите "
                      "«Открыть в программе».")
        else:
            reason = ("Такие файлы не показываем из-за безопасности. "
                      "Нажмите «Открыть в программе».")
        return _pv_none(project, rel_out, name, size,
                         "PREVIEW_DENIED", reason)
    if ext in PV_IMAGE_EXT:
        if size > PV_LIMIT_IMAGE:
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_TOO_BIG",
                "Картинка слишком большая для предпросмотра "
                "(больше 25 МБ). Нажмите «Открыть в программе».")
        try:
            with open(full, "rb") as fh:
                data = fh.read(PV_LIMIT_IMAGE + 1)
        except OSError:
            raise FsApiError("NOT_FOUND", "file not found", 404)
        if len(data) > PV_LIMIT_IMAGE:
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_TOO_BIG",
                "Картинка слишком большая для предпросмотра "
                "(больше 25 МБ). Нажмите «Открыть в программе».")
        if not _pv_sig_ok(ext, data[:12]):
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_BROKEN",
                "Файл похож на битый: картинка не открывается. "
                "Попробуйте «Открыть в программе».")
        w, h = _pv_png_size(data) if ext == "png" else (None, None)
        out = {"project": project, "rel": rel_out, "kind": "image",
               "name": name, "size": size, "truncated": False,
               "mime": PV_IMAGE_EXT[ext],
               "data": _b64.b64encode(data).decode("ascii"),
               "code": "PREVIEW_OK",
               "reason": "Картинка. Не понравилась — закройте крестиком."}
        if w is not None:
            out["width"] = w
            out["height"] = h
        return (200, out)
    if ext in PV_PDF_EXT:
        if size > PV_LIMIT_PDF:
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_TOO_BIG",
                "Документ слишком большой для предпросмотра "
                "(больше 50 МБ). Нажмите «Открыть в программе».")
        try:
            with open(full, "rb") as fh:
                head = fh.read(6)
        except OSError:
            raise FsApiError("NOT_FOUND", "file not found", 404)
        if not _pv_sig_ok("pdf", head):
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_BROKEN",
                "Файл похож на битый: документ не читается. "
                "Попробуйте «Открыть в программе».")
        try:
            with open(full, "rb") as fh:
                data = fh.read(PV_LIMIT_PDF + 1)
        except OSError:
            raise FsApiError("NOT_FOUND", "file not found", 404)
        if len(data) > PV_LIMIT_PDF:
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_TOO_BIG",
                "Документ слишком большой для предпросмотра "
                "(больше 50 МБ). Нажмите «Открыть в программе».")
        return (200, {"project": project, "rel": rel_out, "kind": "pdf",
                      "name": name, "size": size, "truncated": False,
                      "mime": "application/pdf",
                      "data": _b64.b64encode(data).decode("ascii"),
                      "code": "PREVIEW_OK",
                      "reason": "Документ. Листайте прямо здесь."})
    if ext in PV_TEXT_EXT or ext in PV_CODE_EXT:
        kind = "text" if ext in PV_TEXT_EXT else "code"
        try:
            with open(full, "rb") as fh:
                data = fh.read(PV_LIMIT_TEXT + 1)
        except OSError:
            raise FsApiError("NOT_FOUND", "file not found", 404)
        truncated = len(data) > PV_LIMIT_TEXT
        if truncated:
            data = data[:PV_LIMIT_TEXT]
        if b"\x00" in data[:65536]:
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_NOT_TEXT",
                "Файл не похож на текст. Нажмите "
                "«Открыть в программе».")
        text = data.decode("utf-8", errors="replace")
        if truncated:
            reason = ("Показано начало файла (первые ~1 МБ из %s). "
                      "Целиком — через «Открыть в программе»."
                      % _pv_hsize(size))
        else:
            reason = "Текст файла. Читайте прямо здесь."
        return (200, {"project": project, "rel": rel_out, "kind": kind,
                      "name": name, "size": size, "truncated": truncated,
                      "mime": "text/plain; charset=utf-8", "text": text,
                      "code": "PREVIEW_TRUNCATED" if truncated
                      else "PREVIEW_OK", "reason": reason})
    if ext in PV_VIDEO_EXT or ext in PV_AUDIO_EXT:
        is_video = ext in PV_VIDEO_EXT
        if size > PV_LIMIT_MEDIA:
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_TOO_BIG",
                "Файл слишком большой для предпросмотра "
                "(больше 200 МБ). Нажмите «Открыть в программе».")
        try:
            with open(full, "rb") as fh:
                data = fh.read(PV_LIMIT_MEDIA + 1)
        except OSError:
            raise FsApiError("NOT_FOUND", "file not found", 404)
        if len(data) > PV_LIMIT_MEDIA:
            return _pv_none(
                project, rel_out, name, size, "PREVIEW_TOO_BIG",
                "Файл слишком большой для предпросмотра "
                "(больше 200 МБ). Нажмите «Открыть в программе».")
        mime = PV_VIDEO_EXT.get(ext) or PV_AUDIO_EXT.get(ext)
        return (200, {"project": project, "rel": rel_out,
                      "kind": "video" if is_video else "audio",
                      "name": name, "size": size, "truncated": False,
                      "mime": mime,
                      "data": _b64.b64encode(data).decode("ascii"),
                      "code": "PREVIEW_OK",
                      "reason": "Нажмите кнопку запуска ниже."})
    if ext in PV_OFFICE_EXT:
        return (200, {"project": project, "rel": rel_out, "kind": "office",
                      "name": name, "size": size, "truncated": False,
                      "code": "PREVIEW_OK",
                      "reason": "Это документ Office (%s). Содержимое "
                      "показываем значком, а прочитать можно через "
                      "«Открыть в программе»." % _pv_hsize(size)})
    return _pv_none(project, rel_out, name, size, "PREVIEW_UNSUPPORTED",
                    "Пока не умеем показывать такие файлы. Нажмите "
                    "«Открыть в программе».")


def _pv_hsize(n):
    try:
        n = int(n)
    except Exception:
        return "?"
    if n < 1024:
        return "%d Б" % n
    if n < 1024 * 1024:
        return "%.1f КБ" % (n / 1024.0)
    if n < 1024 * 1024 * 1024:
        return "%.1f МБ" % (n / 1048576.0)
    return "%.1f ГБ" % (n / 1073741824.0)
