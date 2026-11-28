"""LocalDocs quick file-name index (ADDENDUM v2 backend contract).

Stdlib-only in-memory index of file names under a set of roots.
Used by POST /quick-search; refreshed in the background.

Contract (work/spec_commercial.md, ADDENDUM v2):
  roots = Desktop/Documents/Downloads/Videos/Music of the user profile
          + every added project folder;
  skips .git, node_modules, AppData, Windows, Program Files,
  $Recycle.Bin, dot-dirs, symlink loops; max 600k entries;
  searchable while the first scan is still running (partial results).
"""

import os
import threading

__all__ = ["QuickIndex", "default_roots"]

# Directory names never descended into (compared case-insensitively,
# because the main platform is Windows).
_SKIP_NAMES = frozenset({
    ".git",
    "node_modules",
    "appdata",
    "windows",
    "program files",
    "program files (x86)",
    "$recycle.bin",
    # developer / cache clutter that drowns real documents in a whole-PC file-name search
    "__pycache__", "venv", ".venv", "site-packages", "lib", "dist", "build", "target",
    "indexes", "logs", "cache", ".cache", "vendor", "_vendor", "node_modules.bak",
    "models", "models_onnx", "tessdata", "licenses", "world_backups", "workspaces",
    "portable moved with spaces", "$windows.~bt", "system volume information",
})

# Binary / generated file types that are never what a person is looking for by name.
_SKIP_EXT = frozenset({
    ".pyc", ".pyo", ".map", ".dll", ".pdb", ".obj", ".o", ".so", ".lib", ".class",
    ".tmp", ".lock", ".lnk.tmp", ".idx", ".pack", ".faiss", ".npy", ".onnx", ".safetensors",
})

_RESCAN_INTERVAL_S = 600.0
_PUBLISH_EVERY = 2000

# Root realpath cache: only_roots arrive as raw strings per request;
# resolving them via os.path.realpath on every hit was a syscall storm.
_REALPATH_CACHE: dict = {}
_REALPATH_LOCK = threading.Lock()


def _cached_realpath(path):
    """Cached os.path.realpath (roots are few, stable; avoids per-hit syscalls)."""
    try:
        key = os.path.normcase(os.path.normpath(str(path)))
    except Exception:
        try:
            return os.path.realpath(path)
        except Exception:
            return ""
    with _REALPATH_LOCK:
        hit = _REALPATH_CACHE.get(key)
        if hit is not None:
            return hit
    try:
        real = os.path.realpath(path)
    except Exception:
        return ""
    if not real:
        return ""
    with _REALPATH_LOCK:
        if len(_REALPATH_CACHE) < 8192:
            _REALPATH_CACHE[key] = real
        return _REALPATH_CACHE.get(key, real)


def _norm_real(path):
    """normcase(normpath(cached-realpath(path))) for allowed-set compares."""
    try:
        return os.path.normcase(os.path.normpath(_cached_realpath(path)))
    except Exception:
        return ""


def _is_under_allowed(entry_path, allowed):
    """Prefix check without syscalls (entry paths are already under realpath roots)."""
    try:
        fp = os.path.normcase(os.path.normpath(entry_path))
    except Exception:
        return False
    for a in allowed:
        try:
            if fp == a or fp.startswith(a.rstrip(os.sep) + os.sep):
                return True
        except Exception:
            continue
    return False

# Smart-search fallback (P4b): typo tolerance only when exact finds nothing.
_FUZZY_CAP = 30000
_FUZZY_CAP_HINT = "слишком много файлов для проверки опечаток"
_FUZZY_EMPTY_HINT = "по имени ничего не найдено"


class _SearchResult(list):
    """List of hits with fallback meta (backward compatible: == [] when empty)."""

    def __init__(self, items, corrected_from=None, hints=None):
        super().__init__(items if items is not None else [])
        self.corrected_from = corrected_from
        self.hints = list(hints) if hints else []


def _norm(path):
    """Normalize a path for comparison: abspath + normpath + normcase."""
    try:
        p = os.path.abspath(os.path.expanduser(str(path)))
    except Exception:
        return ""
    return os.path.normcase(os.path.normpath(p))


def default_roots(project_folders=None):
    """Return existing user-profile roots + project folders.

    Candidates: Desktop/Documents/Downloads/Videos/Music under the user
    profile (os.path.expanduser('~'), with USERPROFILE/HOMEDRIVE fallbacks
    on Windows) plus every folder from *project_folders*.
    Result is deduplicated and contains no nested duplicates (the outermost
    path wins). Only existing directories are returned.
    """
    candidates = []
    homes = []
    try:
        homes.append(os.path.expanduser("~"))
    except Exception:
        pass
    for env in ("USERPROFILE",):
        val = os.environ.get(env)
        if val:
            homes.append(val)
    hd, hp = os.environ.get("HOMEDRIVE"), os.environ.get("HOMEPATH")
    if hd and hp:
        homes.append(os.path.join(hd, hp))
    seen_home = set()
    for home in homes:
        if not home:
            continue
        key = _norm(home)
        if not key or key in seen_home:
            continue
        seen_home.add(key)
        for sub in ("Desktop", "Documents", "Downloads", "Videos", "Music"):
            try:
                candidates.append(os.path.join(home, sub))
            except Exception:
                continue
    for folder in project_folders or []:
        try:
            if folder:
                candidates.append(str(folder))
        except Exception:
            continue

    # Normalize, keep existing dirs, deduplicate (order-preserving).
    uniq = []
    seen = set()
    for cand in candidates:
        try:
            real = os.path.realpath(cand)
        except Exception:
            continue
        key = _norm(real)
        if not key or key in seen:
            continue
        try:
            if not os.path.isdir(cand) and not os.path.isdir(real):
                continue
        except Exception:
            continue
        seen.add(key)
        uniq.append(os.path.normpath(real if os.path.isdir(real) else os.path.abspath(cand)))

    # Drop nested duplicates: parents first, keep outermost.
    uniq.sort(key=len)
    out = []
    for path in uniq:
        pkey = os.path.normcase(path)
        nested = False
        for kept in out:
            kkey = os.path.normcase(kept)
            if pkey == kkey or pkey.startswith(kkey + os.sep):
                nested = True
                break
        if not nested:
            out.append(path)
    return out



def _parse_query_filters(query):
    """Parse operators for both engines; size comparisons are strict via min/max_size_exclusive flags (21.10)."""
    import localdocs_filters as filters
    return filters.parse(query)


def _matches_query_filters(spec, hit):
    import localdocs_filters as filters
    need_size = spec['min_size'] is not None or spec['max_size'] is not None
    need_time = spec['min_t'] is not None or spec['max_t'] is not None
    missing_size = hit.get('size') is None or hit['size'] < 0
    missing_time = hit.get('mtime') is None or hit['mtime'] < 0
    if (need_size and missing_size) or (need_time and missing_time):
        try:
            st = os.stat(hit['path'])
        except (OSError, KeyError):
            return False
        hit.update(size=st.st_size, mtime=st.st_mtime)
    return filters.matches(spec, hit)

class QuickIndex:
    """Thread-safe in-memory file-name index over a set of roots.

    :param roots_provider: callable returning an iterable of root paths
        (called on every scan), or a static iterable of root paths.
    :param max_entries: cap on stored entries (default 600000).
    """

    def __init__(self, roots_provider, max_entries=600000):
        self._roots_provider = roots_provider
        try:
            self._max_entries = max(1, int(max_entries))
        except Exception:
            self._max_entries = 600000
        self._lock = threading.RLock()
        self._scan_lock = threading.Lock()
        self._entries = ()   # immutable tuple: snapshot is a reference, no per-search copy
        self._roots = []
        self._scanning = False
        self._done = False
        self._thread = None
        self._stop = threading.Event()
        self._blob_cache = None
        self._blob_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------

    def start(self):
        """Start the daemon background thread (periodic rescan, 600 s)."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        t = threading.Thread(target=self._loop, name="localdocs-quick", daemon=True)
        self._thread = t
        t.start()
        return t

    def stop(self):
        """Signal the background thread to stop (best effort)."""
        self._stop.set()
        return True

    def _loop(self):
        # Immediate first scan, then every 600 s.
        while not self._stop.is_set():
            try:
                with self._scan_lock:
                    self._scan()
            except Exception:
                pass
            if self._stop.wait(_RESCAN_INTERVAL_S):
                break

    def rescan_now(self):
        """Run one scan synchronously in the calling thread."""
        with self._scan_lock:
            return self._scan()

    def status(self):
        """Return {phase, files, roots}."""
        with self._lock:
            entries = len(self._entries)
            roots = list(self._roots)
            scanning = self._scanning
            done = self._done
        if scanning and not done:
            phase = "scanning"
        elif done:
            phase = "ready"
        else:
            phase = "idle"
        return {"phase": phase, "files": entries, "roots": roots}

    def _resolve_roots(self):
        try:
            raw = self._roots_provider() if callable(self._roots_provider) else self._roots_provider
        except Exception:
            return []
        roots = []
        seen = set()
        try:
            items = list(raw or [])
        except Exception:
            return []
        for item in items:
            try:
                real = os.path.realpath(item)
            except Exception:
                continue
            if not real:
                continue
            try:
                if not os.path.isdir(real):
                    continue
            except Exception:
                continue
            norm = os.path.normpath(real)
            key = os.path.normcase(norm)
            if key in seen:
                continue
            seen.add(key)
            roots.append(norm)
        roots.sort(key=len)
        out = []
        for path in roots:
            pkey = os.path.normcase(path)
            if any(pkey == os.path.normcase(k) or pkey.startswith(os.path.normcase(k) + os.sep) for k in out):
                continue
            out.append(path)
        return out

    def _scan(self):
        roots = self._resolve_roots()
        with self._lock:
            self._roots = list(roots)
            if not self._done:
                self._scanning = True
        if not roots:
            with self._lock:
                self._entries = ()
                self._done = True
                self._scanning = False
            return {"files": 0, "roots": []}
        cap = self._max_entries
        new = []
        last_pub = 0
        visited = set()
        seen_real = set()
        try:
            for root in roots:
                stack = [root]
                try:
                    rst = os.stat(root)
                    visited.add((rst.st_dev, rst.st_ino))
                    seen_real.add(os.path.normcase(os.path.realpath(root)))
                except OSError:
                    pass
                while stack:
                    if len(new) >= cap:
                        break
                    cur = stack.pop()
                    try:
                        with os.scandir(cur) as it:
                            entries = list(it)
                    except OSError:
                        continue
                    for entry in entries:
                        if len(new) >= cap:
                            break
                        try:
                            name = entry.name
                        except OSError:
                            continue
                        try:
                            is_link = entry.is_symlink()
                        except OSError:
                            continue
                        try:
                            is_dir = entry.is_dir(follow_symlinks=False)
                        except OSError:
                            continue
                        if is_dir:
                            if name.startswith("."):
                                continue
                            try:
                                if name.casefold() in _SKIP_NAMES:
                                    continue
                            except Exception:
                                continue
                            if is_link:
                                continue
                            full = entry.path
                            try:
                                rkey = os.path.normcase(os.path.realpath(full))
                            except Exception:
                                continue
                            if rkey in seen_real:
                                continue
                            try:
                                st = entry.stat(follow_symlinks=False)
                                # DirEntry.stat() on Windows reports st_ino == 0: only trust real inode numbers
                                dkey = (st.st_dev, st.st_ino) if st.st_ino else None
                            except OSError:
                                continue
                            if dkey is not None:
                                if dkey in visited:
                                    continue
                                visited.add(dkey)
                            seen_real.add(rkey)
                            stack.append(full)
                            continue
                        try:
                            is_file = entry.is_file(follow_symlinks=False)
                        except OSError:
                            continue
                        if not is_file:
                            if not is_link:
                                continue
                            try:
                                if not entry.is_file(follow_symlinks=True):
                                    continue
                            except OSError:
                                continue
                        if os.path.splitext(name)[1].lower() in _SKIP_EXT:
                            continue
                        try:
                            st = entry.stat(follow_symlinks=False)
                            size, mtime = int(st.st_size), float(st.st_mtime)
                        except OSError:
                            try:
                                st2 = os.stat(entry.path)
                                size, mtime = int(st2.st_size), float(st2.st_mtime)
                            except OSError:
                                size, mtime = 0, 0.0
                        p = entry.path
                        try:
                            folder = os.path.dirname(p)
                        except Exception:
                            folder = ""
                        try:
                            ncf, pcf = name.casefold(), p.casefold()
                        except Exception:
                            continue
                        new.append({"path": p, "name": name, "folder": folder,
                                    "size": size, "mtime": mtime, "root": root,
                                    "_ncf": ncf, "_pcf": pcf})
                        if len(new) - last_pub >= _PUBLISH_EVERY:
                            last_pub = len(new)
                            with self._lock:
                                self._entries = tuple(new)
                if len(new) >= cap:
                    break
        finally:
            with self._lock:
                self._entries = tuple(new)
                self._done = True
                self._scanning = False
        return {"files": len(new), "roots": list(roots)}

    def _candidates(self, snapshot, tokens):
        """Entries whose casefolded path contains `token` (one C-speed scan over a joined string)."""
        import bisect
        n = len(snapshot)
        sid = id(snapshot)
        first = snapshot[0] if n else None
        with self._blob_lock:
            cache = self._blob_cache
            if cache is None or len(cache) != 5 or cache[0] != sid or cache[1] != n or (n and cache[2] is not first):
                nl = chr(10)
                starts, parts, pos = [], [], 0
                for e in snapshot:
                    starts.append(pos)
                    parts.append(e["_pcf"])
                    pos += len(e["_pcf"]) + 1
                cache = (sid, n, first, nl.join(parts), starts)
                self._blob_cache = cache
            blob, starts = cache[3], cache[4]
        token = min(tokens, key=blob.count)  # the rarest token keeps the candidate list short
        out, at = [], blob.find(token)
        while at >= 0:
            idx = bisect.bisect_right(starts, at) - 1
            out.append(snapshot[idx])
            nxt = starts[idx + 1] if idx + 1 < n else -1
            if nxt < 0:
                break
            at = blob.find(token, nxt)
        return out

    def search(self, query, limit=50, only_roots=None):
        """Case-insensitive file-name search over the current snapshot.

        All whitespace-separated query tokens (casefolded) must occur in
        the file name or, weaker, in the full path. Ranking: exact name >
        name startswith > name contains > path contains, then shorter
        path, then most recent mtime. Works on partial results mid-scan.
        """
        try:
            q = str(query or "")
        except Exception:
            return []
        try:
            q, spec = _parse_query_filters(q)
            tokens = q.casefold().split()
        except Exception:
            return []
        if not tokens and spec is None:
            return []
        qfull = " ".join(tokens)
        try:
            lim = max(1, min(int(limit), 100))
        except Exception:
            lim = 50
        allowed = None
        if only_roots is not None:
            allowed = set()
            try:
                for r in only_roots:
                    try:
                        nr = _norm_real(r)
                        if nr:
                            allowed.add(nr)
                    except Exception:
                        continue
            except Exception:
                return []
            if not allowed:
                return []
        with self._lock:
            snapshot = self._entries  # immutable tuple: reference, no per-search copy
        scored = []
        for e in (self._candidates(snapshot, tokens) if tokens else snapshot):
            if allowed is not None:
                try:
                    if not _is_under_allowed(e.get("path", ""), allowed):
                        continue
                except Exception:
                    continue
            n, p = e["_ncf"], e["_pcf"]
            all_in_name = all(t in n for t in tokens)
            if all_in_name:
                if n == qfull:
                    rank = 0
                elif n.startswith(qfull):
                    rank = 1
                else:
                    rank = 2
            elif all(t in p for t in tokens):
                rank = 3
            else:
                continue
            try:
                key = (rank, len(e["path"]), -float(e["mtime"]))
            except Exception:
                key = (rank, len(str(e.get("path", ""))), 0.0)
            scored.append((key, e))
        scored.sort(key=lambda item: item[0])
        out = []
        if spec is None:
            for _, e in scored[:lim]:
                out.append({"path": e["path"], "name": e["name"], "folder": e["folder"],
                            "size": e["size"], "mtime": e["mtime"], "root": e["root"]})
        else:
            # Deferred spec/stat: filter in rank order, os.stat only for examined
            # rows until lim is filled (was stat-storm over all candidates).
            for _, e in scored:
                try:
                    if not _matches_query_filters(spec, e):
                        continue
                except Exception:
                    continue
                out.append({"path": e["path"], "name": e["name"], "folder": e["folder"],
                            "size": e["size"], "mtime": e["mtime"], "root": e["root"]})
                if len(out) >= lim:
                    break
        if out:
            # Exact matches always win; fuzzy is fallback only.
            return _SearchResult(out)
        # --- typo fallback: empty exact + single token with >=3 letters ---
        if len(tokens) == 1:
            tok = tokens[0]
            try:
                _letters = sum(1 for ch in tok if ch.isalpha())
            except Exception:
                _letters = 0
            if len(tok) >= 3 and _letters >= 3:
                pool = snapshot
                if allowed is not None:
                    _filt = []
                    for e in snapshot:
                        try:
                            if _is_under_allowed(e.get("path", ""), allowed):
                                _filt.append(e)
                        except Exception:
                            continue
                    pool = _filt
                if pool:
                    _hints = []
                    if len(pool) > _FUZZY_CAP:
                        _hints.append(_FUZZY_CAP_HINT)
                        try:
                            import heapq as _hq
                            pool = _hq.nlargest(_FUZZY_CAP, pool, key=lambda d: d.get("mtime", 0))
                        except Exception:
                            try:
                                pool = sorted(pool, key=lambda d: d.get("mtime", 0), reverse=True)[:_FUZZY_CAP]
                            except Exception:
                                pool = pool[:_FUZZY_CAP]
                    try:
                        _names = [e.get("name", "") if isinstance(e.get("name", ""), str) else str(e.get("name", "")) for e in pool]
                    except Exception:
                        _names = []
                    _pairs = []
                    try:
                        import localdocs_fuzzy as _fz
                        _pairs = _fz.fuzzy_search(_names, tok, max_edits=2, limit=lim) or []
                    except Exception:
                        _pairs = []
                    out2 = []
                    for _idx, _dist in (_pairs or []):
                        try:
                            if int(_dist) > 2:
                                continue
                            _e = pool[int(_idx)]
                        except Exception:
                            continue
                        try:
                            if spec is not None and not _matches_query_filters(spec, _e):
                                continue
                        except Exception:
                            continue
                        out2.append({"path": _e.get("path"), "name": _e.get("name"),
                                     "folder": _e.get("folder"), "size": _e.get("size"),
                                     "mtime": _e.get("mtime"), "root": _e.get("root"),
                                     "corrected_from": tok})
                        if len(out2) >= lim:
                            break
                    if out2:
                        return _SearchResult(out2, corrected_from=tok, hints=_hints)
                    if _hints:
                        return _SearchResult([], hints=_hints)
                    return _SearchResult([], hints=[_FUZZY_EMPTY_HINT])
                return _SearchResult([], hints=[_FUZZY_EMPTY_HINT])
        return _SearchResult(out)