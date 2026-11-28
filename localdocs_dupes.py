"""Search for duplicate files by size + blake2b hash.

Only reads; never deletes or moves anything.
"""

from __future__ import annotations

import hashlib
import os
import time

_PARTIAL_HEAD = 64 * 1024
_PARTIAL_TAIL = 64 * 1024
_STREAM_CHUNK = 1024 * 1024
_SKIP_DIRS = {"windows", "$recycle.bin", "node_modules", ".git"}
_PROGRESS_MIN_INTERVAL = 1.0


def _is_stop(should_stop) -> bool:
    if should_stop is None:
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


def _report(progress, scanned: int, groups: int, last: list) -> None:
    if progress is None:
        return
    now = time.monotonic()
    if now - last[0] < _PROGRESS_MIN_INTERVAL:
        return
    last[0] = now
    try:
        progress({"scanned": scanned, "groups": groups})
    except Exception:
        pass


def _partial_hash(path: str, size: int) -> str | None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(_PARTIAL_HEAD)
            if size > _PARTIAL_HEAD:
                try:
                    fh.seek(-_PARTIAL_TAIL, os.SEEK_END)
                except OSError:
                    fh.seek(0, os.SEEK_SET)
                    head = fh.read(_PARTIAL_HEAD)
                    return hashlib.blake2b(head).hexdigest()
                tail = fh.read(_PARTIAL_TAIL)
                return hashlib.blake2b(head + tail).hexdigest()
            return hashlib.blake2b(head).hexdigest()
    except OSError:
        return None


def _full_hash(path: str) -> str | None:
    try:
        digest = hashlib.blake2b()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_STREAM_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def find_duplicates(roots, min_size=1024, max_files=200000,
                    should_stop=None, progress=None) -> dict:
    """Find duplicate files under *roots*.

    Returns {"groups": [{"size": int, "hash": str,
                         "files": [{"path": str, "mtime": float}]}],
             "wasted_bytes": int, "scanned": int, "stopped": bool}.
    """
    if isinstance(roots, (str, bytes, os.PathLike)):
        roots = [roots]
    else:
        try:
            roots = list(roots)
        except TypeError:
            roots = [roots]

    scanned = 0
    stopped = False
    last_progress = [0.0]
    by_size: dict[int, list[tuple[str, float]]] = {}

    def add_candidate(path: str, size: int, mtime: float) -> bool:
        nonlocal scanned, stopped
        if size < min_size:
            return True
        if scanned >= max_files:
            stopped = True
            return False
        by_size.setdefault(size, []).append((path, mtime))
        scanned += 1
        _report(progress, scanned, 0, last_progress)
        return scanned < max_files

    stack: list[str] = []
    for root in roots:
        if _is_stop(should_stop):
            stopped = True
            break
        try:
            r = os.fspath(root)
        except TypeError:
            continue
        try:
            if os.path.islink(r):
                continue
        except OSError:
            continue
        try:
            if os.path.isfile(r):
                try:
                    st = os.stat(r)
                except OSError:
                    continue
                if _is_stop(should_stop):
                    stopped = True
                    break
                if not add_candidate(r, st.st_size, st.st_mtime):
                    stopped = True
                    break
            elif os.path.isdir(r):
                stack.append(r)
        except OSError:
            continue

    while stack and not stopped:
        if _is_stop(should_stop):
            stopped = True
            break
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                items = list(entries)
        except OSError:
            continue
        for entry in items:
            if _is_stop(should_stop):
                stopped = True
                break
            try:
                if entry.is_symlink():
                    continue
            except OSError:
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                try:
                    if entry.name.lower() in _SKIP_DIRS:
                        continue
                except OSError:
                    continue
                stack.append(entry.path)
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if not add_candidate(entry.path, st.st_size, st.st_mtime):
                stopped = True
                break
        if stopped and stack:
            break

    groups: list[dict] = []
    wasted_bytes = 0
    if not stopped:
        # Stage 2: partial hash (first 64KB + last 64KB) within same-size groups.
        partial: dict[tuple[int, str], list[tuple[str, float]]] = {}
        for size, candidates in by_size.items():
            if len(candidates) < 2:
                continue
            for path, mtime in candidates:
                if _is_stop(should_stop):
                    stopped = True
                    break
                digest = _partial_hash(path, size)
                _report(progress, scanned, 0, last_progress)
                if digest is None:
                    continue
                partial.setdefault((size, digest), []).append((path, mtime))
            if stopped:
                break
    if not stopped:
        # Stage 3: full blake2b streaming 1MB for partial matches.
        full: dict[tuple[int, str], list[tuple[str, float]]] = {}
        for (size, _part), candidates in partial.items():
            if len(candidates) < 2:
                continue
            for path, mtime in candidates:
                if _is_stop(should_stop):
                    stopped = True
                    break
                digest = _full_hash(path)
                _report(progress, scanned, 0, last_progress)
                if digest is None:
                    continue
                full.setdefault((size, digest), []).append((path, mtime))
            if stopped:
                break
        if not stopped:
            for (size, digest), members in full.items():
                if len(members) < 2:
                    continue
                members_sorted = sorted(members, key=lambda m: (m[1], m[0]))
                groups.append({
                    "size": size,
                    "hash": digest,
                    "files": [{"path": p, "mtime": t} for p, t in members_sorted],
                })
            groups.sort(key=lambda g: (g["size"] * (len(g["files"]) - 1),
                                       g["size"], g["hash"]), reverse=True)
            wasted_bytes = sum(g["size"] * (len(g["files"]) - 1) for g in groups)

    if stopped:
        groups.sort(key=lambda g: (g["size"] * (len(g["files"]) - 1),
                                   g["size"], g["hash"]), reverse=True)
        wasted_bytes = sum(g["size"] * (len(g["files"]) - 1) for g in groups)

    return {"groups": groups, "wasted_bytes": wasted_bytes,
            "scanned": scanned, "stopped": stopped}


def scan_fattest(roots, limit=50, stale_days=365, should_stop=None,
                 progress=None) -> dict:
    """Top-N biggest files untouched for >stale_days (stat only, no reads).

    Walk only (size/mtime via stat); never opens file contents, never
    deletes/moves. Stale means now - mtime > stale_days * 86400.
    Returns {"fattest": [{"path": str, "size": int, "mtime": float}],
             "scanned": int, "stopped": bool}.
    """
    if isinstance(roots, (str, bytes, os.PathLike)):
        roots = [roots]
    else:
        try:
            roots = list(roots)
        except TypeError:
            roots = [roots]
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 50))
    try:
        stale_days = float(stale_days)
    except (TypeError, ValueError):
        stale_days = 365.0
    cutoff = time.time() - max(0.0, stale_days) * 86400.0

    scanned = 0
    stopped = False
    last_progress = [0.0]
    stale: list[tuple[str, int, float]] = []

    stack: list[str] = []
    for root in roots:
        if _is_stop(should_stop):
            stopped = True
            break
        try:
            r = os.fspath(root)
        except TypeError:
            continue
        try:
            if os.path.islink(r):
                continue
        except OSError:
            continue
        try:
            if os.path.isfile(r):
                try:
                    st = os.stat(r)
                except OSError:
                    continue
                if _is_stop(should_stop):
                    stopped = True
                    break
                scanned += 1
                _report(progress, scanned, 0, last_progress)
                if st.st_mtime < cutoff:
                    stale.append((r, st.st_size, st.st_mtime))
            elif os.path.isdir(r):
                stack.append(r)
        except OSError:
            continue

    while stack and not stopped:
        if _is_stop(should_stop):
            stopped = True
            break
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                items = list(entries)
        except OSError:
            continue
        for entry in items:
            if _is_stop(should_stop):
                stopped = True
                break
            try:
                if entry.is_symlink():
                    continue
            except OSError:
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                try:
                    if entry.name.lower() in _SKIP_DIRS:
                        continue
                except OSError:
                    continue
                stack.append(entry.path)
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            scanned += 1
            _report(progress, scanned, 0, last_progress)
            if st.st_mtime < cutoff:
                stale.append((entry.path, st.st_size, st.st_mtime))

    stale.sort(key=lambda t: (t[1], t[0]), reverse=True)
    fattest = [{"path": p, "size": s, "mtime": m}
               for p, s, m in stale[:limit]]
    return {"fattest": fattest, "scanned": scanned, "stopped": stopped}


def keep_newest_plan(group) -> dict:
    """Cleanup plan for one duplicate group: keep newest, recycle the rest.

    Newest = max by (mtime, path) for determinism. Pure (no I/O).
    Returns {"keep": str | None, "recycle": [str, ...]}; recycle sorted
    by (mtime, path) ascending. Caller must send each recycle path via
    fsapi recycle (never permanent) and surface OK/PARTIAL + Undo.
    """
    try:
        files = (group or {}).get("files") or []
    except AttributeError:
        files = []
    norm: list[tuple[str, float]] = []
    for f in files:
        try:
            p = f.get("path")
            m = float(f.get("mtime") or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue
        if isinstance(p, str) and p:
            norm.append((p, m))
    if not norm:
        return {"keep": None, "recycle": []}
    norm.sort(key=lambda t: (t[1], t[0]))
    keep = norm[-1][0]
    recycle = [p for p, _ in norm[:-1]]
    return {"keep": keep, "recycle": recycle}
