"""Fast non-admin directory walker built on NtQueryDirectoryFile (one syscall per directory, parsed in bulk).

~3x faster than os.scandir in Python and returns size / last-write time for free (no per-file stat).
Output records start with (id, name, parent_id, flags, root_pos); file records
append size and Unix mtime for NameIndex._assemble. Directory metadata is unknown.
"""
from __future__ import annotations

import ctypes
import itertools
import os
import queue
import struct
import threading
from ctypes import wintypes as W

BUF = 1 << 18
_HDR = struct.Struct("<IIQQQQqqII").unpack_from  # next, index, 4 times, eof, alloc, attrs, name length
_AVAILABLE = os.name == "nt"

if _AVAILABLE:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _nt = ctypes.WinDLL("ntdll")
    _k32.CreateFileW.restype = W.HANDLE
    _k32.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, W.LPVOID, W.DWORD, W.DWORD, W.HANDLE]
    _k32.CloseHandle.argtypes = [W.HANDLE]

    class _IOSB(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_long), ("Info", ctypes.c_size_t)]

    _nt.NtQueryDirectoryFile.argtypes = [W.HANDLE, W.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_IOSB),
                                         ctypes.c_void_p, W.ULONG, W.ULONG, ctypes.c_ubyte, ctypes.c_void_p, ctypes.c_ubyte]
    _nt.NtQueryDirectoryFile.restype = ctypes.c_long
    _INVALID = W.HANDLE(-1).value


_BS = chr(92)
_PREFIX = _BS * 2 + "?" + _BS
_UNC_PREFIX = _PREFIX + "UNC" + _BS


def to_nt_path(path: str) -> str:
    """Map a user path to an extended-length NT path for CreateFileW.

    Drive paths ``C:\\dir`` -> ``\\\\?\\C:\\dir``; UNC paths
    ``\\\\server\\share\\dir`` -> ``\\\\?\\UNC\\server\\share\\dir`` (never
    ``'\\\\?\\' + '\\\\server'``); already-prefixed paths pass through.
    Pure (no I/O), so it is unit-testable without a network.
    """
    if path.startswith(_PREFIX):
        return path
    path = path.replace("/", _BS)  # the extended-length prefix disables slash normalisation
    if path.startswith(_PREFIX):
        return path
    if path.startswith(_BS * 2):
        return _UNC_PREFIX + path[2:]
    if path.startswith("//"):
        return _UNC_PREFIX + path[2:]
    return _PREFIX + path


def available() -> bool:
    return _AVAILABLE


def _classify_error(exc: BaseException) -> str:
    """Map a directory-open failure to a per-root status string."""
    winerr = getattr(exc, "winerror", None)
    err = getattr(exc, "errno", None)
    if isinstance(exc, PermissionError) or winerr == 5 or err in (5, 13):  # raw Win32 5 arrives as errno 5 (OSError(err, ...))
        return "denied"
    return "offline"


def list_dir(path: str, buf=None):
    """Return ([(name, size, filetime)], [(name, attrs)]) for one directory; reparse points are not followed.

    Raises OSError when the directory cannot be opened (so walk() can report
    per-root ok/denied/offline instead of silently returning empty).
    On non-Windows platforms uses os.scandir as a portable fallback.
    """
    if not _AVAILABLE:
        files, dirs = [], []
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_symlink():
                        continue  # never follow reparse points / junctions
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append((entry.name, 0x10))
                    elif entry.is_file(follow_symlinks=False):
                        try:
                            st = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        files.append((entry.name, st.st_size,
                                      (st.st_mtime + 11644473600.0) * 10000000.0))
                except OSError:
                    continue
        return files, dirs
    files, dirs = [], []
    nt_path = to_nt_path(path)
    h = _k32.CreateFileW(nt_path, 1, 7, None, 3, 0x02000000, None)
    if h in (None, _INVALID):
        err = ctypes.get_last_error()
        raise OSError(err, "CreateFileW failed for %r" % (path,), path)
    iosb = _IOSB()
    addr = ctypes.addressof(buf)
    restart = 1
    try:
        while True:
            status = _nt.NtQueryDirectoryFile(h, None, None, None, ctypes.byref(iosb), addr, BUF, 1, 0, None, restart)
            restart = 0
            if status != 0 or not iosb.Info:
                break
            mv = memoryview(buf)
            off = 0
            while True:
                nxt, _i, _c, _a, wt, _ch, eof, _al, attrs, nlen = _HDR(mv, off)
                name = bytes(mv[off + 64: off + 64 + nlen]).decode("utf-16le", "replace")
                if name != "." and name != "..":
                    if attrs & 0x10:
                        if not attrs & 0x400:
                            dirs.append((name, attrs))
                    else:
                        files.append((name, eof, wt))
                if nxt == 0:
                    break
                off += nxt
    finally:
        _k32.CloseHandle(h)
    return files, dirs


class _Records(list):
    """List subclass carrying the per-root status dict as ``.status``.

    Returning a list subclass keeps ``isinstance(recs, list)`` and all
    sequence behaviour identical for existing callers.
    """


def walk(roots, workers, cap, skip_names, skip_ext, penalty_ext, flag_dir, flag_hidden, flag_junk, on_error=None,
         status=None):
    """Parallel (work-stealing) walk. roots: list of absolute paths. Returns the record list.

    Per-root health is exposed without changing the return value for existing
    callers: the returned list has a ``.status`` attribute (dict mapping each
    root to ``"ok"`` / ``"denied"`` / ``"offline"``), and when the optional
    ``status`` argument is a dict it is filled with the same mapping.
    The ``cap`` entry limit is enforced per entry (final length never exceeds
    ``cap``), not only per directory.
    """
    roots = list(roots)
    try:
        cap_n = None if cap is None else int(cap)
    except (TypeError, ValueError):
        cap_n = None
    statuses: dict = {}
    status_lock = threading.Lock()

    def _set_status(root, value):
        with status_lock:
            statuses[root] = value

    tasks = queue.Queue()
    ids = itertools.count()
    lock = threading.Lock()
    pending = [0]
    done = threading.Event()
    claimed = [0]
    shared: _Records = _Records()

    def _claim():
        """Atomically claim one entry slot; return new id or None when capped."""
        with lock:
            if cap_n is not None and claimed[0] >= cap_n:
                return None
            rid = next(ids)
            claimed[0] += 1
            return rid

    for pos, root in enumerate(roots):
        if cap_n is not None and len(shared) >= cap_n:
            _set_status(root, "offline")
            continue
        rid = _claim()
        if rid is None:
            _set_status(root, "offline")
            continue
        shared.append((rid, os.path.basename(root) or root, -1, flag_dir, pos))
        tasks.put((root, rid, pos, True))
        pending[0] += 1
    if not roots:
        shared.status = dict(statuses)
        if isinstance(status, dict):
            status.update(statuses)
        return shared
    if pending[0] <= 0:
        done.set()
        shared.status = dict(statuses)
        if isinstance(status, dict):
            status.update(statuses)
        return shared

    def worker():
        buf = None if not _AVAILABLE else ctypes.create_string_buffer(BUF)
        local = []
        while True:
            try:
                cur, pid, pos, is_root = tasks.get(timeout=0.1)
            except queue.Empty:
                if done.is_set():
                    break
                continue
            new_dirs = 0
            try:
                files, dirs = list_dir(cur, buf)
                if is_root:
                    _set_status(cur, "ok")
                for name, attrs in dirs:
                    low = name.casefold()
                    if name.startswith(".") or low in skip_names:
                        continue
                    rid = _claim()
                    if rid is None:
                        capped = True
                        break
                    local.append((rid, name, pid, flag_dir, pos))
                    tasks.put((cur.rstrip(chr(92)) + chr(92) + name, rid, pos, False))
                    new_dirs += 1
                else:
                    capped = False
                if not capped:
                    for name, size, filetime in files:
                        ext = os.path.splitext(name)[1].lower()
                        if ext in skip_ext:
                            continue
                        fl = 0
                        if name.startswith("."):
                            fl |= flag_hidden
                        if ext in penalty_ext:
                            fl |= flag_junk
                        rid = _claim()
                        if rid is None:
                            break
                        local.append((rid, name, pid, fl, pos, int(size),
                                      int(filetime) / 10000000.0 - 11644473600.0))
            except Exception as exc:  # one broken directory must never stop the walk
                if is_root:
                    _set_status(cur, _classify_error(exc))
                if callable(on_error):
                    try:
                        on_error(exc)
                    except Exception:
                        pass
            finally:
                with lock:
                    pending[0] += new_dirs - 1
                    if pending[0] <= 0:
                        done.set()
                if len(local) >= 20000:
                    with lock:
                        shared.extend(local)
                    local = []
        if local:
            with lock:
                shared.extend(local)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, int(workers)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if cap_n is not None and len(shared) > cap_n:
        shared.sort(key=lambda r: r[0])
        del shared[cap_n:]
    shared.status = dict(statuses)
    for root in roots:
        if root not in shared.status:
            shared.status[root] = "ok"
    if isinstance(status, dict):
        status.update(shared.status)
    return shared
