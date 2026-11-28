"""LocalDocs Windows file-operation engine (P5 sandbox).

Sandbox scope: C:/Users/ninja/Desktop/LocalDocs/work/product_recovery_20261022/P5_fs/dev
No server/UI integration. Main project is read-only from here.

Public surface:
    FsEngine(allowed_roots, journal_path=None, owner_check=None,
             shell_backend=None, max_journal_ops=200)
        .execute(action, items, options=None) -> dict
        .undo(op_id) -> dict

Actions: "rename", "create_folder", "copy", "move" ("cut_paste" is an alias
of "move"), "delete_recycle" (alias "recycle"). "undo" is a method, not an
execute() action.

Collision policy (options["collision"]): "fail" (default), "skip",
"rename_new". There is deliberately NO "overwrite" policy: the engine never
overwrites an existing destination.

Recycle bin (forced recycle-only, never permanent delete):
Real deletions go through localdocs_shellops.recycle_to_bin, which uses
IFileOperation with FOF_ALLOWUNDO | FOFX_RECYCLEONDELETE (Win8+, MS Learn
IFileOperation::SetOperationFlags: "When a file is deleted, send it to the
Recycle Bin rather than permanently deleting it") plus FOF_WANTNUKEWARNING,
FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT | FOFX_EARLYFAILURE, on a
COM STA apartment (CoInitializeEx COINIT_APARTMENTTHREADED). Volumes are
pre-gated to DRIVE_FIXED with a confirmable bin; anything else is
RECYCLE_UNSUPPORTED and the source is preserved. SHFileOperationW
FOF_ALLOWUNDO alone is NOT claimed as guaranteed recycle (MS Learn:
"Preserve undo information, if possible" and SHFileOperation is replaced
by IFileOperation as of Vista). There is NO os.remove/rmtree fallback.
Return code, GetAnyOperationsAborted / 4-byte fAnyOperationsAborted, and
post-call existence are all checked.

ctypes ABI (x64, per MS Learn SHFILEOPSTRUCTW / Win32 docs):
SHFILEOPSTRUCTW = {HWND(8)@0, UINT(4)@8, PCZZWSTR(8)@16, PCZZWSTR(8)@24,
FILEOP_FLAGS WORD(2)@32 + 2 pad, BOOL 4-byte@36, LPVOID(8)@40, PCWSTR(8)@48},
sizeof 56. BOOL is c_int (4 bytes), never c_bool (1 byte). GetFileAttributesW
restype is DWORD (c_uint32) so 0xFFFFFFFF compares equal to INVALID.
MoveFileExW restype is BOOL (c_int). HANDLEs are 64-bit c_void_p with
INVALID_HANDLE_VALUE = c_void_p(-1). CreateFileW/GetFileInformationByHandle/
CloseHandle/GetDriveTypeW/SHQueryRecycleBinW/CoInitializeEx/CoCreateInstance/
SHCreateItemFromParsingName all declare argtypes/restype.

Security model (enforced in this service boundary, never in UI):
local absolute drive-letter paths only; lexical rejection of UNC, device
namespaces, ADS streams, reserved DOS names, trailing dots/spaces;
canonical allowlist containment; allowed-root and system-path mutation
refusal; reparse-point/symlink rejection along every ancestor and inside
input directory trees. Owner authorization is a caller-supplied predicate
(owner_check); it is NOT an NTFS ACL/SID ownership check. Default with no
explicit owner_check is DENY. Ownership is re-checked immediately before
each mutation and again before each undo. Validation happens immediately
before each action. Atomic OS primitives (MoveFileExW without REPLACE,
O_EXCL creates, os.mkdir) are used so a destination appearing after
validation causes a clean ITEM_EXISTS failure instead of an overwrite.

Identity: stable Windows file identity (volume serial + file index via
GetFileInformationByHandle, fallback st_dev/st_ino) plus content
verification (SHA-256 for files; name/size/hash manifest for trees).
Same-size/mtime replacement or edited data never matches, so undo refuses
with MODIFIED and preserves user data.

Cross-volume move: staged copy + strong verification (size + SHA-256 per
file, structure for trees) + journaled staged outcome BEFORE any source
cleanup, then collecting per-file source cleanup. Partial cleanup failure
keeps the staged journal (resume/undo metadata), reports a truthful partial
outcome, and never silently loses data. No silent overwrite (O_EXCL/mkdir
only). Disabling cross-volume is not supported; it stays functional.

Journal: cross-process serialized (global RLock + msvcrt lockfile),
atomic tmp+fsync+os.replace, bounded, merged (never silent
last-writer-wins). Write failure after a successful FS action yields a
truthful PARTIAL outcome with journal_error + journal_pending recovery
evidence, never an unhandled exception or stale SUCCESS. Corrupt/malformed
JSON or schema fails safe to empty (bad records skipped), never raises.

Known unresolved threat (not claimed as solved, not accepted by the user):
cross-process TOCTOU. An external process with sufficient rights can still
swap a directory for a junction between our ancestor check and the
filesystem call, and can rename a file over a destination on
volumes/processes outside our lock. In-process conflicting actions are
serialized with an RLock; ancestors are re-validated immediately before
acting and atomic no-overwrite primitives are used, which narrows but does
not close the window.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import threading
import time
import uuid

try:
    import msvcrt as _msvcrt
except Exception:  # pragma: no cover
    _msvcrt = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_OK = "OK"
STATUS_SKIPPED = "SKIPPED"
STATUS_FAILED = "FAILED"

OP_SUCCESS = "SUCCESS"
OP_PARTIAL = "PARTIAL_SUCCESS"
OP_FAILED = "FAILED"
OP_UNSUPPORTED = "UNSUPPORTED"

COLLISION_FAIL = "fail"
COLLISION_SKIP = "skip"
COLLISION_RENAME_NEW = "rename_new"
COLLISIONS = (COLLISION_FAIL, COLLISION_SKIP, COLLISION_RENAME_NEW)

RESERVED_BASENAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + ["COM%d" % i for i in range(1, 10)]
    + ["LPT%d" % i for i in range(1, 10)]
)

_BAD_CHARS = set('<>"|?*')
_ACTION_ALIASES = {
    "cut_paste": "move",
    "cut": "move",
    "paste": "move",
    "recycle": "delete_recycle",
    "delete": "delete_recycle",
    "delete_recycle": "delete_recycle",
    "mkdir": "create_folder",
}

_UNDOABLE = ("rename", "move", "copy", "create_folder")

_WINERROR_MAP = {
    2: "NOT_FOUND",
    3: "NOT_FOUND",
    5: "ACCESS_DENIED",
    32: "LOCKED",
    33: "LOCKED",
    80: "ITEM_EXISTS",
    112: "DISK_FULL",
    183: "ITEM_EXISTS",
}

_FO_DELETE = 0x0003
_FOF_ALLOWUNDO = 0x0040
_FOF_NOCONFIRMATION = 0x0010
_FOF_NOERRORUI = 0x0400
_FOF_SILENT = 0x0004
_FOF_WANTNUKEWARNING = 0x4000
_RECYCLE_FLAGS = (
    _FOF_ALLOWUNDO
    | _FOF_NOCONFIRMATION
    | _FOF_NOERRORUI
    | _FOF_SILENT
    | _FOF_WANTNUKEWARNING
)

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_ERROR_NOT_SAME_DEVICE = 17

_JOURNAL_IO_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class FsError(Exception):
    """Service-boundary error with a stable machine-readable code."""

    def __init__(self, code, message, item=None):
        super().__init__(message)
        self.code = code
        self.item = item


def _win_code(oerr):
    """Map an OSError/WindowsError to a stable engine error code."""
    if isinstance(oerr, FileNotFoundError):
        return "NOT_FOUND"
    if isinstance(oerr, FileExistsError):
        return "ITEM_EXISTS"
    if isinstance(oerr, PermissionError):
        return "ACCESS_DENIED"
    winerror = getattr(oerr, "winerror", None)
    if winerror in _WINERROR_MAP:
        return _WINERROR_MAP[winerror]
    if "being used by another process" in str(oerr):
        return "LOCKED"
    return "OS_ERROR"


# ---------------------------------------------------------------------------
# Low-level Win32 helpers (builtin ctypes only; no third-party dependency)
# Correct x64 ABI throughout: BOOL=c_int(4B), DWORD=c_uint32,
# HANDLE/HWND/LPVOID=c_void_p(8B), HRESULT=c_long.
# ---------------------------------------------------------------------------

def _kernel32():
    return ctypes.windll.kernel32


def _get_long_single(p):
    """Single GetLongPathNameW query; expanded string or None on failure."""
    try:
        k32 = _kernel32()
        fn = k32.GetLongPathNameW
        fn.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        fn.restype = ctypes.c_uint32  # DWORD length, never signed
        buf_len = 512
        while True:
            buf = ctypes.create_unicode_buffer(buf_len)
            n = int(fn(p, buf, buf_len))
            if n == 0:
                return None  # failure -> pass-through by caller
            if n >= buf_len:
                if buf_len > 32768:
                    return None
                buf_len = n + 2
                continue
            if not buf.value:
                return None
            return buf.value
    except Exception:
        return None


def _expand_83(path):
    """Expand 8.3 short-name segments via GetLongPathNameW.

    Strips a ``\\\\?\\`` prefix for the query and restores it afterwards.
    Falls back to longest-existing-ancestor expansion so not-yet-existing
    tails (future dst) still get their existing prefix canonicalized before
    containment checks. Pass-through on any failure.
    """
    if not isinstance(path, str) or not path:
        return path
    prefix = ""
    p = path
    if p.startswith("\\\\?\\"):
        prefix = "\\\\?\\"
        p = p[4:]
        if not p:
            return path
    try:
        single = _get_long_single(p)
        if single is not None:
            return (prefix + single) if prefix else single
        cur = p
        tails = []
        while True:
            parent = os.path.dirname(cur)
            if parent == cur or not parent:
                return path
            tails.insert(0, os.path.basename(cur))
            cur = parent
            if len(tails) > 256:
                return path
            if os.path.lexists(cur):
                exp = _get_long_single(cur)
                if exp is not None and exp != cur:
                    joined = os.path.join(exp, *tails)
                    return (prefix + joined) if prefix else joined
                return path
    except Exception:
        return path
    return path


def _long_path(path):
    """Expand 8.3 via GetLongPathNameW, then add extended-length prefix."""
    try:
        exp = _expand_83(path)
        if isinstance(exp, str):
            path = exp
    except Exception:
        pass
    if path.startswith("\\\\?\\") or len(path) < 240:
        return path
    return "\\\\?\\" + path


def _get_file_attributes(path):
    fn = _kernel32().GetFileAttributesW
    fn.argtypes = [ctypes.c_wchar_p]
    fn.restype = ctypes.c_uint32  # DWORD: INVALID (0xFFFFFFFF) stays unsigned
    return int(fn(_long_path(path)))


def _is_reparse(path):
    """True if an existing path is a reparse point (symlink/junction)."""
    try:
        attrs = _get_file_attributes(path)
    except Exception:
        return os.path.islink(path)
    if attrs == _INVALID_FILE_ATTRIBUTES:
        return False  # failed query is not evidence of a reparse point
    return bool(attrs & _FILE_ATTRIBUTE_REPARSE_POINT)


def _get_last_error():
    fn = _kernel32().GetLastError
    fn.argtypes = []
    fn.restype = ctypes.c_uint32
    return int(fn())


def _move_file_noreplace(src, dst):
    """Atomic-ish move that FAILS if dst exists (no REPLACE flag).

    Returns (ok, winerror). Same-volume renames are atomic. Cross-volume
    moves return ok=False with ERROR_NOT_SAME_DEVICE so the caller can use
    the explicit staged copy+verify+delete path instead of a blind fallback.
    """
    fn = _kernel32().MoveFileExW
    fn.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    fn.restype = ctypes.c_int  # BOOL is 4 bytes, not c_bool (1 byte)
    ok = int(fn(_long_path(src), _long_path(dst), 0))
    if ok:
        return True, 0
    return False, _get_last_error()


def _copy_stream_noreplace(src, dst):
    """Copy file bytes with O_EXCL destination (atomic no-overwrite).

    Raises FileExistsError if dst appeared after validation. Caller removes
    a partial dst on failure (engine-owned partial only).
    """
    chunk = 1024 * 1024
    with open(_long_path(src), "rb") as fin:
        fout = os.open(_long_path(dst),
                       os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY,
                       0o666)
        try:
            with os.fdopen(fout, "wb") as out:
                while True:
                    data = fin.read(chunk)
                    if not data:
                        break
                    out.write(data)
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            try:
                os.remove(_long_path(dst))
            except OSError:
                pass
            raise
    shutil.copystat(_long_path(src), _long_path(dst), follow_symlinks=False)


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", ctypes.c_uint32),
        ("ftCreationTime_dwLow", ctypes.c_uint32),
        ("ftCreationTime_dwHigh", ctypes.c_uint32),
        ("ftLastAccessTime_dwLow", ctypes.c_uint32),
        ("ftLastAccessTime_dwHigh", ctypes.c_uint32),
        ("ftLastWriteTime_dwLow", ctypes.c_uint32),
        ("ftLastWriteTime_dwHigh", ctypes.c_uint32),
        ("dwVolumeSerialNumber", ctypes.c_uint32),
        ("nFileSizeHigh", ctypes.c_uint32),
        ("nFileSizeLow", ctypes.c_uint32),
        ("nNumberOfLinks", ctypes.c_uint32),
        ("nFileIndexHigh", ctypes.c_uint32),
        ("nFileIndexLow", ctypes.c_uint32),
    ]


def _get_volume_file_id(path):
    """Stable Windows identity: (volume_serial, file_index) or None.

    Uses CreateFileW (FILE_READ_ATTRIBUTES, share-all, OPEN_EXISTING) +
    GetFileInformationByHandle. HANDLE is 64-bit c_void_p; restype is
    c_void_p so INVALID_HANDLE_VALUE (-1) is not truncated to 32 bits.
    """
    try:
        k32 = _kernel32()
        cfw = k32.CreateFileW
        cfw.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_void_p]
        cfw.restype = ctypes.c_void_p
        GENERIC_READ = 0x80000000
        FILE_READ_ATTRIBUTES = 0x0080
        SHARE_ALL = 0x7
        OPEN_EXISTING = 3
        FLAG_BACKUP_SEMANTICS = 0x02000000
        h = cfw(_long_path(path), FILE_READ_ATTRIBUTES | GENERIC_READ,
                SHARE_ALL, None, OPEN_EXISTING, FLAG_BACKUP_SEMANTICS, None)
        if h is None or int(ctypes.c_void_p(h).value or 0) in (
                0, 0xFFFFFFFFFFFFFFFF) or h == ctypes.c_void_p(-1).value:
            # Compare as address-sized ints to catch -1 on 64-bit.
            try:
                k32.CloseHandle.argtypes = [ctypes.c_void_p]
                k32.CloseHandle.restype = ctypes.c_int
            except Exception:
                pass
            return None
        try:
            gih = k32.GetFileInformationByHandle
            gih.argtypes = [ctypes.c_void_p,
                            ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION)]
            gih.restype = ctypes.c_int
            info = _BY_HANDLE_FILE_INFORMATION()
            if not int(gih(h, ctypes.byref(info))):
                return None
            idx = (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow)
            return (int(info.dwVolumeSerialNumber), int(idx))
        finally:
            try:
                k32.CloseHandle.argtypes = [ctypes.c_void_p]
                k32.CloseHandle.restype = ctypes.c_int
                k32.CloseHandle(h)
            except Exception:
                pass
    except Exception:
        return None


def _hash_file(path):
    h = hashlib.sha256()
    with open(_long_path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_manifest(root):
    """Map relpath -> ('dir' | {'size': int, 'sha256': str})."""
    if os.path.islink(root) or _is_reparse(root):
        raise FsError("REPARSE_POINT",
                      "tree contains a reparse point at %r" % root)
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                followlinks=False):
        dirnames.sort()
        filenames.sort()
        for d in dirnames:
            p = os.path.join(dirpath, d)
            if os.path.islink(p) or _is_reparse(p):
                raise FsError("REPARSE_POINT",
                              "tree contains a reparse point at %r" % p)
            rel = os.path.relpath(p, root)
            out[rel] = "dir"
        for f in filenames:
            p = os.path.join(dirpath, f)
            if os.path.islink(p) or _is_reparse(p):
                raise FsError("REPARSE_POINT",
                              "tree contains a reparse point at %r" % p)
            rel = os.path.relpath(p, root)
            out[rel] = {"size": os.path.getsize(p), "sha256": _hash_file(p)}
    return out


# ---------------------------------------------------------------------------
# Shell recycle backends
# ---------------------------------------------------------------------------

class _SHFILEOPSTRUCTW(ctypes.Structure):
    """Fixed x64 ABI per MS Learn SHFILEOPSTRUCTW.

    fFlags WORD@32 + 2 pad, fAnyOperationsAborted Win32 BOOL (c_int, 4B)@36,
    hNameMappings@40, lpszProgressTitle@48, sizeof 56. c_bool would place
    the abort flag at offset 34 and it would never be read.
    """

    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("wFunc", ctypes.c_uint),
        ("pFrom", ctypes.c_wchar_p),
        ("pTo", ctypes.c_wchar_p),
        ("fFlags", ctypes.c_ushort),
        ("fAnyOperationsAborted", ctypes.c_int),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", ctypes.c_wchar_p),
    ]


class ShellRecycleBackend:
    """Real Windows Shell recycle via forced recycle-only semantics.

    Primary path delegates to localdocs_shellops.recycle_to_bin:
    IFileOperation + FOF_ALLOWUNDO | FOFX_RECYCLEONDELETE (+WANTNUKEWARNING,
    silent/no-UI/early-fail) on a COM STA apartment, with the volume
    pre-gated to DRIVE_FIXED + confirmable bin. Unsupported/COM/HRESULT/
    abort => failure, source preserved. All UI suppressed; no native
    dialogs. There is NO os.remove/rmtree fallback.
    """

    def recycle(self, path):
        try:
            from localdocs_shellops import recycle_to_bin
        except Exception as exc:
            return {"code": 1, "aborted": False,
                    "error": "shellops unavailable: %r" % exc}
        try:
            res = recycle_to_bin(path)
        except Exception as exc:
            return {"code": 1, "aborted": False,
                    "error": "recycle backend error: %r" % exc}
        return {"code": int((res or {}).get("code", 1)),
                "aborted": bool((res or {}).get("aborted", False)),
                "error": (res or {}).get("error")}


class FakeShellBackend:
    """Test-only shell double: simulates recycle without touching the bin.

    mode="ok": moves the path into a scratch dir (simulated bin).
    mode="fail": returns a nonzero code (recycle unavailable/failed).
    mode="aborted": reports an aborted operation.
    """

    def __init__(self, trash_dir, mode="ok", code=0):
        self.trash_dir = trash_dir
        self.mode = mode
        self.code = code
        self.calls = []

    def recycle(self, path):
        self.calls.append(path)
        if self.mode == "aborted":
            return {"code": 0, "aborted": True}
        if self.mode == "fail":
            return {"code": self.code or 1, "aborted": False}
        os.makedirs(self.trash_dir, exist_ok=True)
        base = os.path.basename(path.rstrip("\\/")) or "_root"
        dest = os.path.join(self.trash_dir, base)
        i = 0
        while os.path.lexists(dest):
            i += 1
            dest = os.path.join(self.trash_dir, "%s (%d)" % (base, i))
        os.rename(path, dest)
        return {"code": 0, "aborted": False}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FsEngine:
    """Windows file-operation engine with journaled undo."""

    def __init__(self, allowed_roots, journal_path=None, owner_check=None,
                 shell_backend=None, max_journal_ops=200):
        if not allowed_roots:
            raise ValueError("allowed_roots must not be empty")
        self._roots = [self._canon_root(r) for r in allowed_roots]
        self._journal_path = journal_path
        # Default DENY: without an explicit owner_check no mutation is
        # authorized. This is a caller-supplied authorization predicate,
        # not an NTFS ACL/SID ownership check.
        self._owner_check = owner_check
        self._shell = shell_backend or ShellRecycleBackend()
        self._max_ops = max(1, int(max_journal_ops))
        self._lock = threading.RLock()
        self._ops = {}  # op_id -> journal record
        if journal_path and os.path.exists(journal_path):
            self._load_journal()

    # -- path policy ----------------------------------------------------

    @staticmethod
    def _canon_root(root):
        if not isinstance(root, str):
            raise ValueError("allowed root must be a string")
        s = root.replace("/", "\\")
        if not _drive_abs(s):
            raise ValueError("allowed root must be absolute drive path: %r" % root)
        canon = os.path.normpath(s)
        try:
            canon = os.path.normpath(_expand_83(canon))
        except Exception:
            pass
        norm = os.path.normcase(canon)
        return norm

    def _resolve(self, raw, what="path"):
        """Validate and canonicalize a user-supplied path.

        Returns the canonical path (normpath, original case preserved).
        Raises FsError on any policy violation.
        """
        if not isinstance(raw, str) or not raw.strip():
            raise FsError("INVALID_PATH",
                          "%s must be a non-empty string" % what)
        s = raw.replace("/", "\\")
        if s.startswith("\\\\"):
            raise FsError("BAD_NAMESPACE",
                          "%s uses a forbidden UNC/device namespace: %r" % (what, raw))
        if not _drive_abs(s):
            raise FsError("RELATIVE_PATH",
                          "%s must be a local absolute Windows path: %r" % (what, raw))
        if ":" in s[2:]:
            raise FsError("ADS_BLOCKED",
                          "%s contains a forbidden stream/colon syntax: %r" % (what, raw))
        drive, rest = s[0:2], s[2:]
        if rest.startswith("\\"):
            rest = rest[1:]
        parts = rest.split("\\") if rest else []
        for comp in parts:
            if comp in ("", ".", ".."):
                raise FsError("BAD_NAMESPACE",
                              "%s contains a forbidden component %r: %r"
                              % (what, comp, raw))
            if any(c in _BAD_CHARS or ord(c) < 32 for c in comp):
                raise FsError("BAD_CHARS",
                              "%s contains forbidden characters: %r" % (what, raw))
            if comp[-1] in (" ", "."):
                raise FsError("TRAILING_DOT_SPACE",
                              "%s component ends with dot/space: %r" % (what, raw))
            stem = comp.split(".", 1)[0].upper()
            if stem in RESERVED_BASENAMES:
                raise FsError("RESERVED_NAME",
                              "%s uses a reserved DOS device name: %r" % (what, raw))
        canon = os.path.normpath(s)
        try:
            canon = os.path.normpath(_expand_83(canon))
        except Exception:
            pass
        key = os.path.normcase(canon)
        root = None
        for r in self._roots:
            base = r.rstrip(os.sep)
            if key == r or key == base or key.startswith(base + os.sep):
                root = r
                break
        if root is None:
            raise FsError("OUTSIDE_ROOT",
                          "%s is outside the authorized roots: %r" % (what, raw))
        if key in self._roots:
            raise FsError("ROOT_MUTATION",
                          "%s targets an authorized root itself: %r" % (what, raw))
        if _is_system_path(canon):
            raise FsError("SYSTEM_PATH",
                          "%s targets a protected system location: %r" % (what, raw))
        self._reject_reparse_chain(canon, what, raw)
        return canon

    def _reject_reparse_chain(self, canon, what, raw):
        """Reject reparse points/symlinks on the path and existing ancestors."""
        drive = os.path.splitdrive(canon)[0] + os.sep
        rel = os.path.relpath(canon, drive)
        cur = drive
        for comp in ([drive] if rel in (".", "") else rel.split(os.sep)):
            cur = drive if comp == drive else os.path.join(cur, comp)
            exists = os.path.lexists(cur)
            if not exists:
                return
            if os.path.islink(cur) or _is_reparse(cur):
                raise FsError("REPARSE_POINT",
                              "%s crosses a symlink/reparse point at %r: %r"
                              % (what, cur, raw))

    def _reject_tree_reparse(self, root_path, raw):
        """Reject any symlink/junction inside an existing source tree."""
        stack = [root_path]
        seen = set()
        while stack:
            cur = stack.pop()
            try:
                st = os.lstat(cur)
            except OSError:
                continue
            key = (st.st_dev, st.st_ino) if hasattr(st, "st_ino") else cur.lower()
            if key in seen:
                raise FsError("REPARSE_POINT",
                              "directory tree contains a cycle at %r: %r" % (cur, raw))
            seen.add(key)
            if os.path.islink(cur) or _is_reparse(cur):
                raise FsError("REPARSE_POINT",
                              "directory tree contains a reparse point at %r: %r"
                              % (cur, raw))
            if os.path.isdir(cur) and not os.path.islink(cur) \
                    and not _is_reparse(cur):
                try:
                    with os.scandir(cur) as it:
                        entries = list(it)
                except OSError as exc:
                    raise FsError(_win_code(exc),
                                  "cannot scan directory %r: %s" % (cur, exc))
                for entry in entries:
                    try:
                        if entry.is_symlink() or _is_reparse(entry.path):
                            raise FsError(
                                "REPARSE_POINT",
                                "directory tree contains a reparse point at %r: %r"
                                % (entry.path, raw))
                    except FsError:
                        raise
                    except OSError:
                        continue
                    stack.append(entry.path)

    def _check_owner(self, canon, raw):
        if self._owner_check is None:
            raise FsError("OWNER_DENIED",
                          "no explicit owner authorization configured; "
                          "denying by default for %r" % raw)
        try:
            allowed = self._owner_check(canon)
        except Exception as exc:
            raise FsError("OWNER_DENIED",
                          "owner policy error for %r: %s" % (raw, exc))
        if not allowed:
            raise FsError("OWNER_DENIED",
                          "caller is not the owner of %r" % raw)

    # -- public API ------------------------------------------------------

    def execute(self, action, items, options=None):
        """Run a batch action. Never raises for per-item failures."""
        options = options or {}
        action = _ACTION_ALIASES.get(action, action)
        collision = options.get("collision", COLLISION_FAIL)
        if collision not in COLLISIONS:
            return self._result(None, action, OP_FAILED, [],
                                "INVALID_REQUEST: unknown collision policy %r"
                                % collision)
        if action not in ("rename", "create_folder", "copy", "move",
                          "delete_recycle"):
            return self._result(None, action, OP_FAILED, [],
                                "INVALID_REQUEST: unknown action %r" % action)
        if not isinstance(items, list) or not items:
            return self._result(None, action, OP_FAILED, [],
                                "INVALID_REQUEST: items must be a non-empty list")
        op_id = uuid.uuid4().hex
        with self._lock:
            outcomes = []
            for entry in items:
                try:
                    record = self._dispatch(action, entry, collision, op_id)
                    outcomes.append(record)
                except _SkipItem as exc:
                    outcomes.append({
                        "src": _safe_src(entry),
                        "dst": exc.dst,
                        "status": STATUS_SKIPPED,
                        "code": "SKIPPED",
                        "error": None,
                    })
                except FsError as exc:
                    outcomes.append({
                        "src": _safe_src(entry),
                        "dst": _safe_dst(entry),
                        "status": STATUS_FAILED,
                        "code": exc.code,
                        "error": "%s: %s" % (exc.code, exc),
                    })
                except Exception as exc:  # defensive: batch must survive
                    outcomes.append({
                        "src": _safe_src(entry),
                        "dst": _safe_dst(entry),
                        "status": STATUS_FAILED,
                        "code": "UNEXPECTED",
                        "error": "UNEXPECTED: %r" % exc,
                    })
            status = _batch_status(outcomes)
            journal_items = [o for o in outcomes if o["status"] == STATUS_OK
                             and o.get("journal")]
            journal_error = None
            journal_pending = None
            if journal_items:
                try:
                    self._journal_store({
                        "op_id": op_id,
                        "action": action,
                        "collision": collision,
                        "ts": time.time(),
                        "status": status,
                        "items": [o["journal"] for o in journal_items],
                    })
                except Exception as exc:
                    # FS effects already happened; report truthfully instead
                    # of stale SUCCESS or an unhandled exception.
                    journal_error = "JOURNAL_FAILED: %r" % exc
                    journal_pending = {
                        "op_id": op_id,
                        "action": action,
                        "collision": collision,
                        "ts": time.time(),
                        "status": status,
                        "items": [o["journal"] for o in journal_items],
                    }
                    if status == OP_SUCCESS:
                        status = OP_PARTIAL
            res = self._result(op_id, action, status, outcomes)
            if journal_error:
                res["journal_error"] = journal_error
                res["journal_pending"] = {
                    "op_id": journal_pending["op_id"],
                    "action": journal_pending["action"],
                    "items": len(journal_pending["items"]),
                }
            return res

    def undo(self, op_id):
        """Undo a journaled operation. Never overwrites user data."""
        with self._lock:
            record = self._ops.get(op_id)
            if record is None:
                return self._result(op_id, "undo", OP_FAILED, [],
                                    "UNKNOWN_OP: no journal entry for %r" % op_id)
            action = record["action"]
            if action == "delete_recycle":
                return self._result(
                    op_id, "undo", OP_UNSUPPORTED, [],
                    "UNSUPPORTED_UNDO: recycle-bin restores are manual; "
                    "open the Windows Recycle Bin, locate the exact item name, "
                    "and choose Restore. Automatic restore is not implemented "
                    "because exact-item identity cannot be proven.")
            if action not in _UNDOABLE:
                return self._result(op_id, "undo", OP_UNSUPPORTED, [],
                                    "UNSUPPORTED_UNDO: action %r cannot be undone"
                                    % action)
            outcomes = []
            for item in record["items"]:
                try:
                    outcomes.append(self._undo_item(action, item))
                except FsError as exc:
                    outcomes.append({
                        "src": item.get("dst"),
                        "dst": item.get("src"),
                        "status": STATUS_FAILED,
                        "code": exc.code,
                        "error": "%s: %s" % (exc.code, exc),
                    })
            return self._result(op_id, "undo-%s" % action,
                                _batch_status(outcomes), outcomes)

    # -- dispatch --------------------------------------------------------

    def _dispatch(self, action, entry, collision, op_id=None):
        if not isinstance(entry, dict):
            raise FsError("INVALID_REQUEST", "item must be an object: %r" % entry)
        if action == "rename":
            return self._do_rename(entry, collision)
        if action == "create_folder":
            return self._do_create_folder(entry, collision)
        if action == "copy":
            return self._do_copy(entry, collision)
        if action == "move":
            return self._do_move(entry, collision, op_id)
        if action == "delete_recycle":
            return self._do_recycle(entry)
        raise FsError("INVALID_REQUEST", "unknown action %r" % action)

    # -- actions ----------------------------------------------------------

    def _do_rename(self, entry, collision):
        src_raw = entry.get("src")
        src = self._resolve(src_raw, "src")
        self._check_owner(src, src_raw)
        if not os.path.lexists(src):
            raise FsError("NOT_FOUND", "source does not exist: %r" % src_raw)
        if os.path.isdir(src) and not os.path.islink(src):
            self._reject_tree_reparse(src, src_raw)
        parent = os.path.dirname(src)
        if entry.get("dst"):
            dst = self._resolve(entry.get("dst"), "dst")
            if os.path.dirname(dst) != parent:
                raise FsError("INVALID_REQUEST",
                              "rename must stay in the same directory; "
                              "use move to relocate: %r" % entry.get("dst"))
        elif entry.get("new_name"):
            name = entry["new_name"]
            if not isinstance(name, str) or any(c in name for c in "\\/:"):
                raise FsError("INVALID_REQUEST",
                              "new_name must be a plain file name: %r" % name)
            dst = self._resolve(os.path.join(parent, name), "dst")
        else:
            raise FsError("INVALID_REQUEST",
                          "rename needs 'dst' or 'new_name'")
        self._check_owner(dst, entry.get("dst") or entry.get("new_name"))
        dst = self._apply_collision(dst, collision, is_dir_hint=os.path.isdir(src))
        try:
            pre = _identity(src)
        except OSError as exc:
            raise FsError(_win_code(exc),
                          "rename source unreadable %r: %s" % (src_raw, exc))
        # Re-validate + re-authorize immediately before mutation.
        self._reject_reparse_chain(parent, "parent", parent)
        self._check_owner(src, src_raw)
        self._check_owner(dst, entry.get("dst") or entry.get("new_name"))
        ok, err = _move_file_noreplace(src, dst)
        if not ok:
            raise FsError(_WINERROR_MAP.get(err, "OS_ERROR"),
                          "rename failed %r -> %r (winerror %s)"
                          % (src_raw, dst, err))
        post = _identity(dst)
        if not _verify_copy_identity(src=None, dst=dst, dst_post=post,
                                     src_pre=pre, is_dir=pre["is_dir"]):
            # Rename landed but verification failed; do not journal as clean.
            raise FsError("VERIFY_FAILED",
                          "rename verify failed %r -> %r; manual check needed"
                          % (src_raw, dst))
        return _ok_outcome(src_raw, dst, "rename",
                           {"src": src, "dst": dst,
                            "post": post, "pre_is_dir": pre["is_dir"],
                            "pre": pre})

    def _do_create_folder(self, entry, collision):
        raw = entry.get("path", entry.get("dst", entry.get("src")))
        path = self._resolve(raw, "path")
        self._check_owner(path, raw)
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            raise FsError("NOT_FOUND",
                          "parent directory does not exist: %r" % parent)
        self._reject_reparse_chain(parent, "parent", parent)
        target = self._apply_collision(path, collision, is_dir_hint=True)
        self._check_owner(target, raw)
        self._reject_reparse_chain(parent, "parent", parent)
        try:
            os.mkdir(_long_path(target))
        except OSError as exc:
            raise FsError(_win_code(exc),
                          "create_folder failed %r: %s" % (raw, exc))
        return _ok_outcome(raw, target, "create_folder",
                           {"src": None, "dst": target,
                            "post": _identity(target), "pre_is_dir": True})

    def _do_copy(self, entry, collision):
        src_raw, dst_raw = entry.get("src"), entry.get("dst")
        src = self._resolve(src_raw, "src")
        self._check_owner(src, src_raw)
        if not os.path.lexists(src):
            raise FsError("NOT_FOUND", "source does not exist: %r" % src_raw)
        if os.path.isdir(src) and not os.path.islink(src):
            self._reject_tree_reparse(src, src_raw)
        if "dst" not in entry:
            raise FsError("INVALID_REQUEST", "copy needs 'dst'")
        dst = self._resolve(dst_raw, "dst")
        self._check_owner(dst, dst_raw)
        if _same_path(src, dst):
            raise FsError("INVALID_REQUEST",
                          "copy source and destination are identical: %r" % src_raw)
        if os.path.isdir(src) and _is_within(dst, src):
            raise FsError("INVALID_REQUEST",
                          "cannot copy a folder into itself: %r" % dst_raw)
        dst_parent = os.path.dirname(dst)
        if not os.path.isdir(dst_parent):
            raise FsError("NOT_FOUND",
                          "destination parent does not exist: %r" % dst_parent)
        is_dir = os.path.isdir(src) and not os.path.islink(src)
        dst = self._apply_collision(dst, collision, is_dir_hint=is_dir)
        # Re-validate destination's ancestor chain + re-authorize right
        # before acting (narrows post-validation race; atomic primitives
        # below turn a lost race into ITEM_EXISTS, not an overwrite).
        self._reject_reparse_chain(dst_parent, "dst parent", dst_raw)
        self._check_owner(src, src_raw)
        self._check_owner(dst, dst_raw)
        try:
            src_pre = _identity(src)
        except OSError as exc:
            raise FsError(_win_code(exc),
                          "copy source unreadable %r: %s" % (src_raw, exc))
        try:
            if is_dir:
                self._copy_tree(src, dst)
            else:
                _copy_stream_noreplace(src, dst)
        except FileExistsError:
            raise FsError("ITEM_EXISTS",
                          "destination appeared during copy, not overwriting: %r"
                          % dst_raw)
        except OSError as exc:
            raise FsError(_win_code(exc),
                          "copy failed %r -> %r: %s" % (src_raw, dst_raw, exc))
        try:
            dst_post = _identity(dst)
        except OSError as exc:
            try:
                self._remove_tree(dst)
            except Exception:
                pass
            raise FsError("VERIFY_FAILED",
                          "copy target unreadable %r: %s" % (dst_raw, exc))
        if not _verify_copy_identity(src=src, dst=dst, dst_post=dst_post,
                                     src_pre=src_pre, is_dir=is_dir):
            try:
                self._remove_tree(dst)
            except Exception:
                pass
            raise FsError("VERIFY_FAILED",
                          "copy verify failed %r -> %r; copy removed"
                          % (src_raw, dst_raw))
        return _ok_outcome(src_raw, dst, "copy",
                           {"src": src, "dst": dst,
                            "post": dst_post, "pre_is_dir": False,
                            "pre": src_pre})

    def _do_move(self, entry, collision, op_id=None):
        src_raw, dst_raw = entry.get("src"), entry.get("dst")
        src = self._resolve(src_raw, "src")
        self._check_owner(src, src_raw)
        if not os.path.lexists(src):
            raise FsError("NOT_FOUND", "source does not exist: %r" % src_raw)
        if os.path.isdir(src) and not os.path.islink(src):
            self._reject_tree_reparse(src, src_raw)
        if "dst" not in entry:
            raise FsError("INVALID_REQUEST", "move needs 'dst'")
        dst = self._resolve(dst_raw, "dst")
        self._check_owner(dst, dst_raw)
        if _same_path(src, dst):
            raise FsError("INVALID_REQUEST",
                          "move source and destination are identical: %r" % src_raw)
        if os.path.isdir(src) and _is_within(dst, src):
            raise FsError("INVALID_REQUEST",
                          "cannot move a folder into itself: %r" % dst_raw)
        dst_parent = os.path.dirname(dst)
        if not os.path.isdir(dst_parent):
            raise FsError("NOT_FOUND",
                          "destination parent does not exist: %r" % dst_parent)
        is_dir = os.path.isdir(src) and not os.path.islink(src)
        dst = self._apply_collision(dst, collision, is_dir_hint=is_dir)
        self._reject_reparse_chain(dst_parent, "dst parent", dst_raw)
        try:
            pre = _identity(src)
        except OSError as exc:
            raise FsError(_win_code(exc),
                          "move source unreadable %r: %s" % (src_raw, exc))
        self._check_owner(src, src_raw)
        self._check_owner(dst, dst_raw)
        ok, err = _move_file_noreplace(src, dst)
        if ok:
            pass
        elif err == _ERROR_NOT_SAME_DEVICE:
            journal = self._cross_volume_move(src, dst, is_dir, op_id,
                                              src_raw, dst_raw, pre)
            return _ok_outcome(src_raw, dst, "move", journal)
        else:
            raise FsError(_WINERROR_MAP.get(err, "OS_ERROR"),
                          "move failed %r -> %r (winerror %s)"
                          % (src_raw, dst, err))
        try:
            post = _identity(dst)
        except OSError as exc:
            raise FsError(_win_code(exc),
                          "move target unreadable %r: %s" % (dst_raw, exc))
        if not _verify_copy_identity(src=None, dst=dst, dst_post=post,
                                     src_pre=pre, is_dir=pre["is_dir"]):
            raise FsError("VERIFY_FAILED",
                          "move verify failed %r -> %r; manual check needed"
                          % (src_raw, dst))
        return _ok_outcome(src_raw, dst, "move",
                           {"src": src, "dst": dst,
                            "post": post, "pre_is_dir": pre["is_dir"],
                            "pre": pre})

    def _do_recycle(self, entry):
        src_raw = entry.get("src")
        src = self._resolve(src_raw, "src")
        self._check_owner(src, src_raw)
        if not os.path.lexists(src):
            raise FsError("NOT_FOUND", "source does not exist: %r" % src_raw)
        if os.path.isdir(src) and not os.path.islink(src):
            self._reject_tree_reparse(src, src_raw)
        if _drive_type_blocked(src):
            raise FsError("RECYCLE_UNSUPPORTED",
                          "volume does not support the Recycle Bin; "
                          "refusing permanent delete of %r" % src_raw)
        self._check_owner(src, src_raw)
        try:
            res = self._shell.recycle(src)
        except Exception as exc:
            raise FsError("RECYCLE_FAILED",
                          "recycle backend error for %r: %r" % (src_raw, exc))
        code = (res or {}).get("code", 1)
        if (res or {}).get("aborted"):
            raise FsError("RECYCLE_ABORTED",
                          "recycle operation was aborted for %r; item left in place"
                          % src_raw)
        if code != 0:
            detail = (res or {}).get("error")
            raise FsError("RECYCLE_FAILED",
                          "recycle failed for %r with shell code %s%s; "
                          "item left in place, nothing was permanently deleted"
                          % (src_raw, code, (": %s" % detail) if detail else ""))
        if os.path.lexists(src):
            raise FsError("RECYCLE_FAILED",
                          "recycle reported success but %r is still present" % src_raw)
        return {"src": src_raw, "dst": None, "status": STATUS_OK,
                "code": "RECYCLED", "error": None,
                "journal": {"src": src, "dst": None, "post": None,
                            "pre_is_dir": False}}

    # -- collision --------------------------------------------------------

    def _apply_collision(self, dst, collision, is_dir_hint=False):
        if not os.path.lexists(dst):
            return dst
        if collision == COLLISION_FAIL:
            raise FsError("ITEM_EXISTS",
                          "destination already exists, not overwriting: %r" % dst)
        if collision == COLLISION_SKIP:
            raise _SkipItem(dst)
        assert collision == COLLISION_RENAME_NEW
        return _unique_sibling(dst, is_dir_hint)

    # -- cross-volume move -------------------------------------------------

    def _cross_volume_move(self, src, dst, is_dir, op_id, src_raw, dst_raw,
                           src_pre):
        """Staged copy + strong verify + journaled staged outcome + collecting
        source cleanup. Never partially destroys source without a journal.

        Returns the final journal dict. On partial source-cleanup failure it
        updates the staged journal and raises FsError(CROSS_VOLUME_CLEANUP_*)
        with recovery metadata; dst is complete+verified and the remaining
        source files are preserved.
        """
        # 1. Staged copy (engine-owned dst only; O_EXCL/mkdir, no overwrite).
        try:
            if is_dir:
                self._copy_tree(src, dst)
            else:
                _copy_stream_noreplace(src, dst)
        except FileExistsError:
            raise FsError("ITEM_EXISTS",
                          "destination appeared during cross-volume copy, "
                          "source kept: %r" % dst_raw)
        except OSError as exc:
            try:
                self._remove_tree(dst)
            except Exception:
                pass
            raise FsError(_win_code(exc),
                          "cross-volume copy failed %r -> %r: %s; source kept"
                          % (src_raw, dst_raw, exc))
        # 2. Strong verification (size + SHA-256 per file + structure).
        try:
            verified = self._tree_equal_strong(src, dst) if is_dir else \
                _verify_copy_identity(src=src, dst=dst,
                                      dst_post=_identity(dst),
                                      src_pre=src_pre, is_dir=False)
        except FsError:
            raise
        except Exception as exc:
            try:
                self._remove_tree(dst)
            except Exception:
                pass
            raise FsError("VERIFY_FAILED",
                          "cross-volume verify error %r -> %r: %r; copy removed"
                          % (src_raw, dst_raw, exc))
        if not verified:
            try:
                self._remove_tree(dst)
            except Exception:
                pass
            raise FsError("VERIFY_FAILED",
                          "cross-volume verify failed %r -> %r; copy removed, "
                          "source kept" % (src_raw, dst_raw))
        dst_post = _identity(dst)
        staged = {"src": src, "dst": dst, "post": dst_post,
                  "pre": src_pre, "pre_is_dir": src_pre["is_dir"],
                  "cross_volume": True,
                  "staged": True,
                  "manifest": _safe_manifest(dst_post)}
        # 3. Journal the staged outcome BEFORE any source cleanup, so a
        # mid-cleanup failure still has recoverable evidence + resume/undo
        # metadata. Journal-write failure here aborts cleanup (source kept).
        if op_id is not None:
            try:
                self._journal_stage(op_id, "move", staged)
            except Exception as exc:
                try:
                    self._remove_tree(dst)
                except Exception:
                    pass
                raise FsError("JOURNAL_FAILED",
                              "journal stage failed before source cleanup; "
                              "both copies preserved for %r -> %r: %r"
                              % (src_raw, dst_raw, exc))
        # 4. Collecting source cleanup (per-file errors preserved).
        deleted, remaining, errors = _remove_tree_collect(src)
        if remaining or errors:
            # Update staged journal with resume metadata (best effort; the
            # staged record from step 3 already exists even if this fails).
            try:
                staged2 = dict(staged)
                staged2["cleanup_deleted"] = deleted
                staged2["cleanup_remaining"] = remaining
                staged2["cleanup_errors"] = errors[:10]
                staged2["cleanup_partial"] = True
                if op_id is not None:
                    self._journal_stage(op_id, "move", staged2)
            except Exception:
                pass
            raise FsError("CROSS_VOLUME_CLEANUP_PARTIAL",
                          "cross-volume copy verified %r -> %r but source "
                          "cleanup incomplete (%d deleted, %d remaining); "
                          "journal holds resume/undo metadata, no data lost"
                          % (src_raw, dst_raw, len(deleted), len(remaining)))
        final = dict(staged)
        final["staged"] = False
        return final

    def _journal_stage(self, op_id, action, staged_item):
        """Persist a staged cross-volume outcome before source cleanup."""
        with self._lock:
            rec = self._ops.get(op_id)
            if rec is None:
                rec = {"op_id": op_id, "action": action,
                       "collision": COLLISION_FAIL, "ts": time.time(),
                       "status": OP_PARTIAL, "items": []}
                self._ops[op_id] = rec
            # Replace any staged entry for the same src/dst, else append.
            items = rec.get("items", [])
            replaced = False
            for i, it in enumerate(items):
                if it.get("src") == staged_item.get("src") and \
                        it.get("dst") == staged_item.get("dst"):
                    items[i] = staged_item
                    replaced = True
                    break
            if not replaced:
                items.append(staged_item)
            rec["items"] = items
            rec["ts"] = time.time()
            self._journal_write_merged()

    def _copy_tree(self, src, dst):
        if os.path.islink(src) or _is_reparse(src):
            raise FsError("REPARSE_POINT",
                          "tree contains a reparse point at %r" % src)
        os.mkdir(_long_path(dst))
        try:
            with os.scandir(src) as it:
                entries = list(it)
            for entry in entries:
                if entry.is_symlink() or _is_reparse(entry.path):
                    raise FsError("REPARSE_POINT",
                                  "tree contains a reparse point at %r" % entry.path)
                s, d = entry.path, os.path.join(dst, entry.name)
                if entry.is_dir(follow_symlinks=False):
                    self._copy_tree(s, d)
                else:
                    _copy_stream_noreplace(s, d)
            shutil.copystat(_long_path(src), _long_path(dst),
                            follow_symlinks=False)
        except BaseException:
            self._remove_tree(dst)
            raise

    def _tree_equal_strong(self, a, b):
        try:
            ma = _tree_manifest(a)
            mb = _tree_manifest(b)
        except FsError:
            return False
        return ma == mb

    @staticmethod
    def _remove_tree(root):
        """Remove an ENGINE-OWNED tree only (failed copies/move sources)."""
        if os.path.islink(root) or _is_reparse(root):
            raise FsError("REPARSE_POINT",
                          "refusing to remove through a link: %r" % root)
        # Pre-scan top-down BEFORE any deletion: os.walk(followlinks=False)
        # still descends into junctions (they are not symlinks), so a
        # bottom-up walk alone would already enumerate victim files through
        # the junction before we see the link. Raise at the parent first.
        for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                     followlinks=False):
            for name in list(dirnames):
                pp = os.path.join(dirpath, name)
                if os.path.islink(pp) or _is_reparse(pp):
                    raise FsError("REPARSE_POINT",
                                  "refusing to remove through a link: %r" % pp)
            for name in filenames:
                pp = os.path.join(dirpath, name)
                if os.path.islink(pp) or _is_reparse(pp):
                    raise FsError("REPARSE_POINT",
                                  "refusing to remove through a link: %r" % pp)
        for dirpath, dirnames, filenames in os.walk(root, topdown=False,
                                                     followlinks=False):
            for name in filenames:
                p = os.path.join(dirpath, name)
                if os.path.islink(p) or _is_reparse(p):
                    raise FsError("REPARSE_POINT",
                                  "refusing to remove through a link: %r" % p)
                os.remove(_long_path(p))
            for name in dirnames:
                p = os.path.join(dirpath, name)
                if os.path.islink(p) or _is_reparse(p):
                    raise FsError("REPARSE_POINT",
                                  "refusing to remove through a link: %r" % p)
                os.rmdir(_long_path(p))
        os.rmdir(_long_path(root))

    # -- undo items --------------------------------------------------------

    def _undo_item(self, action, item):
        if action in ("rename", "move"):
            return self._undo_relocate(item)
        if action == "copy":
            return self._undo_copy(item)
        if action == "create_folder":
            return self._undo_mkdir(item)
        raise FsError("UNSUPPORTED_UNDO", "cannot undo %r" % action)

    def _undo_relocate(self, item):
        # Journal is user-editable: re-resolve every path through the
        # full system guard (containment + system-path + reparse chain).
        cur = self._resolve(item.get("dst"), "undo cur")
        orig = self._resolve(item.get("src"), "undo orig")
        self._check_owner(cur, cur)
        self._check_owner(orig, orig)
        if not os.path.lexists(cur):
            raise FsError("NOT_FOUND",
                          "undo: current location is gone: %r" % cur)
        if not _identity_matches(cur, item["post"]):
            raise FsError("MODIFIED",
                          "undo: current file was modified or replaced; "
                          "refusing to overwrite: %r" % cur)
        if os.path.lexists(orig):
            raise FsError("DEST_EXISTS",
                          "undo: original name is occupied, not overwriting: %r"
                          % orig)
        # Re-validate ancestors + re-authorize immediately before mutation.
        self._reject_reparse_chain(os.path.dirname(orig) or orig,
                                   "undo parent", orig)
        self._check_owner(cur, cur)
        self._check_owner(orig, orig)
        ok, err = _move_file_noreplace(cur, orig)
        if ok:
            return {"src": cur, "dst": orig, "status": STATUS_OK,
                    "code": "UNDONE", "error": None}
        if err == _ERROR_NOT_SAME_DEVICE:
            # Cross-volume undo: staged reverse copy + verify + cleanup.
            return self._undo_relocate_cross_volume(cur, orig, item)
        raise FsError(_WINERROR_MAP.get(err, "OS_ERROR"),
                      "undo move failed %r -> %r (winerror %s)"
                      % (cur, orig, err))

    def _undo_relocate_cross_volume(self, cur, orig, item):
        is_dir = bool(item.get("pre_is_dir"))
        try:
            if is_dir:
                self._copy_tree(cur, orig)
                if not self._tree_equal_strong(cur, orig):
                    try:
                        self._remove_tree(orig)
                    except Exception:
                        pass
                    raise FsError("VERIFY_FAILED",
                                  "undo cross-volume verify failed; copy removed")
            else:
                _copy_stream_noreplace(cur, orig)
                if not _identity_matches(orig, item["post"],
                                         compare_vol=False):
                    try:
                        os.remove(_long_path(orig))
                    except OSError:
                        pass
                    raise FsError("VERIFY_FAILED",
                                  "undo cross-volume verify failed; copy removed")
        except FileExistsError:
            raise FsError("DEST_EXISTS",
                          "undo: original appeared during cross-volume undo, "
                          "preserving both: %r" % orig)
        except FsError:
            raise
        except OSError as exc:
            raise FsError(_win_code(exc),
                          "undo cross-volume copy failed: %s" % exc)
        # Cleanup the moved-away copy only after verified reverse copy.
        if os.path.islink(cur) or _is_reparse(cur):
            raise FsError("REPARSE_POINT",
                          "undo: refusing to remove through a link: %r" % cur)
        try:
            if is_dir:
                self._remove_tree(cur)
            else:
                os.remove(_long_path(cur))
        except FsError:
            raise
        except OSError as exc:
            raise FsError("CROSS_VOLUME_CLEANUP_PARTIAL",
                          "undo reverse copy verified but cleanup of %r "
                          "incomplete: %s; both copies exist, manual resume"
                          % (cur, exc))
        return {"src": cur, "dst": orig, "status": STATUS_OK,
                "code": "UNDONE", "error": None}

    def _undo_copy(self, item):
        # Journal is user-editable: re-resolve through the system guard.
        cur = self._resolve(item.get("dst"), "undo cur")
        self._check_owner(cur, cur)
        if not os.path.lexists(cur):
            raise FsError("NOT_FOUND", "undo: copy is already gone: %r" % cur)
        if os.path.islink(cur) or _is_reparse(cur):
            raise FsError("REPARSE_POINT",
                          "undo: refusing to remove through a link: %r" % cur)
        if not _identity_matches(cur, item["post"]):
            raise FsError("MODIFIED",
                          "undo: copy was modified after the operation; "
                          "preserving user edits: %r" % cur)
        self._check_owner(cur, cur)
        if os.path.isdir(cur) and not os.path.islink(cur) \
                and not _is_reparse(cur):
            try:
                os.rmdir(_long_path(cur))  # only if empty: preserve contents
            except OSError as exc:
                raise FsError("NOT_EMPTY",
                              "undo: folder is no longer empty, preserving: %r"
                              % cur)
        else:
            os.remove(_long_path(cur))
        return {"src": cur, "dst": None, "status": STATUS_OK,
                "code": "UNDONE", "error": None}

    def _undo_mkdir(self, item):
        # Journal is user-editable: re-resolve through the system guard.
        cur = self._resolve(item.get("dst"), "undo cur")
        self._check_owner(cur, cur)
        if not os.path.lexists(cur):
            raise FsError("NOT_FOUND", "undo: folder is already gone: %r" % cur)
        if not os.path.isdir(cur) or os.path.islink(cur) or _is_reparse(cur):
            raise FsError("MODIFIED",
                          "undo: path is no longer a plain folder: %r" % cur)
        self._check_owner(cur, cur)
        try:
            os.rmdir(_long_path(cur))  # only if empty: preserve user files
        except OSError:
            raise FsError("NOT_EMPTY",
                          "undo: folder is no longer empty, preserving: %r" % cur)
        return {"src": cur, "dst": None, "status": STATUS_OK,
                "code": "UNDONE", "error": None}

    # -- journal -----------------------------------------------------------

    def _journal_store(self, record):
        with _JOURNAL_IO_LOCK:
            with self._lock:
                self._ops[record["op_id"]] = record
            self._journal_write_merged()

    def _journal_write_merged(self):
        """Cross-process serialized, atomic, bounded, merged journal write.

        Reads the on-disk journal, unions it with in-memory ops (never
        silent last-writer-wins), prunes to max_ops newest, and publishes
        atomically via tmp + fsync + os.replace. Raises on I/O failure so
        the caller can report a truthful partial outcome.
        """
        with _JOURNAL_IO_LOCK:
            with self._lock:
                mem = dict(self._ops)
                max_ops = self._max_ops
                jpath = self._journal_path
            if not jpath:
                return
            lock_fh = None
            tmp = None
            lock_path = jpath + ".lock"
            try:
                parent = os.path.dirname(os.path.abspath(jpath))
                os.makedirs(parent, exist_ok=True)
                # Cross-process lock (best effort on Windows via msvcrt).
                acquired = False
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(lock_path)),
                                exist_ok=True)
                    lock_fh = open(lock_path, "ab+")
                    if _msvcrt is not None:
                        import time as _t
                        deadline = _t.time() + 10
                        while True:
                            try:
                                lock_fh.seek(0)
                                _msvcrt.locking(lock_fh.fileno(),
                                                _msvcrt.LK_NBLCK, 1)
                                acquired = True
                                break
                            except OSError:
                                if _t.time() >= deadline:
                                    break
                                _t.sleep(0.02)
                    else:
                        acquired = True
                except Exception:
                    lock_fh = lock_fh  # proceed; RLock still guards threads
                if _msvcrt is not None and lock_fh is not None                         and not acquired:
                    raise FsError("JOURNAL_FAILED",
                                  "journal lock not acquired for %r; "
                                  "refusing unlocked write" % jpath)
                disk_ops = {}
                try:
                    with open(jpath, encoding="utf-8") as fh:
                        data = json.load(fh)
                    for rec in _extract_records(data):
                        disk_ops[rec["op_id"]] = rec
                    if not disk_ops:
                        _ops_raw = data.get("ops") if isinstance(
                            data, dict) else None
                        if isinstance(data, dict) and _ops_raw == []:
                            pass  # legitimate empty journal
                        elif os.path.exists(jpath):
                            try:
                                bak = jpath + ".bak"
                                with open(jpath, "rb") as _s,                                         open(bak, "wb") as _d:
                                    _d.write(_s.read())
                            except Exception:
                                pass
                            raise FsError(
                                "JOURNAL_FAILED",
                                "journal schema invalid at %r, backup kept, "
                                "not truncating" % jpath)
                except FileNotFoundError:
                    disk_ops = {}
                except FsError:
                    raise
                except OSError as exc:
                    raise FsError("JOURNAL_FAILED",
                                  "journal read failed for %r: %s"
                                  % (jpath, exc))
                except ValueError as exc:
                    try:
                        bak = jpath + ".bak"
                        with open(jpath, "rb") as _s,                                 open(bak, "wb") as _d:
                            _d.write(_s.read())
                    except Exception:
                        pass
                    raise FsError("JOURNAL_FAILED",
                                  "journal corrupt at %r, backup kept, "
                                  "not truncating: %s" % (jpath, exc))
                merged = dict(disk_ops)
                merged.update(mem)
                # Prune to newest max_ops by ts.
                if len(merged) > max_ops:
                    ordered = sorted(merged.values(),
                                     key=lambda r: (r.get("ts", 0),
                                                    r.get("op_id", "")))
                    for rec in ordered[:len(merged) - max_ops]:
                        merged.pop(rec["op_id"], None)
                tmp = "%s.tmp-%s-%s-%s" % (jpath, os.getpid(), threading.get_ident(), uuid.uuid4().hex)
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"version": 1, "ops": sorted(
                        merged.values(), key=lambda r: (r.get("ts", 0),
                                                       r.get("op_id", "")))},
                        fh, ensure_ascii=False, indent=1)
                    fh.flush()
                    os.fsync(fh.fileno())
                try:
                    with open(tmp, encoding="utf-8") as vfh:
                        vdata = json.load(vfh)
                    if not isinstance(vdata.get("ops"), list):
                        raise ValueError("verify: ops is not a list")
                except Exception as exc:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    tmp = None
                    raise FsError("JOURNAL_FAILED",
                                  "journal verify failed for %r: %s"
                                  % (jpath, exc))
                os.replace(tmp, jpath)
                tmp = None
                with self._lock:
                    self._ops = merged
            finally:
                if tmp is not None:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                if lock_fh is not None:
                    try:
                        if _msvcrt is not None:
                            try:
                                lock_fh.seek(0)
                                _msvcrt.locking(lock_fh.fileno(),
                                                _msvcrt.LK_UNLCK, 1)
                            except OSError:
                                pass
                        lock_fh.close()
                    except Exception:
                        pass

    def _load_journal(self):
        try:
            with open(self._journal_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return  # corrupt JSON or unreadable: start empty, never replay
        try:
            recs = _extract_records(data)
        except Exception:
            return
        with self._lock:
            for rec in recs[-self._max_ops:]:
                self._ops[rec["op_id"]] = rec

    # -- results ------------------------------------------------------------

    @staticmethod
    def _result(op_id, action, status, items, error=None):
        clean = []
        for o in items:
            c = {k: o.get(k) for k in ("src", "dst", "status", "code", "error")}
            clean.append(c)
        res = {"op_id": op_id, "action": action, "status": status,
               "items": clean}
        if error:
            res["error"] = error
        return res


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

class _SkipItem(Exception):
    def __init__(self, dst):
        super().__init__(dst)
        self.dst = dst


def _drive_abs(s):
    return (len(s) >= 3 and s[1] == ":" and s[2] == "\\"
            and s[0].isalpha() and ("\\" not in s[0]))


def _is_system_path(canon):
    cands = []
    for var in ("SystemRoot", "windir", "ProgramFiles",
                "ProgramFiles(x86)", "ProgramW6432"):
        val = os.environ.get(var)
        if val:
            cands.append(os.path.normcase(os.path.normpath(val)))
    key = os.path.normcase(canon)
    for c in cands:
        if key == c or key.startswith(c + os.sep):
            return True
    return False


def _drive_type_blocked(path):
    """True when the volume must not be recycled (never permanent delete).

    Only DRIVE_FIXED (3) with a confirmable bin is allowed. All other
    types (UNKNOWN=0, NO_ROOT=1, REMOVABLE=2, REMOTE=4, CDROM=5, RAMDISK=6)
    or an unconfirmable bin/policy state are blocked.
    """
    try:
        from localdocs_shellops import volume_supports_recycle
        supported, _reason = volume_supports_recycle(path)
        return not supported
    except Exception:
        pass
    try:
        GetDriveTypeW = _kernel32().GetDriveTypeW
        GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
        GetDriveTypeW.restype = ctypes.c_uint
        root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
        dtype = int(GetDriveTypeW(root))
        # Allow ONLY fixed disks; everything else lacks a reliable bin.
        return dtype != 3
    except Exception:
        return True  # fail closed when the type cannot be determined


def _same_path(a, b):
    try:
        a = _expand_83(a)
    except Exception:
        pass
    try:
        b = _expand_83(b)
    except Exception:
        pass
    return os.path.normcase(os.path.normpath(a)) == \
        os.path.normcase(os.path.normpath(b))


def _is_within(child, parent):
    try:
        child = _expand_83(child)
    except Exception:
        pass
    try:
        parent = _expand_83(parent)
    except Exception:
        pass
    ck = os.path.normcase(os.path.normpath(child))
    pk = os.path.normcase(os.path.normpath(parent))
    base = pk.rstrip(os.sep)
    return ck == pk or ck == base or ck.startswith(base + os.sep)


def _identity(path):
    st = os.stat(_long_path(path))
    is_dir = os.path.isdir(path) and not os.path.islink(path)
    fid = _get_volume_file_id(path)
    vol, fidx = fid if fid is not None else (None, None)
    if is_dir:
        try:
            manifest = _tree_manifest(path)
        except FsError:
            raise
        except OSError:
            manifest = {}
        mhash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True,
                       default=str).encode("utf-8")).hexdigest()
        return {"is_dir": True, "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns), "vol": vol,
                "file_index": fidx, "manifest": manifest,
                "manifest_hash": mhash}
    return {"is_dir": False, "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "vol": vol,
            "file_index": fidx, "sha256": _hash_file(path)}


def _safe_manifest(post):
    if not isinstance(post, dict):
        return None
    m = post.get("manifest")
    if isinstance(m, dict) and len(m) <= 2000:
        return m
    return {"manifest_hash": post.get("manifest_hash")}


def _identity_matches(cur_path, stored_post, compare_vol=True):
    """Strong identity compare: vol/file-index + size/hash/manifest.

    Same-size/mtime replacement or edited bytes never matches. Missing
    stored fields (old journals) fail closed to MODIFIED.
    """
    if not isinstance(stored_post, dict):
        return False
    try:
        cur = _identity(cur_path)
    except OSError:
        return False
    if bool(cur.get("is_dir")) != bool(stored_post.get("is_dir")):
        return False
    if cur.get("is_dir"):
        if compare_vol and stored_post.get("vol") is not None and \
                stored_post.get("file_index") is not None:
            if cur.get("vol") != stored_post.get("vol") or \
                    cur.get("file_index") != stored_post.get("file_index"):
                return False
        sm = stored_post.get("manifest")
        if isinstance(sm, dict):
            if cur.get("manifest") != sm:
                return False
        elif stored_post.get("manifest_hash"):
            if cur.get("manifest_hash") != stored_post.get("manifest_hash"):
                return False
        else:
            return False
        return True
    # File: size + hash required; vol/file-index required when stored.
    if cur.get("size") != stored_post.get("size"):
        return False
    if not stored_post.get("sha256") or \
            cur.get("sha256") != stored_post.get("sha256"):
        return False
    if compare_vol and stored_post.get("vol") is not None and \
            stored_post.get("file_index") is not None:
        if cur.get("vol") != stored_post.get("vol") or \
                cur.get("file_index") != stored_post.get("file_index"):
            return False
    return True


def _verify_copy_identity(src, dst, dst_post, src_pre, is_dir):
    """Strong copy verification: size + SHA-256 (+ structure for trees)."""
    try:
        if is_dir:
            if not isinstance(dst_post, dict) or not dst_post.get("is_dir"):
                return False
            if src is None:
                return bool(dst_post.get("manifest_hash"))
            try:
                return _tree_manifest(src) == _tree_manifest(dst)
            except (OSError, FsError):
                return False
        if src is None:
            return bool(dst_post.get("sha256"))
        if dst_post.get("size") != src_pre.get("size"):
            return False
        if _hash_file(src) != _hash_file(dst):
            return False
        return dst_post.get("sha256") == _hash_file(dst)
    except (OSError, FsError):
        return False


def _remove_tree_collect(root):
    """Best-effort bottom-up removal collecting per-file outcomes.

    Returns (deleted, remaining, errors). Never raises for per-file
    failures; reparse points are left in place and reported.
    """
    deleted, remaining, errors = [], [], []
    if os.path.islink(root) or _is_reparse(root):
        return ([], [root], ["link preserved %r" % root])
    # Manual top-down collect with junction pruning: os.walk with
    # followlinks=False still descends into junctions (not symlinks), so
    # listing bottom-up first would already enumerate victim files through
    # the link. Prune reparse points before descending instead.
    order = []
    try:
        stack = [root]
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    entries = list(it)
            except OSError as exc:
                remaining.append(cur)
                errors.append("scan %r: %s" % (cur, exc))
                continue
            dirs, files = [], []
            for entry in entries:
                pp = entry.path
                try:
                    if os.path.islink(pp) or _is_reparse(pp):
                        remaining.append(pp)
                        errors.append("link preserved %r" % pp)
                        continue
                except OSError:
                    remaining.append(pp)
                    errors.append("link preserved %r" % pp)
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                        stack.append(pp)
                    else:
                        files.append(entry.name)
                except OSError as exc:
                    remaining.append(pp)
                    errors.append("scan %r: %s" % (pp, exc))
            order.append((cur, dirs, files))
    except OSError as exc:
        return (deleted, [root] + remaining,
                (["walk %r: %s" % (root, exc)] + errors))
    items = [(d, dn, fn) for (d, dn, fn) in order]
    for dirpath, dirnames, filenames in reversed(items):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if os.path.islink(p) or _is_reparse(p):
                remaining.append(p)
                errors.append("link preserved %r" % p)
                continue
            try:
                os.remove(_long_path(p))
                deleted.append(p)
            except OSError as exc:
                remaining.append(p)
                errors.append("remove %r: %s" % (p, exc))
        for name in dirnames:
            p = os.path.join(dirpath, name)
            if os.path.islink(p) or _is_reparse(p):
                remaining.append(p)
                errors.append("link preserved %r" % p)
                continue
            try:
                os.rmdir(_long_path(p))
                deleted.append(p)
            except OSError as exc:
                remaining.append(p)
                errors.append("rmdir %r: %s" % (p, exc))
    try:
        os.rmdir(_long_path(root))
        deleted.append(root)
    except OSError as exc:
        # If children remain, the root necessarily remains: avoid double
        # counting it when remaining already holds children.
        if not remaining:
            remaining.append(root)
        errors.append("rmdir %r: %s" % (root, exc))
    return (deleted, remaining, errors)


def _extract_records(data):
    """Validate journal schema; return list of good records (may be empty)."""
    if not isinstance(data, dict):
        return []
    ops = data.get("ops")
    if not isinstance(ops, list):
        return []
    good = []
    for rec in ops:
        if not isinstance(rec, dict):
            continue
        if not isinstance(rec.get("op_id"), str):
            continue
        if not isinstance(rec.get("action"), str):
            continue
        items = rec.get("items")
        if not isinstance(items, list):
            continue
        ok_items = True
        for it in items:
            if not isinstance(it, dict):
                ok_items = False
                break
        if not ok_items:
            continue
        if not isinstance(rec.get("ts"), (int, float)):
            rec = dict(rec)
            rec["ts"] = 0
        good.append(rec)
    return good


def _unique_sibling(dst, is_dir_hint=False):
    parent = os.path.dirname(dst)
    base = os.path.basename(dst)
    if not is_dir_hint and "." in base and not base.startswith("."):
        stem, dot, ext = base.rpartition(".")
        pattern = "%s (%%d).%s" % (stem, ext)
    else:
        pattern = "%s (%%d)" % base
    for i in range(1, 1000):
        cand = os.path.join(parent, pattern % i)
        if not os.path.lexists(cand):
            return cand
    raise FsError("ITEM_EXISTS",
                  "no free rename_new candidate near %r" % dst)


def _safe_src(entry):
    return entry.get("src") if isinstance(entry, dict) else None


def _safe_dst(entry):
    if isinstance(entry, dict):
        return entry.get("dst", entry.get("new_name", entry.get("path")))
    return None


def _ok_outcome(src_raw, dst, action, journal):
    return {"src": src_raw, "dst": dst, "status": STATUS_OK,
            "code": action.upper(), "error": None, "journal": journal}


def _batch_status(outcomes):
    ok = [o for o in outcomes if o["status"] == STATUS_OK]
    bad = [o for o in outcomes if o["status"] == STATUS_FAILED]
    skipped = [o for o in outcomes if o["status"] == STATUS_SKIPPED]
    if bad and (ok or skipped):
        return OP_PARTIAL
    if bad:
        return OP_FAILED
    return OP_SUCCESS
