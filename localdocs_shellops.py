"""LocalDocs shell recycle operations (P5 sandbox).

Official Microsoft docs used (not anonymous snippets):
- IFileOperation (shobjidl_core.h):
  https://learn.microsoft.com/en-us/windows/win32/api/shobjidl_core/nn-shobjidl_core-ifileoperation
  Replaces SHFileOperation as of Windows Vista.
- IFileOperation::SetOperationFlags:
  https://learn.microsoft.com/en-us/windows/win32/api/shobjidl_core/nf-shobjidl_core-ifileoperation-setoperationflags
  FOF_ALLOWUNDO (0x0040) = "Preserve undo information, if possible."
  FOFX_RECYCLEONDELETE (0x00080000, Win8+) = "When a file is deleted, send it
  to the Recycle Bin rather than permanently deleting it."
  FOF_WANTNUKEWARNING (0x4000) warns if an item is destroyed rather
  than recycled. FOF_NOERRORUI+FOFX_EARLYFAILURE stops the whole set on error.
- SHFileOperationA:
  https://learn.microsoft.com/en-us/windows/win32/api/shellapi/nf-shellapi-shfileoperationa
  "This function has been replaced in Windows Vista by IFileOperation."
  "SHFileOperation can return 0 for success if the user cancels" -> must
  check fAnyOperationsAborted as well as the return value.
- SHFILEOPSTRUCTW:
  https://learn.microsoft.com/en-us/windows/win32/api/shellapi/ns-shellapi-shfileopstructw
  struct { HWND hwnd; UINT wFunc; PCZZWSTR pFrom; PCZZWSTR pTo;
  FILEOP_FLAGS fFlags; BOOL fAnyOperationsAborted; LPVOID hNameMappings;
  PCWSTR lpszProgressTitle; }
  HWND/LPVOID/PCWSTR are 8-byte pointers on x64. UINT is 4 bytes.
  FILEOP_FLAGS is a WORD (2 bytes). BOOL is a 4-byte Win32 BOOL (not C bool).
  Correct x64 layout: hwnd@0 wFunc@8 pFrom@16 pTo@24 fFlags@32 (WORD)
  + 2 bytes padding, fAnyOperationsAborted@36 (4-byte BOOL),
  hNameMappings@40 lpszProgressTitle@48, sizeof=56.

This module is the ONLY place that talks to the real Windows Shell for
recycle. It never permanently deletes: any unsupported volume, disabled
bin, COM/apartment problem, nonzero HRESULT, or abort is reported as a
failure and the source is left in place. No os.remove/rmtree fallback.
All confirmation/progress UI is suppressed via flags, so no native
dialogs open.
"""

from __future__ import annotations

import ctypes
import os

# ---------------------------------------------------------------------------
# Operation flags (Shellapi.h / Shobjidl.h, see MS Learn URLs above)
# ---------------------------------------------------------------------------

FO_DELETE = 0x0003

FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040
FOF_NOERRORUI = 0x0400
FOF_WANTNUKEWARNING = 0x4000

FOFX_RECYCLEONDELETE = 0x00080000
FOFX_EARLYFAILURE = 0x00100000

# Forced-recycle semantics: ALLOWUNDO (undo-able) + RECYCLEONDELETE (Win8+:
# send to bin rather than permanently deleting) + WANTNUKEWARNING so a nuke
# would warn/fail instead of silently destroying + silent/no-UI + early fail.
RECYCLE_OPERATION_FLAGS = (
    FOF_ALLOWUNDO
    | FOFX_RECYCLEONDELETE
    | FOF_WANTNUKEWARNING
    | FOF_NOCONFIRMATION
    | FOF_NOERRORUI
    | FOF_SILENT
    | FOFX_EARLYFAILURE
)

# Legacy SHFileOperationW flags (fixed ABI, abort reliably readable).
# Includes WANTNUKEWARNING so a permanent-destroy path warns instead of
# silently nuking; still NOT claimed as guaranteed recycle (see module docs).
_LEGACY_RECYCLE_FLAGS = (
    FOF_ALLOWUNDO
    | FOF_NOCONFIRMATION
    | FOF_NOERRORUI
    | FOF_SILENT
    | FOF_WANTNUKEWARNING
)

DRIVE_FIXED = 3

_COINT_APARTMENTTHREADED = 0x2
_COINT_DISABLE_OLE1DDE = 0x4
_S_OK = 0
_RPC_E_CHANGED_MODE = -2147417850  # 0x80010106 signed

_CLSCTX_ALL = 0x17


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, s):
        import uuid as _uuid

        u = _uuid.UUID(s)
        b = u.bytes_le
        g = cls()
        g.Data1 = int.from_bytes(b[0:4], "little")
        g.Data2 = int.from_bytes(b[4:6], "little")
        g.Data3 = int.from_bytes(b[6:8], "little")
        for i in range(8):
            g.Data4[i] = b[8 + i]
        return g


_CLSID_FileOperation = _GUID.from_string("3ad05575-8857-4850-9277-11b85bdb8e09")
_IID_IFileOperation = _GUID.from_string("947aab5f-0a5c-4c13-b4d6-4bf7836fc9f8")
_IID_IShellItem = _GUID.from_string("43826d1e-e718-42ee-bc55-a1e261c37bfe")


class SHFILEOPSTRUCTW(ctypes.Structure):
    """Fixed ABI per MS Learn SHFILEOPSTRUCTW docs.

    fFlags is WORD (c_ushort, 2 bytes at offset 32); fAnyOperationsAborted
    is Win32 BOOL (c_int, 4 bytes at offset 36 after 2 padding bytes).
    Using c_bool (1 byte) misplaces the abort flag at offset 34 and the
    shell-written abort is never read.
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


class SHQUERYRBINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("i64Size", ctypes.c_int64),
        ("i64NumItems", ctypes.c_int64),
    ]


def struct_offsets():
    """Return (offsets dict, sizeof) for ABI tests."""
    return (
        {
            "hwnd": SHFILEOPSTRUCTW.hwnd.offset,
            "wFunc": SHFILEOPSTRUCTW.wFunc.offset,
            "pFrom": SHFILEOPSTRUCTW.pFrom.offset,
            "pTo": SHFILEOPSTRUCTW.pTo.offset,
            "fFlags": SHFILEOPSTRUCTW.fFlags.offset,
            "fAnyOperationsAborted": SHFILEOPSTRUCTW.fAnyOperationsAborted.offset,
            "hNameMappings": SHFILEOPSTRUCTW.hNameMappings.offset,
            "lpszProgressTitle": SHFILEOPSTRUCTW.lpszProgressTitle.offset,
        },
        ctypes.sizeof(SHFILEOPSTRUCTW),
    )


def _windll(name):
    try:
        return getattr(ctypes.windll, name)
    except Exception:
        return None


def get_drive_type(path):
    """GetDriveTypeW with correct ABI. Returns int or None."""
    k32 = _windll("kernel32")
    if k32 is None:
        return None
    try:
        fn = k32.GetDriveTypeW
        fn.argtypes = [ctypes.c_wchar_p]
        fn.restype = ctypes.c_uint
        root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
        return int(fn(root))
    except Exception:
        return None


def query_recycle_bin(root_path):
    """SHQueryRecycleBinW with correct ABI.

    Returns (ok, size_bytes, num_items). ok=False means the bin state
    could not be confirmed for that volume.
    """
    sh = _windll("shell32")
    if sh is None:
        return (False, 0, 0)
    try:
        fn = sh.SHQueryRecycleBinW
        fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(SHQUERYRBINFO)]
        fn.restype = ctypes.c_long  # HRESULT
        info = SHQUERYRBINFO()
        info.cbSize = ctypes.sizeof(SHQUERYRBINFO)
        hr = fn(root_path, ctypes.byref(info))
        if int(hr) != _S_OK:
            return (False, 0, 0)
        return (True, int(info.i64Size), int(info.i64NumItems))
    except Exception:
        return (False, 0, 0)


def _global_nuke_policy():
    """Best-effort check for a global no-recycle policy (fail-open).

    Returns True if a policy explicitly disables the bin. Registry read
    failures return False; the IFileOperation forced-recycle call remains
    the authoritative guarantee (it fails instead of nuking).
    """
    try:
        import winreg

        for hive, sub, name in (
            (
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Policies\Explorer",
                "NoRecycleFiles",
            ),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"Software\Microsoft\Windows\CurrentVersion\Policies\Explorer",
                "NoRecycleFiles",
            ),
        ):
            try:
                with winreg.OpenKey(hive, sub) as key:
                    val, _ = winreg.QueryValueEx(key, name)
                    if int(val) != 0:
                        return True
            except OSError:
                continue
    except Exception:
        pass
    return False


def volume_supports_recycle(path):
    """Decide whether a recycle may be attempted. Never deletes.

    Returns (supported, reason). Only DRIVE_FIXED volumes with a
    confirmable bin and no global nuke policy are supported. Anything
    else (removable=2, remote=4, cdrom=5, ramdisk=6, unknown=0, no-root=1,
    unqueryable bin) is UNSUPPORTED and the caller must preserve source.
    """
    dtype = get_drive_type(path)
    if dtype is None:
        return (False, "cannot determine drive type; refusing recycle")
    if dtype != DRIVE_FIXED:
        return (False, "volume drive type %s is not recycle-capable" % dtype)
    if _global_nuke_policy():
        return (False, "global NoRecycleFiles policy is set")
    root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
    ok, _size, _n = query_recycle_bin(root)
    if not ok:
        return (False, "Recycle Bin state for %s could not be confirmed" % root)
    return (True, "ok")


def _ensure_com_apartment():
    """CoInitializeEx(COINIT_APARTMENTTHREADED) with correct ABI.

    Returns (ok, we_initialized, error).
    IFileOperation requires an STA; if the thread is already MTA
    (RPC_E_CHANGED_MODE) we report failure instead of operating in the
    wrong apartment.
    """
    ole = _windll("ole32")
    if ole is None:
        return (False, False, "ole32 unavailable")
    try:
        fn = ole.CoInitializeEx
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        fn.restype = ctypes.c_long
        hr = int(fn(None, _COINT_APARTMENTTHREADED | _COINT_DISABLE_OLE1DDE))
        if hr in (_S_OK, 1):  # S_OK or S_FALSE (already init same model)
            return (True, hr == _S_OK, None)
        if hr == _RPC_E_CHANGED_MODE:
            return (False, False, "COM thread already in wrong apartment (MTA)")
        return (False, False, "CoInitializeEx HRESULT 0x%08X" % (hr & 0xFFFFFFFF))
    except Exception as exc:
        return (False, False, "CoInitializeEx error: %r" % exc)


def _vtbl_func(ppv, index, restype, argtypes):
    WINFUNCTYPE = ctypes.WINFUNCTYPE
    vtbl_addr = ctypes.cast(ppv, ctypes.POINTER(ctypes.c_void_p))[0]
    func_addr = ctypes.cast(vtbl_addr, ctypes.POINTER(ctypes.c_void_p))[index]
    proto = WINFUNCTYPE(restype, *argtypes)
    return proto(func_addr)


def _release(ppv):
    try:
        fn = _vtbl_func(ppv, 2, ctypes.c_ulong, [ctypes.c_void_p])
        fn(ppv)
    except Exception:
        pass


def _recycle_via_ifileoperation(path):
    """True forced-recycle path. Returns dict, never raises, never deletes
    permanently. Caller must still verify the source is gone."""
    ole = _windll("ole32")
    sh = _windll("shell32")
    if ole is None or sh is None:
        return {"code": 1, "aborted": False, "method": "IFileOperation",
                "error": "ole32/shell32 unavailable", "hresult": None}
    ok, we_init, err = _ensure_com_apartment()
    if not ok:
        return {"code": 1, "aborted": False, "method": "IFileOperation",
                "error": err or "COM apartment unavailable", "hresult": None}
    ppv = ctypes.c_void_p(None)
    try:
        cci = ole.CoCreateInstance
        cci.argtypes = [ctypes.POINTER(_GUID), ctypes.c_void_p,
                        ctypes.c_uint32, ctypes.POINTER(_GUID),
                        ctypes.POINTER(ctypes.c_void_p)]
        cci.restype = ctypes.c_long
        hr = int(cci(ctypes.byref(_CLSID_FileOperation), None, _CLSCTX_ALL,
                     ctypes.byref(_IID_IFileOperation),
                     ctypes.byref(ppv)))
        # CLSCTX_ALL: in-proc + local/remote server; COM picks FileOperation.
        if hr != _S_OK or not ppv.value:
            return {"code": 1, "aborted": False, "method": "IFileOperation",
                    "error": "CoCreateInstance HRESULT 0x%08X" % (hr & 0xFFFFFFFF),
                    "hresult": hr}
        op = ppv.value
        # SetOperationFlags(flags)
        try:
            fn = _vtbl_func(op, 5, ctypes.c_long, [ctypes.c_void_p, ctypes.c_uint32])
            hr = int(fn(op, ctypes.c_uint32(RECYCLE_OPERATION_FLAGS)))
            if hr != _S_OK:
                return {"code": 1, "aborted": False, "method": "IFileOperation",
                        "error": "SetOperationFlags HRESULT 0x%08X" % (hr & 0xFFFFFFFF),
                        "hresult": hr}
        except Exception as exc:
            return {"code": 1, "aborted": False, "method": "IFileOperation",
                    "error": "SetOperationFlags call failed: %r" % exc, "hresult": None}
        # SHCreateItemFromParsingName(path, None, IID_IShellItem, &item)
        item = ctypes.c_void_p(None)
        try:
            fn2 = sh.SHCreateItemFromParsingName
            fn2.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p,
                            ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
            fn2.restype = ctypes.c_long
            hr = int(fn2(path, None, ctypes.byref(_IID_IShellItem),
                         ctypes.byref(item)))
            if hr != _S_OK or not item.value:
                return {"code": 1, "aborted": False, "method": "IFileOperation",
                        "error": "SHCreateItemFromParsingName HRESULT 0x%08X" % (hr & 0xFFFFFFFF),
                        "hresult": hr}
        except Exception as exc:
            return {"code": 1, "aborted": False, "method": "IFileOperation",
                    "error": "SHCreateItemFromParsingName failed: %r" % exc,
                    "hresult": None}
        try:
            # DeleteItem(this, item, progressSink=None): three arguments incl. this.
            try:
                fn3 = _vtbl_func(op, 18, ctypes.c_long,
                                 [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
                hr = int(fn3(op, item.value, None))
                if hr != _S_OK:
                    return {"code": 1, "aborted": False,
                            "method": "IFileOperation",
                            "error": "DeleteItem HRESULT 0x%08X" % (hr & 0xFFFFFFFF),
                            "hresult": hr}
            except Exception as exc:
                return {"code": 1, "aborted": False, "method": "IFileOperation",
                        "error": "DeleteItem call failed: %r" % exc, "hresult": None}
            # PerformOperations()
            try:
                fn4 = _vtbl_func(op, 21, ctypes.c_long, [ctypes.c_void_p])
                hr = int(fn4(op))
            except Exception as exc:
                return {"code": 1, "aborted": False, "method": "IFileOperation",
                        "error": "PerformOperations call failed: %r" % exc,
                        "hresult": None}
            # GetAnyOperationsAborted(&aborted) even on failure
            aborted = False
            try:
                out = ctypes.c_int(0)
                fn5 = _vtbl_func(op, 22, ctypes.c_long,
                                 [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
                hr2 = int(fn5(op, ctypes.byref(out)))
                if hr2 == _S_OK:
                    aborted = bool(out.value)
                else:
                    aborted = False
            except Exception:
                aborted = False
            if aborted:
                return {"code": 0, "aborted": True, "method": "IFileOperation",
                        "error": None, "hresult": hr}
            if hr != _S_OK:
                return {"code": 1, "aborted": False, "method": "IFileOperation",
                        "error": "PerformOperations HRESULT 0x%08X" % (hr & 0xFFFFFFFF),
                        "hresult": hr}
            return {"code": 0, "aborted": False, "method": "IFileOperation",
                    "error": None, "hresult": hr}
        finally:
            _release(item.value)
    finally:
        if ppv.value:
            _release(ppv.value)
        if we_init:
            try:
                ole.CoUninitialize()
            except Exception:
                pass


def recycle_legacy_shfileop(path):
    """Legacy SHFileOperationW path with FIXED ABI (for diagnostics only).

    NOT claimed as guaranteed recycle. Includes FOF_WANTNUKEWARNING and
    requires the caller to check both the return code and the 4-byte
    fAnyOperationsAborted flag at offset 36.
    """
    sh = _windll("shell32")
    if sh is None:
        return {"code": 1, "aborted": False, "method": "SHFileOperationW",
                "error": "shell32 unavailable"}
    try:
        from_buf = ctypes.create_unicode_buffer(path + "\0\0")
        op = SHFILEOPSTRUCTW()
        op.hwnd = None
        op.wFunc = FO_DELETE
        op.pFrom = ctypes.cast(from_buf, ctypes.c_wchar_p).value
        op.pTo = None
        op.fFlags = _LEGACY_RECYCLE_FLAGS
        op.fAnyOperationsAborted = 0
        op.hNameMappings = None
        op.lpszProgressTitle = None
        fn = sh.SHFileOperationW
        fn.argtypes = [ctypes.POINTER(SHFILEOPSTRUCTW)]
        fn.restype = ctypes.c_int
        op._keepalive = from_buf  # noqa: SLF001
        code = int(fn(ctypes.byref(op)))
        return {"code": code, "aborted": bool(op.fAnyOperationsAborted),
                "method": "SHFileOperationW", "error": None}
    except Exception as exc:
        return {"code": 1, "aborted": False, "method": "SHFileOperationW",
                "error": "%r" % exc}


def recycle_to_bin(path):
    """Recycle-only entry point. Never permanently deletes.

    1. Pre-check volume_supports_recycle (FIXED + confirmable bin only).
    2. Perform IFileOperation forced recycle (FOFX_RECYCLEONDELETE).
    Unsupported/COM/HRESULT/abort => failure dict; source left in place.
    """
    if not isinstance(path, str) or not path:
        return {"code": 1, "aborted": False, "method": "IFileOperation",
                "error": "invalid path"}
    supported, reason = volume_supports_recycle(path)
    if not supported:
        return {"code": 1, "aborted": False, "method": "IFileOperation",
                "error": "RECYCLE_UNSUPPORTED: " + reason}
    return _recycle_via_ifileoperation(path)
