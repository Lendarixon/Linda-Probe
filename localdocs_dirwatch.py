"""Event-driven Windows directory watcher; no administrator rights required.

Construction starts the daemon and waits for its first read to be armed. start()
is idempotent; instances cannot be restarted after stop(). Callbacks run on the
worker and must return promptly. stop() discards undelivered events, joins for
1.5 seconds by default, and returns False if a callback or driver still blocks.
No public entry point propagates ordinary exceptions; on_error receives text.
"""

import ctypes
import math
import os
import struct
import threading
import time


_ACTIONS = {1: "added", 2: "removed", 3: "modified",
            4: "renamed_old", 5: "renamed_new"}
_BUFFER_SIZE = 256 * 1024
_FILTER = 0x01 | 0x02 | 0x08 | 0x10 | 0x40
_INFINITE = 0xFFFFFFFF
_ERROR_IO_PENDING = 997
_ERROR_OPERATION_ABORTED = 995
_ERROR_NOTIFY_ENUM_DIR = 1022


def _report(callback, error):
    try:
        if callback is not None:
            callback(str(error))
    except Exception:
        pass


def _path_key(path):
    return os.path.normcase(os.path.normpath(path))


def _parse_buffer(buf, root):
    """Strict internal parser: reject malformed chains rather than lose events."""
    data = memoryview(buf).cast("B")
    if not data:
        return [{"overflow": True}]
    offset = 0
    events = []
    while True:
        if offset + 12 > len(data):
            raise ValueError("Truncated FILE_NOTIFY_INFORMATION header")
        next_offset, action, length = struct.unpack_from("<III", data, offset)
        end = offset + 12 + length
        if length == 0 or length % 2 or end > len(data):
            raise ValueError("Invalid notification filename length")
        if next_offset and (next_offset % 4 or next_offset < 12 + length
                            or offset + next_offset + 12 > len(data)):
            raise ValueError("Invalid notification chain offset")
        name = bytes(data[offset + 12:end]).decode("utf-16-le", "surrogatepass")
        # Kernel names are relative. Reject corrupt names escaping the root.
        parts = name.replace(chr(92), "/").split("/")
        if (os.path.isabs(name) or os.path.splitdrive(name)[0]
                or any(part in ("", ".", "..") for part in parts)
                or chr(0) in name or ":" in name):
            raise ValueError("Invalid relative notification filename")
        if action not in _ACTIONS:
            raise ValueError("Unknown notification action")
        events.append({"path": os.path.abspath(os.path.join(root, *parts)),
                       "action": _ACTIONS[action]})
        if not next_offset:
            return events
        offset += next_offset


def parse_notify_buffer(buf, root, on_error=None):
    """Parse notification bytes; malformed input returns [] and reports an error.

    An empty buffer represents overflow. This helper also works off Windows.
    """
    try:
        return _parse_buffer(buf, os.path.abspath(os.fsdecode(root)))
    except Exception as exc:
        _report(on_error, exc)
        return []


class _Batch:
    """Fixed windows (not trailing-edge debounce); continuous traffic flushes."""

    def __init__(self, interval):
        self.interval = interval
        self.deadline = None
        self.events = {}

    def add(self, events, now):
        for event in events:
            key = ("overflow",) if event.get("overflow") else (
                _path_key(event["path"]), event["action"])
            if self.deadline is None:
                self.deadline = now + self.interval
            self.events.setdefault(key, event)

    def take(self, now):
        if self.deadline is None or now < self.deadline:
            return []
        events = list(self.events.values())
        self.events.clear()
        self.deadline = None
        return events


class _Overlapped(ctypes.Structure):
    # Offset/OffsetHigh occupy the same space as the native pointer union.
    _fields_ = [("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", ctypes.c_uint32),
                ("OffsetHigh", ctypes.c_uint32),
                ("hEvent", ctypes.c_void_p)]


def _winapi():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = ctypes.c_void_p
    dword = ctypes.c_uint32
    boolean = ctypes.c_int
    overlap = ctypes.POINTER(_Overlapped)
    signatures = {
        "CreateFileW": ([ctypes.c_wchar_p, dword, dword, handle,
                         dword, dword, handle], handle),
        "CreateEventW": ([handle, boolean, boolean, ctypes.c_wchar_p], handle),
        "SetEvent": ([handle], boolean),
        "ResetEvent": ([handle], boolean),
        "CloseHandle": ([handle], boolean),
        "ReadDirectoryChangesW": ([handle, handle, dword, boolean, dword,
                                   ctypes.POINTER(dword), overlap, handle], boolean),
        "WaitForMultipleObjects": ([dword, ctypes.POINTER(handle), boolean,
                                    dword], dword),
        "GetOverlappedResult": ([handle, overlap, ctypes.POINTER(dword),
                                 boolean], boolean),
        "CancelIoEx": ([handle, overlap], boolean),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result
    return api


class DirWatcher:
    """Watch a directory tree, coalescing (path, action) in arrival order.

    ignore accepts directory/file prefixes, relative to path or absolute. Matching
    is case-insensitive on Windows and respects directory boundaries. A rename
    is delivered as separate old/new events; either side may be ignored.
    """

    def __init__(self, path, callback, recursive=True, debounce_s=1.0,
                 ignore=None, on_error=None):
        self.on_error = on_error
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._api = None
        self._stop_handle = None
        self._active = False
        self._valid = False
        try:
            self.path = os.path.abspath(os.fsdecode(path))
            self.callback = callback
            self.recursive = bool(recursive)
            self.debounce_s = float(debounce_s)
            if not callable(callback):
                raise ValueError("callback must be callable")
            if not math.isfinite(self.debounce_s) or self.debounce_s < 0:
                raise ValueError("debounce_s must be finite and nonnegative")
            if isinstance(ignore, (str, bytes, os.PathLike)):
                ignore = [ignore]
            self._ignore = tuple(_path_key(os.path.abspath(os.path.join(
                self.path, os.fsdecode(prefix)))) for prefix in (ignore or ()))
            self._valid = True
            self.start()
        except Exception as exc:
            _report(self.on_error, exc)

    def start(self):
        """Return True once monitoring is armed; False on failure or after stop."""
        try:
            if not self._valid or self._stop.is_set():
                return False
            if os.name != "nt":
                _report(self.on_error, "DirWatcher requires Windows")
                return False
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name="LocalDocs-DirWatcher", daemon=True)
                    self._thread.start()
            if not self._ready.wait(5.0):
                _report(self.on_error, "Directory watcher startup timed out")
                self.stop()
                return False
            return self._active and not self._stop.is_set()
        except Exception as exc:
            _report(self.on_error, exc)
            return False

    def stop(self, timeout=1.5):
        """Wake the native wait and join; False means shutdown is still pending.

        Calling from the callback requests shutdown without joining itself.
        A blocking user callback cannot be forcibly interrupted by Python.
        """
        try:
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be finite and nonnegative")
            self._stop.set()
            with self._lock:
                if self._stop_handle is not None:
                    if not self._api.SetEvent(self._stop_handle):
                        _report(self.on_error, ctypes.WinError(ctypes.get_last_error()))
                thread = self._thread
            if thread is None:
                return True
            if thread is threading.current_thread():
                return False
            thread.join(timeout)
            if thread.is_alive():
                _report(self.on_error, "Directory watcher shutdown is still pending")
                return False
            return True
        except Exception as exc:
            _report(self.on_error, exc)
            return False

    def _ignored(self, path):
        key = _path_key(path)
        return any(key == prefix or key.startswith(
            prefix if prefix.endswith(os.sep) else prefix + os.sep)
            for prefix in self._ignore)

    def _run(self):
        api = None
        directory = io_event = stop_event = None
        pending = False
        overlapped = _Overlapped()
        # DWORD-aligned storage, kept alive until canceled I/O has completed.
        buffer = (ctypes.c_uint32 * (_BUFFER_SIZE // 4))()
        transferred = ctypes.c_uint32()
        batch = _Batch(self.debounce_s)
        try:
            api = _winapi()
            directory = api.CreateFileW(self.path, 0x0001, 0x0007, None,
                                        3, 0x02000000 | 0x40000000, None)
            if directory in (None, ctypes.c_void_p(-1).value):
                directory = None
                raise ctypes.WinError(ctypes.get_last_error())
            io_event = api.CreateEventW(None, True, False, None)
            if not io_event:
                raise ctypes.WinError(ctypes.get_last_error())
            stop_event = api.CreateEventW(None, True, False, None)
            if not stop_event:
                raise ctypes.WinError(ctypes.get_last_error())
            with self._lock:
                self._api = api
                self._stop_handle = stop_event
            # Stop is first so it wins when both events are signaled.
            handles = (ctypes.c_void_p * 2)(stop_event, io_event)
            read_size = _BUFFER_SIZE
            while not self._stop.is_set():
                if not pending:
                    if not api.ResetEvent(io_event):
                        raise ctypes.WinError(ctypes.get_last_error())
                    overlapped = _Overlapped()
                    overlapped.hEvent = io_event
                    queued = api.ReadDirectoryChangesW(
                        directory, buffer, read_size, self.recursive, _FILTER,
                        None, ctypes.byref(overlapped), None)
                    error = 0 if queued else ctypes.get_last_error()
                    if not queued and error != _ERROR_IO_PENDING:
                        if error == 87 and read_size > 64 * 1024:
                            # Network redirectors limit notification buffers.
                            read_size = 64 * 1024
                            continue
                        if error == _ERROR_NOTIFY_ENUM_DIR:
                            batch.add([{"overflow": True}], time.monotonic())
                        else:
                            raise ctypes.WinError(error)
                    else:
                        pending = True
                        self._active = True
                        self._ready.set()

                events = batch.take(time.monotonic())
                if events and not self._stop.is_set():
                    try:
                        self.callback(events)
                    except Exception as exc:
                        _report(self.on_error, exc)
                if not pending:
                    # Prevent spinning on a repeatedly reported overflow.
                    self._stop.wait(0.01)
                    continue
                if self._stop.is_set():
                    break
                timeout = _INFINITE
                if batch.deadline is not None:
                    timeout = min(_INFINITE - 1, max(0, math.ceil(
                        (batch.deadline - time.monotonic()) * 1000)))
                result = api.WaitForMultipleObjects(2, handles, False, timeout)
                if result == 0:
                    break
                if result == 258:  # WAIT_TIMEOUT: deliver the pending batch.
                    continue
                if result != 1:
                    raise ctypes.WinError(ctypes.get_last_error())
                completed = api.GetOverlappedResult(
                    directory, ctypes.byref(overlapped), ctypes.byref(transferred), False)
                error = 0 if completed else ctypes.get_last_error()
                pending = False
                if not completed:
                    if error == _ERROR_OPERATION_ABORTED and self._stop.is_set():
                        break
                    if error != _ERROR_NOTIFY_ENUM_DIR:
                        raise ctypes.WinError(error)
                    events = [{"overflow": True}]
                else:
                    try:
                        if transferred.value > read_size:
                            raise ValueError("Notification byte count exceeds buffer")
                        events = _parse_buffer(
                            ctypes.string_at(buffer, transferred.value), self.path)
                    except Exception as exc:
                        _report(self.on_error, exc)
                        events = [{"overflow": True}]
                batch.add([event for event in events if event.get("overflow")
                           or not self._ignored(event["path"])], time.monotonic())
                # The next iteration rearms the read before calling user code.
        except Exception as exc:
            _report(self.on_error, exc)
        finally:
            self._active = False
            self._ready.set()
            if api is not None:
                if pending:
                    if not api.CancelIoEx(directory, ctypes.byref(overlapped)):
                        error = ctypes.get_last_error()
                        if error != 1168:  # ERROR_NOT_FOUND: already completed.
                            _report(self.on_error, ctypes.WinError(error))
                    # CancelIoEx only requests cancellation. Keep storage alive
                    # until completion, including on a slow driver.
                    if not api.GetOverlappedResult(directory, ctypes.byref(overlapped),
                                                   ctypes.byref(transferred), True):
                        error = ctypes.get_last_error()
                        if error != _ERROR_OPERATION_ABORTED:
                            _report(self.on_error, ctypes.WinError(error))
                with self._lock:
                    self._stop_handle = None
                for handle in (directory, io_event, stop_event):
                    if handle and not api.CloseHandle(handle):
                        _report(self.on_error, ctypes.WinError(ctypes.get_last_error()))
