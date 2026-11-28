"""NTFS V2 USN parsing and read-only volume access (stdlib and NumPy only).

Parsing is portable and needs no Windows privileges. Volume operations require
Windows and normally elevation. Public operations report errors to an optional
``on_error(message)`` callback and return empty results, None, or False.
Volume handles belong to the caller; close them with close_volume(). No journal
is created or changed by this module.
"""

import ctypes
import os
import queue
import struct
import threading

import numpy as np


FSCTL_ENUM_USN_DATA = 0x900B3
FSCTL_READ_USN_JOURNAL = 0x900BB
FSCTL_QUERY_USN_JOURNAL = 0x900F4
FSCTL_CREATE_USN_JOURNAL = 0x900E7

USN_REASON_DATA_OVERWRITE = 0x00000001
USN_REASON_DATA_EXTEND = 0x00000002
USN_REASON_DATA_TRUNCATION = 0x00000004
USN_REASON_FILE_CREATE = 0x00000100
USN_REASON_FILE_DELETE = 0x00000200
USN_REASON_RENAME_OLD_NAME = 0x00001000
USN_REASON_RENAME_NEW_NAME = 0x00002000
USN_REASON_CLOSE = 0x80000000

ERROR_HANDLE_EOF = 38
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

RECORD_DTYPE = np.dtype([
    ("frn", "<u8"), ("parent", "<u8"), ("usn", "<i8"),
    ("reason", "<u4"), ("attrs", "<u4"),
    ("name_off", "<u4"), ("name_len", "<u2"),
])
_HEADER_DTYPE = np.dtype({
    "names": ["major", "frn", "parent", "usn", "reason", "attrs",
              "name_bytes", "name_rel"],
    "formats": ["<u2", "<u8", "<u8", "<i8", "<u4", "<u4", "<u2", "<u2"],
    "offsets": [4, 8, 16, 24, 40, 52, 56, 58],
    "itemsize": 60,
})
_U32 = struct.Struct("<I")
_V2 = struct.Struct("<IHHQQqqIIIIHH")
_KERNEL32 = None


def _report(callback, message):
    if callback is not None:
        try:
            callback(str(message))
        except Exception:
            pass


def _empty_records():
    return np.empty(0, dtype=RECORD_DTYPE), np.empty(0, dtype="<u2")


def _empty_volume():
    return {key: np.empty(0, dtype=dtype) for key, dtype in (
        ("frn", "<u8"), ("parent", "<u8"), ("attrs", "<u4"),
        ("name_off", "<u8"), ("name_len", "<u2"), ("blob", "<u2"),
    )}


def _kernel32(on_error=None):
    global _KERNEL32
    try:
        if os.name != "nt":
            _report(on_error, "USN volume operations require Windows")
            return None
        if _KERNEL32 is None:
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = ctypes.c_void_p
            dword = ctypes.c_uint32
            api.CreateFileW.argtypes = [
                ctypes.c_wchar_p, dword, dword, ctypes.c_void_p,
                dword, dword, handle,
            ]
            api.CreateFileW.restype = handle
            api.DeviceIoControl.argtypes = [
                handle, dword, ctypes.c_void_p, dword, ctypes.c_void_p,
                dword, ctypes.POINTER(dword), ctypes.c_void_p,
            ]
            api.DeviceIoControl.restype = ctypes.c_int
            api.CloseHandle.argtypes = [handle]
            api.CloseHandle.restype = ctypes.c_int
            _KERNEL32 = api
        return _KERNEL32
    except Exception as exc:
        _report(on_error, "Loading Windows API: " + str(exc))
        return None


def _valid_handle(handle):
    value = getattr(handle, "value", handle)
    return isinstance(value, int) and value not in (0, -1, INVALID_HANDLE_VALUE)


def _ioctl(api, handle, code, request, output):
    """Return (bytes_written, Win32_error); capture last error immediately."""
    written = ctypes.c_uint32()
    source = ctypes.create_string_buffer(request, len(request)) if request else None
    ok = api.DeviceIoControl(
        handle, code, source, len(request), output, ctypes.sizeof(output),
        ctypes.byref(written), None,
    )
    error = 0 if ok else ctypes.get_last_error()
    if written.value > ctypes.sizeof(output):
        raise ValueError("DeviceIoControl returned an invalid byte count")
    return written.value, error


def _buffer_bytes(buffer_mb):
    size = int(float(buffer_mb) * 1024 * 1024)
    if not 4096 <= size <= 64 * 1024 * 1024:
        raise ValueError("buffer_mb must specify between 4 KiB and 64 MiB")
    return size


def open_volume(letter, on_error=None):
    """Open a drive letter for synchronous reading; return HANDLE or None."""
    try:
        if not isinstance(letter, str):
            raise ValueError("letter must be a drive letter, optionally with colon")
        drive = letter.strip()
        if len(drive) == 2 and drive[1] == ":":
            drive = drive[0]
        if len(drive) != 1 or not ("A" <= drive.upper() <= "Z"):
            raise ValueError("invalid drive letter")
        api = _kernel32(on_error)
        if api is None:
            return None
        path = chr(92) * 2 + "." + chr(92) + drive.upper() + ":"
        handle = api.CreateFileW(
            path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, 0, None,
        )
        error = ctypes.get_last_error() if not _valid_handle(handle) else 0
        if not _valid_handle(handle):
            _report(on_error, "CreateFileW failed: Win32 error " + str(error))
            return None
        return handle
    except Exception as exc:
        _report(on_error, "open_volume: " + str(exc))
        return None


def close_volume(handle, on_error=None):
    """Close an owned volume handle, returning success without raising."""
    try:
        if not _valid_handle(handle):
            return False
        api = _kernel32(on_error)
        if api is None:
            return False
        if not api.CloseHandle(handle):
            error = ctypes.get_last_error()
            _report(on_error, "CloseHandle failed: Win32 error " + str(error))
            return False
        return True
    except Exception as exc:
        _report(on_error, "close_volume: " + str(exc))
        return False


def query_journal(handle, on_error=None):
    """Return journal_id, first_usn, next_usn, max_usn, or None on failure."""
    try:
        if not _valid_handle(handle):
            raise ValueError("invalid volume handle")
        api = _kernel32(on_error)
        if api is None:
            return None
        output = ctypes.create_string_buffer(128)
        count, error = _ioctl(api, handle, FSCTL_QUERY_USN_JOURNAL, b"", output)
        if error:
            raise OSError(error, "FSCTL_QUERY_USN_JOURNAL failed")
        if count < 56:
            raise ValueError("short USN_JOURNAL_DATA_V0 response")
        journal_id, first, next_usn, _lowest, maximum = struct.unpack_from(
            "<Qqqqq", output, 0,
        )
        return {"journal_id": journal_id, "first_usn": first,
                "next_usn": next_usn, "max_usn": maximum}
    except Exception as exc:
        _report(on_error, "query_journal: " + str(exc))
        return None


def parse_usn_buffer(buf, start=8, on_error=None):
    """Return (structured_records, uint16_name_blob), owning their storage.

    ``start`` is a byte offset, normally 8 to skip the ENUM/READ cursor, or 0
    for a bare record stream. It must be even. name_off and name_len count
    UTF-16 code units, including surrogate pairs as two units, not bytes.
    Unknown versions (including V3) are skipped. Invalid V2 names are skipped;
    an invalid record length or incomplete tail ends parsing at the valid prefix.
    The only per-record Python loop discovers record boundaries. Header and
    name validation, field extraction, and ragged name gathering use NumPy.
    """
    try:
        view = memoryview(buf).cast("B")
        size = len(view)
        if not isinstance(start, int) or start < 0 or start > size or start % 2:
            raise ValueError("start must be an even byte offset within the buffer")
        offsets = []
        lengths = []
        position = start
        unpack = _U32.unpack_from
        while position + 8 <= size:
            length = unpack(view, position)[0]
            if length < 8 or length % 8 or length > size - position:
                _report(on_error, "Invalid or truncated USN record at " + str(position))
                break
            offsets.append(position)
            lengths.append(length)
            position += length
        else:
            if position != size:
                _report(on_error, "Truncated USN header at " + str(position))
        if not offsets:
            return _empty_records()

        offsets = np.asarray(offsets, dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        units = np.frombuffer(view[:size - size % 2], dtype="<u2")
        major = units[(offsets + 4) // 2]
        keep = (major == 2) & (lengths >= 60)
        if np.any((major == 2) & (lengths < 60)):
            _report(on_error, "Skipping short V2 headers")
        offsets = offsets[keep]
        lengths = lengths[keep]
        if not offsets.size:
            return _empty_records()

        # A stride of one byte exposes a header at every possible byte offset;
        # fancy indexing gathers only the record boundaries found above.
        headers = np.ndarray(
            shape=(size - 59,), dtype=_HEADER_DTYPE, buffer=view, strides=(1,),
        )[offsets]
        name_bytes = headers["name_bytes"].astype(np.int64)
        relative = headers["name_rel"].astype(np.int64)
        valid = ((name_bytes % 2 == 0) & (relative % 2 == 0)
                 & (relative >= 60) & (relative + name_bytes <= lengths))
        if not np.all(valid):
            _report(on_error, "Skipping invalid V2 name bounds or alignment")
        headers = headers[valid]
        offsets = offsets[valid]
        relative = relative[valid]
        counts = name_bytes[valid] // 2
        if not counts.size:
            return _empty_records()
        ends = np.cumsum(counts, dtype=np.int64)
        total = int(ends[-1])
        if total > np.iinfo(np.uint32).max:
            raise ValueError("name blob exceeds uint32 parser offsets")
        starts = ends - counts
        result = np.empty(counts.size, dtype=RECORD_DTYPE)
        for field in ("frn", "parent", "usn", "reason", "attrs"):
            result[field] = headers[field]
        result["name_off"] = starts
        result["name_len"] = counts
        source_starts = (offsets + relative) // 2
        gather = np.repeat(source_starts - starts, counts)
        gather += np.arange(total, dtype=np.int64)
        blob = units[gather]
        return result, blob
    except Exception as exc:
        _report(on_error, "parse_usn_buffer: " + str(exc))
        return _empty_records()


def _enum_chunks(api, handle, size, max_usn, pipelined):
    """Yield borrowed buffers; a yielded buffer stays owned until resumed.

    Only the reader thread touches the synchronous handle in pipelined mode.
    Three buffers bound memory and provide backpressure without copying chunks.
    """
    def read(output, cursor):
        request = struct.pack("<Qqq", cursor, 0, max_usn)
        count, error = _ioctl(api, handle, FSCTL_ENUM_USN_DATA, request, output)
        if error == ERROR_HANDLE_EOF:
            return None
        if error:
            raise OSError(error, "FSCTL_ENUM_USN_DATA failed")
        if count < 8:
            raise ValueError("short enumeration cursor")
        next_cursor = struct.unpack_from("<Q", output, 0)[0]
        if next_cursor <= cursor:
            raise ValueError("enumeration cursor did not advance")
        return output, count, next_cursor

    if pipelined:
        free = queue.Queue()
        ready = queue.Queue()
        stop = threading.Event()
        for _ in range(3):
            free.put(ctypes.create_string_buffer(size))

        def reader():
            cursor = 0
            try:
                while not stop.is_set():
                    try:
                        output = free.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if stop.is_set():
                        break
                    chunk = read(output, cursor)
                    if chunk is None:
                        break
                    cursor = chunk[2]
                    ready.put(chunk)
            except Exception as exc:
                ready.put(exc)
            finally:
                ready.put(None)

        thread = threading.Thread(target=reader, name="localdocs-usn-enum", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            # No I/O has started: safely retain synchronous compatibility.
            pipelined = False
        else:
            try:
                while True:
                    chunk = ready.get()
                    if chunk is None:
                        return
                    if isinstance(chunk, Exception):
                        raise chunk
                    try:
                        yield chunk
                    finally:
                        free.put(chunk[0])
            finally:
                stop.set()
                # Never return while a reader still owns the caller's handle.
                thread.join()
    if not pipelined:
        output = ctypes.create_string_buffer(size)
        cursor = 0
        while True:
            chunk = read(output, cursor)
            if chunk is None:
                return
            yield chunk
            cursor = chunk[2]


def enum_volume(handle, callback=None, buffer_mb=4, on_error=None, pipelined=False):
    """Enumerate V2 MFT entries into column arrays and one uint16 ``blob``.

    callback(progress_dict), if supplied, receives records, name_units, and
    next_frn after each successful chunk. Callback failures are logged and
    enumeration continues. Unexpected I/O failures return empty arrays rather
    than presenting an incomplete enumeration as a complete index.
    Enumeration is not a snapshot: retain a starting journal cursor and replay
    changes after the build if a consistent live index is needed.
    pipelined=True overlaps parsing with a reader using a three-buffer ring;
    the default retains synchronous operation and the same result schema.
    """
    chunks_iter = None
    try:
        if not _valid_handle(handle):
            raise ValueError("invalid volume handle")
        size = _buffer_bytes(buffer_mb)
        api = _kernel32(on_error)
        if api is None:
            return _empty_volume()
        journal = query_journal(handle, on_error=on_error)
        if journal is None:
            return _empty_volume()
        chunks = []
        blobs = []
        name_units = 0
        record_count = 0
        chunks_iter = _enum_chunks(api, handle, size, journal["max_usn"], pipelined)
        for output, count, next_cursor in chunks_iter:
            records, blob = parse_usn_buffer(
                memoryview(output).cast("B")[:count], 8, on_error=on_error,
            )
            if records.size:
                wide_offsets = records["name_off"].astype(np.uint64)
                wide_offsets += name_units
                chunks.append((records, wide_offsets))
                blobs.append(blob)
                record_count += records.size
                name_units += blob.size
            cursor = next_cursor
            if callback is not None:
                try:
                    callback({"records": int(record_count),
                              "name_units": int(name_units), "next_frn": cursor})
                except Exception as exc:
                    _report(on_error, "Enumeration callback: " + str(exc))
        if not chunks:
            return _empty_volume()
        result = {field: np.concatenate([chunk[0][field] for chunk in chunks])
                  for field in ("frn", "parent", "attrs", "name_len")}
        result["name_off"] = np.concatenate([chunk[1] for chunk in chunks])
        result["blob"] = np.concatenate(blobs)
        return result
    except Exception as exc:
        _report(on_error, "enum_volume: " + str(exc))
        return _empty_volume()
    finally:
        if chunks_iter is not None:
            chunks_iter.close()


class JournalReader:
    """Synchronous journal reader; caller owns and must keep the handle open.

    read() is not thread-safe. last_error is None after successful I/O and a
    diagnostic string after failure; start_usn advances only on successful reads.
    Failures such as journal replacement/wrap require caller resynchronization.
    """

    def __init__(self, handle, journal_id, start_usn, on_error=None, buffer_mb=4):
        self.handle = handle
        self.journal_id = journal_id
        self.start_usn = start_usn
        self.on_error = on_error
        self.last_error = None
        self._output = None
        try:
            if not _valid_handle(handle):
                raise ValueError("invalid volume handle")
            # Validate integer ranges before allocating or performing I/O.
            struct.pack("<qQ", start_usn, journal_id)
            self._output = ctypes.create_string_buffer(_buffer_bytes(buffer_mb))
        except Exception as exc:
            self.last_error = str(exc)
            _report(on_error, "JournalReader: " + self.last_error)

    def read(self, timeout_ms=1000):
        """Return (records, name_blob), using ReturnOnlyOnClose=0 and wait=1.

        Windows' V0 Timeout is an integer number of seconds, rounded up here
        with a minimum of one second. Negative timeouts are rejected. This is
        the kernel's retry interval, NOT a hard wall-clock deadline: the OS can
        repeat the wait until matching data arrives or the I/O is cancelled.
        See Microsoft's READ_USN_JOURNAL_DATA_V0 Timeout documentation:
        https://learn.microsoft.com/en-us/windows/win32/api/winioctl/ns-winioctl-read_usn_journal_data_v0
        """
        try:
            if self._output is None:
                raise ValueError(self.last_error or "reader is not initialized")
            milliseconds = float(timeout_ms)
            if not 0 <= milliseconds <= 0xFFFFFFFF:
                raise ValueError("timeout_ms must be finite and nonnegative")
            seconds = max(1, int(milliseconds / 1000))
            if seconds * 1000 < milliseconds:
                seconds += 1
            api = _kernel32(self.on_error)
            if api is None:
                raise OSError("Windows API unavailable")
            request = struct.pack(
                "<qIIQQQ", self.start_usn, 0xFFFFFFFF, 0,
                0, 0, self.journal_id,  # BytesToWaitFor=0: return at once; 1 would block forever on a quiet volume
            )
            count, error = _ioctl(
                api, self.handle, FSCTL_READ_USN_JOURNAL, request, self._output,
            )
            if error:
                raise OSError(error, "FSCTL_READ_USN_JOURNAL failed")
            if count < 8:
                raise ValueError("short journal cursor")
            next_usn = struct.unpack_from("<q", self._output, 0)[0]
            if next_usn < self.start_usn:
                raise ValueError("journal cursor moved backwards")
            parsed = parse_usn_buffer(
                memoryview(self._output).cast("B")[:count], 8,
                on_error=self.on_error,
            )
            self.start_usn = next_usn
            self.last_error = None
            return parsed
        except Exception as exc:
            self.last_error = str(exc)
            _report(self.on_error, "JournalReader.read: " + self.last_error)
            return _empty_records()


def is_admin(on_error=None):
    """Return whether this Windows process is elevated; False elsewhere."""
    try:
        if os.name != "nt":
            return False
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        shell.IsUserAnAdmin.argtypes = []
        shell.IsUserAnAdmin.restype = ctypes.c_int
        return bool(shell.IsUserAnAdmin())
    except Exception as exc:
        _report(on_error, "is_admin: " + str(exc))
        return False


def build_test_buffer(records, next_start=0, on_error=None):
    """Build an 8-byte cursor followed by aligned synthetic V2 records.

    records is an iterable of mappings with name and optional frn, parent, usn,
    reason, attrs, timestamp, source, security_id. Defaults are empty name,
    sequential frn starting at 1, and zero for other fields. next_start is the
    unsigned ENUM/READ cursor. Empty input returns the cursor alone; invalid
    input returns b'' and reports an error. No filesystem access is performed.
    """
    try:
        output = bytearray(struct.pack("<Q", next_start))
        for index, record in enumerate(records):
            name = record.get("name", "").encode("utf-16-le", errors="surrogatepass")
            if len(name) > 65534:
                raise ValueError("name is too long for FileNameLength")
            length = (60 + len(name) + 7) & ~7
            entry = bytearray(length)
            _V2.pack_into(
                entry, 0, length, 2, 0, record.get("frn", index + 1),
                record.get("parent", 0), record.get("usn", 0),
                record.get("timestamp", 0), record.get("reason", 0),
                record.get("source", 0), record.get("security_id", 0),
                record.get("attrs", 0), len(name), 60,
            )
            entry[60:60 + len(name)] = name
            output.extend(entry)
        return bytes(output)
    except Exception as exc:
        _report(on_error, "build_test_buffer: " + str(exc))
        return b""
