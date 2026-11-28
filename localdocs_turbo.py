"""Owned native name worker with serialized transport and atomic snapshots."""
import atexit
import os
import socket
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace


class TurboError(RuntimeError):
    pass


class Unsupported(TurboError):
    pass


def _own_process(process):
    """A Windows job kills the worker even when Python terminates abruptly."""
    import ctypes as c
    from ctypes import wintypes as w

    class Basic(c.Structure):
        _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64),
                    ('flags', w.DWORD), ('minimum', c.c_size_t), ('maximum', c.c_size_t),
                    ('active', w.DWORD), ('affinity', c.c_size_t),
                    ('priority', w.DWORD), ('scheduling', w.DWORD)]

    class IO(c.Structure):
        _fields_ = [(name, c.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

    class Extended(c.Structure):
        _fields_ = [('basic', Basic), ('io', IO), ('process_memory', c.c_size_t),
                    ('job_memory', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]

    kernel = c.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise TurboError('CreateJobObject failed')
    info = Extended()
    info.basic.flags = 0x2000
    if (not kernel.SetInformationJobObject(job, 9, c.byref(info), c.sizeof(info)) or
            not kernel.AssignProcessToJobObject(job, w.HANDLE(int(process._handle)))):
        kernel.CloseHandle(job)
        raise TurboError('cannot own worker process')
    return kernel, job


class Turbo:
    def __init__(self, cache_dir=None, timeout=10):
        self.cache_dir, self.timeout = cache_dir, timeout
        self.process = None
        self._socket = self._stream = self._job = None
        self._lock = threading.RLock()
        self._export_lock = threading.Lock()
        self._closed = False
        self._directory = self.snapshot = self.index = None
        self.n_exported = self.generation = self.revision = self.layout = 0
        self.has_expansion = False
        atexit.register(self.stop)

    def _request(self, line, search=False):
        try:
            with self._lock:
                if self.process is None or self.process.poll() is not None or self._stream is None:
                    raise TurboError('native worker is not running')
                if '\n' in line or '\r' in line:
                    raise Unsupported('line breaks in request')
                self._stream.write((line + '\n').encode('utf-8'))
                self._stream.flush()
                first = self._stream.readline()
                if not first:
                    raise TurboError('native worker disconnected')
                fields = first.decode('utf-8').rstrip('\r\n').split('\t')
                if fields[0] == 'ERR':
                    if len(fields) > 1 and fields[1] == 'unsupported':
                        raise Unsupported('\t'.join(fields[1:]))
                    raise TurboError('\t'.join(fields[1:]))
                if not search:
                    return fields
                if len(fields) != 4 or fields[0] != 'OK':
                    raise TurboError('invalid SEARCH response')
                total, truncated, generation = map(int, fields[1:])
                if total < 0 or truncated not in (0, 1) or generation < 1:
                    raise TurboError('invalid SEARCH header')
                rows, unknown, seen = [], [], set()
                while True:
                    raw = self._stream.readline()
                    if not raw:
                        raise TurboError('incomplete SEARCH response')
                    value = raw.decode('utf-8').strip()
                    if value == 'END':
                        break
                    row = int(value[:-1] if value.endswith('?') else value)
                    if not 0 <= row < self.n_exported:
                        raise TurboError('invalid snapshot row')
                    if row in seen or len(rows) >= total:
                        raise TurboError('invalid SEARCH row count')
                    seen.add(row)
                    rows.append(row)
                    if value.endswith('?'):
                        unknown.append(row)
                if not truncated and total != len(rows):
                    raise TurboError('incomplete SEARCH rows')
                return SimpleNamespace(rows=rows, unknown=unknown, total=total,
                                       truncated=bool(truncated), gen=generation)
        except Unsupported:
            raise
        except Exception as exc:
            raise TurboError(str(exc)) from exc

    def start(self):
        with self._lock:
            if self._closed:
                raise TurboError('worker was stopped')
            if self.process is not None and self.process.poll() is None:
                return True
            # Opting into search must never compile or install anything. A
            # packaged worker is sufficient even on machines without csc.
            exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native', 'ldname.exe')
            if not os.path.isfile(exe):
                raise TurboError('native/ldname.exe unavailable')
            try:
                if self._directory is None:
                    if self.cache_dir:
                        os.makedirs(self.cache_dir, exist_ok=True)
                    self._directory = tempfile.TemporaryDirectory(prefix='localdocs-turbo-', dir=self.cache_dir)
                    self.snapshot = os.path.join(self._directory.name, 'names.bin')
                if not os.path.exists(self.snapshot):
                    from localdocs_fastindex import NameIndex
                    from native.export_names import write
                    write(NameIndex(), self.snapshot)
                with socket.socket() as listener:
                    listener.bind(('127.0.0.1', 0))
                    port = listener.getsockname()[1]
                self.process = subprocess.Popen(
                    [exe, 'serve', self.snapshot, str(port)], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self._job = _own_process(self.process)
                deadline = time.monotonic() + self.timeout
                while True:
                    try:
                        self._socket = socket.create_connection(('127.0.0.1', port), timeout=.25)
                        break
                    except OSError:
                        if self.process.poll() is not None or time.monotonic() >= deadline:
                            raise TurboError('native startup timed out')
                        time.sleep(.025)
                self._socket.settimeout(self.timeout)
                self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._stream = self._socket.makefile('rwb')
                if self._request('PING') != ['PONG']:
                    raise TurboError('invalid PING response')
                return True
            except Exception as exc:
                self.stop()
                raise TurboError(str(exc)) from exc

    def export_and_load(self, index):
        from native.export_names import write
        from localdocs_fastindex import NameIndex
        with self._export_lock:
            self.start()
            with index._lock:
                snapshot = NameIndex.from_arrays(
                    *(getattr(index, name).copy() for name in
                      ('blob_low', 'blob_orig', 'off', 'ln', 'parent', 'flags')),
                    roots=list(index._roots), root_of=index._root_of.copy(),
                    size=index.size.copy(), mtime=index.mtime.copy(), depth=index._depth.copy())
                revision, layout = index._turbo_revision, index._turbo_layout
            # Renames append names to the Python arena and make offsets
            # non-monotone. Native FindRow requires a packed blob in row order.
            if not snapshot._contiguous_names():
                import numpy as np
                offsets = np.cumsum(snapshot.ln, dtype=np.uint64) - snapshot.ln
                packed = np.empty(int(snapshot.ln.sum(dtype=np.uint64)), dtype=np.uint16)
                for lo in range(0, len(snapshot), 16384):
                    hi = min(len(snapshot), lo + 16384)
                    lengths = snapshot.ln[lo:hi].astype(np.int64)
                    which = np.repeat(np.arange(hi - lo), lengths)
                    starts = np.cumsum(lengths) - lengths
                    columns = np.arange(int(lengths.sum())) - starts[which]
                    at = int(offsets[lo])
                    packed[at:at + len(which)] = snapshot.blob_low[
                        snapshot.off[lo:hi][which].astype(np.int64) + columns]
                snapshot.blob_low = packed
                snapshot.off = offsets
            fd, temporary = tempfile.mkstemp(dir=self._directory.name, suffix='.bin')
            os.close(fd)
            # The current snapshot may still be memory-mapped by Windows.
            # LOAD a new file, never replace the file held by the worker.
            target = temporary + '.loaded'
            try:
                write(snapshot, temporary)
                with self._lock:
                    if self._closed:
                        raise TurboError('worker stopped during export')
                    os.replace(temporary, target)
                    response = self._request('LOAD\t' + target)
                    if len(response) != 3 or response[0] != 'OK' or int(response[1]) != len(snapshot):
                        raise TurboError('invalid LOAD response')
                    with index._lock:
                        self.snapshot = target
                        self.index, self.n_exported = index, len(snapshot)
                        self.generation = int(response[2])
                        self.has_expansion = bool((snapshot.blob_orig == 0x0130).any())
                        self.revision, self.layout = revision, layout
                        if index._turbo_layout == layout:
                            index._turbo_touched = {k: v for k, v in index._turbo_touched.items() if v > revision}
                return self.generation
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def search(self, query, limit=50):
        return self._request('SEARCH\t%d\t%s' % (int(limit), str(query).replace('\t', ' ')), search=True)

    def stop(self):
        with self._lock:
            self._closed = True
            atexit.unregister(self.stop)
            for obj in (self._stream, self._socket):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
            self._stream = self._socket = None
            if self.process is not None:
                try:
                    if self.process.poll() is None:
                        self.process.kill()
                    self.process.wait(timeout=5)
                except Exception:
                    pass
            if self._job is not None:
                kernel, handle = self._job
                kernel.CloseHandle(handle)
                self._job = None
            if self._directory is not None:
                try:
                    self._directory.cleanup()
                except OSError:
                    pass
                self._directory = None
