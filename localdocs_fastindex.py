"""LocalDocs fast file-name index (spec work/spec_fastindex.md, Module 3).

Columnar numpy name index shared by the scandir walker and a USN source.
Conventions: stdlib plus numpy only. Public methods never raise; they
return empty values and report through an optional on_error callback.
Windows paths are built with os.path and chr(92), never with
backslash-heavy literals.
"""

import collections
import json
import tempfile
import ntpath
import os
import queue
import threading
import time
import types
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from localdocs_arena import _NameArena, _DefaultColumn, _MappedBytes  # data classes live in their own module (engine file stays under the 256 KiB agent limit)

__all__ = ["NameIndex", "FastQuickIndex"]

_BS = chr(92)
_SL = chr(47)

FLAG_DIR = 1
FLAG_HIDDEN = 2
FLAG_JUNK = 4
FLAG_TOMB = 8

BLOCK = 64
BLOOM_BITS = 16384
BLOOM_U64 = BLOOM_BITS // 64
BLOOM_VERSION = 2

_TOP_STAT = 200
_LRU_SIZE = 128


def _parse_search_query(query):
    """Keep plain name search usable when the optional filter module is absent.

    Never reinterpret operator/regex/wildcard syntax as a plain name. Installed
    filter support remains authoritative for the complete query language.
    """
    try:
        from localdocs_filters import parse
    except ModuleNotFoundError as exc:
        if exc.name != "localdocs_filters":
            raise
        clean = str(query or "").strip()
        if any(c in clean for c in (":", "*", "?", "[", "]", "^", "$",
                                    "|", "(", ")", "{", "}", _BS, _SL)):
            return "", None
        return clean, None
    return parse(query)


def _filter_results(*args, **kwargs):
    try:
        from localdocs_filters import FilterResults
    except ModuleNotFoundError as exc:
        if exc.name != "localdocs_filters":
            raise

        class FilterResults(list):
            def __init__(self, rows=(), truncated=False, invalid_regex=False):
                super().__init__(rows)
                self.truncated = truncated
                self.invalid_regex = invalid_regex
    return FilterResults(*args, **kwargs)

_SKIP_NAMES = frozenset({
    ".git",
    "node_modules",
    "appdata",
    "windows",
    "program files",
    "program files (x86)",
    "$recycle.bin",
    "__pycache__", "venv", ".venv", "site-packages", "lib", "dist", "build", "target",
    "indexes", "logs", "cache", ".cache", "vendor", "_vendor", "node_modules.bak",
    "models", "models_onnx", "tessdata", "licenses", "world_backups", "workspaces",
    "portable moved with spaces", "$windows.~bt", "system volume information",
})

_SKIP_EXT = frozenset({
    ".pyc", ".pyo", ".map", ".dll", ".pdb", ".obj", ".o", ".so", ".lib", ".class",
    ".tmp", ".lock", ".lnk.tmp", ".idx", ".pack", ".faiss", ".npy", ".onnx", ".safetensors",
})

_DOC_EXT = frozenset({
    ".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".odt", ".ods", ".odp",
    ".rtf", ".csv", ".epub", ".doc", ".xls", ".ppt",
})

_PENALTY_EXT = frozenset({
    ".dll", ".sys", ".obj", ".pdb", ".so", ".class", ".cache", ".tmp", ".log",
})

_WORD_SEPS = frozenset({" ", "_", "-", ".", "(", ")", "[", "]", "{", "}", ","})


def _build_lower_table():
    out = np.arange(65536, dtype=np.uint16)
    for i in range(65536):
        try:
            c = chr(i)
        except Exception:
            continue
        try:
            low = c.lower()
        except Exception:
            continue
        if low:
            try:
                out[i] = ord(low[0]) & 0xFFFF
            except Exception:
                pass
    return out


_LOWER = _build_lower_table()


def _norm(path):
    try:
        p = os.path.abspath(os.path.expanduser(str(path)))
    except Exception:
        return ""
    try:
        return os.path.normcase(os.path.normpath(p))
    except Exception:
        return ""


def _prefix_match(path, prefix):
    try:
        a = os.path.normcase(os.path.normpath(str(path)))
        b = os.path.normcase(os.path.normpath(str(prefix)))
    except Exception:
        return False
    if a == b:
        return True
    try:
        if b.endswith((os.sep, _BS, _SL)):
            return a.startswith(b)
        return a.startswith(b + os.sep) or a.startswith(b + _BS) or a.startswith(b + _SL)
    except Exception:
        return False


def _dedup_roots(roots):
    out = []
    seen = set()
    try:
        items = list(roots or [])
    except Exception:
        return []
    cands = []
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
        try:
            norm = os.path.normpath(real)
        except Exception:
            continue
        key = os.path.normcase(norm)
        if key in seen:
            continue
        seen.add(key)
        cands.append(norm)
    cands.sort(key=len)
    for path in cands:
        dup = False
        for kept in out:
            if _prefix_match(path, kept):
                dup = True
                break
        if not dup:
            out.append(path)
    return out


def _encode_one(text):
    try:
        raw = str(text).encode("utf-16-le", "surrogatepass")
    except Exception:
        raw = "".encode("utf-16-le")
    try:
        arr = np.frombuffer(raw, dtype=np.uint16)
    except Exception:
        arr = np.zeros((0,), dtype=np.uint16)
    return arr


def _integer_column(values, narrow, wide):
    """Inspect before casting: never wrap an out-of-range length/depth/parent."""
    values = np.asarray(values)
    bounds = np.iinfo(narrow)
    dtype = narrow
    if values.size and (int(values.min()) < bounds.min or
                        int(values.max()) > bounds.max):
        dtype = wide
    bounds = np.iinfo(dtype)
    if values.size and (int(values.min()) < bounds.min or
                        int(values.max()) > bounds.max):
        raise ValueError("integer column exceeds supported range")
    return np.ascontiguousarray(values, dtype=dtype)


def _lower_units(arr):
    """Lowercase UTF-16 with bounded scratch; preserve shared name offsets."""
    out = _LOWER[arr]
    for start in range(0, max(0, arr.size - 1), 65536):
        part = arr[start:min(arr.size, start + 65537)]
        pairs = np.flatnonzero((part[:-1] >= 0xD800) & (part[:-1] <= 0xDBFF)
                              & (part[1:] >= 0xDC00) & (part[1:] <= 0xDFFF)) + start
        if not pairs.size:
            continue
        points = ((arr[pairs].astype(np.uint32) - 0xD800) * 1024
                  + arr[pairs + 1].astype(np.uint32) - 0xDC00 + 0x10000)
        for point in np.unique(points):
            low = chr(int(point)).lower()
            if len(low) == 1 and ord(low) >= 0x10000:
                value = ord(low) - 0x10000
                positions = pairs[points == point]
                out[positions] = 0xD800 + (value >> 10)
                out[positions + 1] = 0xDC00 + (value & 1023)
    # bytes.find uses this same immutable owner; no first-query raw copy.
    return np.frombuffer(out.tobytes(), dtype=np.uint16)


def _lower_codes(text):
    arr = _encode_one(text)
    try:
        return _lower_units(arr)
    except Exception:
        return arr


def _probe_arrays(h, bits=BLOOM_BITS):
    """Identical scalar/vector probes; mix polynomial hashes before masking."""
    h = np.asarray(h, dtype=np.uint64)
    h = ((h ^ (h >> np.uint64(16))) * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    h = ((h ^ (h >> np.uint64(15))) * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(16))
    mask = np.uint64(bits - 1)
    step = (h >> np.uint64(16)) | np.uint64(1)
    return tuple((h + np.uint64(i) * step) & mask for i in range(3))


def _probes(h):
    return tuple(int(p) for p in _probe_arrays(h))

def _trigram_hashes_str(s):
    # Hash the same UTF-16 units as the name blob, including surrogate pairs.
    return [int(h) for h in _trigram_hashes_arr(_lower_codes(s))]


def _trigram_hashes_arr(codes):
    try:
        c = np.ascontiguousarray(codes, dtype=np.uint64)
    except Exception:
        return np.zeros((0,), dtype=np.uint64)
    if c.shape[0] < 3:
        return np.zeros((0,), dtype=np.uint64)
    try:
        h = (c[:-2] * np.uint64(33) + c[1:-1]) * np.uint64(33) + c[2:]
        return (h & np.uint64(0xFFFFFFFF)).astype(np.uint64)
    except Exception:
        return np.zeros((0,), dtype=np.uint64)



_snapshot_lock = threading.RLock()
_snapshot_maps = {}


def _track_snapshot_map(array, path):
    # Track the native mapping, not the index: exported views may outlive it.
    key = os.path.normcase(os.path.abspath(path))
    with _snapshot_lock:
        refs = _snapshot_maps.setdefault(key, [])
        refs[:] = [ref for ref in refs if ref() is not None]
        refs.append(weakref.ref(array._mmap))
    return array


def _snapshot_generation(path):
    """Resolve a small, never-mapped pointer; accept old standalone archives."""
    with np.load(path, allow_pickle=False) as pointer:
        if "snapshot_generation" not in pointer:
            return os.path.abspath(path)
        name = str(pointer["snapshot_generation"])
    target = os.path.abspath(path)
    if (os.path.basename(name) != name or
            not name.startswith(os.path.basename(target) + ".v2.") or
            not name.endswith(".npz")):
        raise ValueError("invalid snapshot generation")
    return os.path.join(os.path.dirname(target), name)


def _collect_snapshot_generations(target, current):
    """Retry retired/orphan cleanup; sharing violations are harmless leftovers.

    Never rename or replace a mapped generation. Another process may still
    own a mapping, in which case Windows refuses unlink and a later save retries.
    A failed cleanup never invalidates either the pointer or an existing reader.
    """
    directory = os.path.dirname(target)
    prefix = os.path.basename(target) + ".v2."
    for name in os.listdir(directory):
        if not (name.startswith(prefix) and name.endswith(".npz")):
            continue
        path = os.path.join(directory, name)
        if path == current:
            continue
        # An unreferenced recent file may belong to another writer between
        # data fsync and pointer publication. Leave it for a later session.
        try:
            if time.time() - os.path.getmtime(path) < 86400:
                continue
        except OSError:
            continue
        key = os.path.normcase(os.path.abspath(path))
        refs = _snapshot_maps.get(key, [])
        refs[:] = [ref for ref in refs if ref() is not None]
        if refs:
            continue
        _snapshot_maps.pop(key, None)
        try:
            os.unlink(path)
        except OSError:
            pass


def _replace_snapshot(temporary, target):
    """Publish immutable data under a new name, then atomically switch a pointer.

    The destination data file is unique even across processes. Only the tiny
    pointer is replaced; it is read with an ordinary, promptly closed handle.
    Legacy standalone NPZ readers also close their probe before creating maps.
    """
    with _snapshot_lock:
        fd, generation = tempfile.mkstemp(prefix=os.path.basename(target) + ".v2.",
                                           suffix=".npz", dir=os.path.dirname(target))
        os.close(fd)
        os.replace(temporary, generation)
        fd, pointer = tempfile.mkstemp(prefix=".localdocs-pointer-", suffix=".npz",
                                       dir=os.path.dirname(target))
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, version=np.int64(2),
                         snapshot_generation=np.array(os.path.basename(generation)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pointer, target)
        finally:
            if os.path.exists(pointer):
                os.unlink(pointer)
        _collect_snapshot_generations(target, generation)


def _atomic_savez(path, data):
    """Commit a complete archive, preserving the previous file on failure."""
    target = os.path.abspath(path)
    fd, temporary = tempfile.mkstemp(prefix=".localdocs-", suffix=".npz",
                                     dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **data)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_snapshot(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)





class _ArenaBytes:
    """Expose UTF-16 byte positions to the existing exact matcher, without a copy."""
    def __init__(self, arena):
        self.arena = arena
        self.lookup = {int(unit): i for i, unit in enumerate(arena.alphabet)}
        self.raw = (_MappedBytes(arena.codes) if isinstance(arena.codes, np.memmap)
                    else arena.codes.base if isinstance(arena.codes.base, bytes)
                    and len(arena.codes.base) == arena.codes.nbytes else arena.codes.tobytes())
        self.needles = {}

    def find(self, needle, start=0, end=None):
        if start & 1:
            start += 1
        encoded = self.needles.get(needle)
        if encoded is None:
            try:
                encoded = bytes(self.lookup[int(unit)] for unit in np.frombuffer(needle, dtype=np.uint16))
            except KeyError:
                return -1
            # This cache is per query-size, bounded independently of row count.
            if len(self.needles) >= 128:
                self.needles.clear()
            self.needles[needle] = encoded
        stop = len(self.arena) if end is None else int(end) // 2
        at = self.raw.find(encoded, int(start) // 2, stop)
        return at * 2 if at >= 0 else -1


def _pack_name_arenas(low, original):
    """Bounded scratch, with a no-loss UTF-16 fallback for large alphabets."""
    if isinstance(low, _NameArena) and isinstance(original, _NameArena):
        return low, original
    if low.shape != original.shape:
        return low, original
    present = np.zeros(65536, dtype=bool)
    differences = 0
    for start in range(0, len(low), 65536):
        part, orig = low[start:start + 65536], original[start:start + 65536]
        present[part] = True
        differences += int(np.count_nonzero(part != orig))
    alphabet = np.flatnonzero(present).astype(np.uint16)
    if len(alphabet) > 256 or not len(alphabet):
        return low, original
    position_dtype = np.uint32 if len(low) < 2 ** 32 else np.uint64
    # A heavily mixed-case corpus can be smaller as two byte-coded arenas.
    # Keep the simpler UTF-16 fallback if the overlay would exceed its owner.
    if differences * (np.dtype(position_dtype).itemsize + 2) > original.nbytes:
        return low, original
    lookup = np.zeros(65536, dtype=np.uint8)
    lookup[alphabet] = np.arange(len(alphabet), dtype=np.uint8)
    codes = np.empty(len(low), dtype=np.uint8)
    positions = np.empty(differences, dtype=position_dtype)
    spelling = np.empty(differences, dtype=np.uint16)
    cursor = 0
    for start in range(0, len(low), 65536):
        part, orig = low[start:start + 65536], original[start:start + 65536]
        codes[start:start + len(part)] = lookup[part]
        changed = np.flatnonzero(part != orig)
        count = len(changed)
        positions[cursor:cursor + count] = changed + start
        spelling[cursor:cursor + count] = orig[changed]
        cursor += count
    # One immutable bytes owner also supplies bytes.find; never retain a raw copy.
    codes = np.frombuffer(codes.tobytes(), dtype=np.uint8)
    packed = _NameArena(codes, alphabet)
    return packed, (_NameArena(codes, alphabet, positions, spelling) if differences else packed)


def _names_arenas(items):
    """Build coded arenas in bounded chunks, without two full UTF-16 temporaries."""
    n = len(items)
    lengths = np.empty(n, dtype=np.uint32)
    for i, name in enumerate(items):
        lengths[i] = len(str(name).encode("utf-16-le", "surrogatepass")) // 2
    total = int(lengths.sum(dtype=np.uint64))
    present = np.zeros(65536, dtype=bool)
    differences = 0
    def chunks():
        for first in range(0, n, 16384):
            raw = "".join(str(name) for name in items[first:first + 16384]).encode("utf-16-le", "surrogatepass")
            original = np.frombuffer(raw, dtype=np.uint16)
            yield original, _lower_units(original)
    for original, low in chunks():
        present[low] = True
        differences += int(np.count_nonzero(low != original))
    alphabet = np.flatnonzero(present).astype(np.uint16)
    position_dtype = np.uint32 if total < 2 ** 32 else np.uint64
    if (not total or np.any(present[0xD800:0xE000]) or len(alphabet) > 256 or
            differences * (np.dtype(position_dtype).itemsize + 2) > total * 2):
        return None
    lookup = np.zeros(65536, dtype=np.uint8)
    lookup[alphabet] = np.arange(len(alphabet), dtype=np.uint8)
    codes = np.empty(total, dtype=np.uint8)
    positions = np.empty(differences, dtype=position_dtype)
    spelling = np.empty(differences, dtype=np.uint16)
    cursor = offset = 0
    for original, low in chunks():
        codes[offset:offset + len(low)] = lookup[low]
        changed = np.flatnonzero(low != original)
        count = len(changed)
        positions[cursor:cursor + count] = changed + offset
        spelling[cursor:cursor + count] = original[changed]
        offset += len(low)
        cursor += count
    codes = np.frombuffer(codes.tobytes(), dtype=np.uint8)
    low = _NameArena(codes, alphabet)
    original = _NameArena(codes, alphabet, positions, spelling) if differences else low
    return low, original, lengths






class _MappedNPZ:
    """Map uncompressed NPY members directly; np.load cannot mmap NPZ members.

    Mutable columns use private copy-on-write maps. Separate loads cannot see
    one another's writes, and no update can modify the persisted snapshot.
    """
    def __init__(self, path):
        import zipfile
        self.path = os.fspath(path)
        with zipfile.ZipFile(self.path) as archive:
            self.members = {i.filename[:-4]: i for i in archive.infolist()
                            if i.filename.endswith('.npy')}

    def __contains__(self, key):
        return key in self.members

    def __getitem__(self, key):
        import struct
        import zipfile
        info = self.members[key]
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError('mapped snapshot must be uncompressed')
        with open(self.path, 'rb') as stream:
            stream.seek(info.header_offset)
            header = stream.read(30)
            if header[:4] != b'PK' + bytes((3, 4)):
                raise ValueError('invalid ZIP member')
            name_len, extra_len = struct.unpack_from('<HH', header, 26)
            stream.seek(name_len + extra_len, 1)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError('unsupported NPY header')
            if dtype.hasobject:
                raise ValueError('object snapshot is unsafe')
            offset = stream.tell()
            count = 1
            for size in shape:
                count *= int(size)
            if offset + count * dtype.itemsize > info.header_offset + 30 + name_len + extra_len + info.file_size:
                raise ValueError('truncated snapshot member')
        if count == 0:
            return np.empty(shape, dtype=dtype)
        if os.name == "nt":
            import ctypes
            import msvcrt
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.CreateFileW.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_void_p)
            api.CreateFileW.restype = ctypes.c_void_p
            handle = api.CreateFileW(os.path.abspath(self.path), 0x80000000,
                                     7, None, 3, 0, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            except Exception:
                api.CloseHandle.argtypes = (ctypes.c_void_p,)
                api.CloseHandle(handle)
                raise
            with os.fdopen(fd, "rb") as stream:
                return _track_snapshot_map(
                    np.memmap(stream, dtype=dtype, mode='c', offset=offset,
                              shape=shape, order='F' if fortran else 'C'), self.path)
        return _track_snapshot_map(
            np.memmap(self.path, dtype=dtype, mode='c', offset=offset,
                      shape=shape, order='F' if fortran else 'C'), self.path)

    def close(self):
        pass  # each array owns its mapping

class _VolumeSource:
    """Owned, read-only volume access; convert swallowed USN errors to failures."""

    def __init__(self, api, handle):
        self.api, self.handle = api, handle

    def _call(self, method, *args):
        errors = []
        result = method(*args, on_error=errors.append)
        if errors:
            raise RuntimeError(str(errors[0]))
        return result

    def query(self):
        return self._call(self.api.query_journal, self.handle)

    def enumerate(self):
        return self._call(self.api.enum_volume, self.handle)

    def reader(self, journal_id, next_usn):
        errors = []
        reader = self.api.JournalReader(self.handle, journal_id, next_usn,
                                        on_error=errors.append)
        reader._fast_errors = errors
        if errors:
            raise RuntimeError(str(errors[0]))
        return reader

    def close(self):
        self.api.close_volume(self.handle)


def _usn_source(letter):
    """Overridable access hook. None preserves the ordinary walker path."""
    try:
        import localdocs_usn
        handle = localdocs_usn.open_volume(letter)  # fails without elevation; IsUserAnAdmin is unreliable for packaged Python
        if handle is not None:
            return _VolumeSource(localdocs_usn, handle)
    except Exception:
        pass
    return None


class NameIndex:
    """Columnar in-memory index of file names.

    Arrays only, no per-file Python objects on the hot path.
    """

    def __init__(self, on_error=None):
        self._memory_mode = ("compact" if os.environ.get("LOCALDOCS_NAME_MEMORY", "fast").strip().lower() == "compact" else "fast")
        self._block = 128 if self._memory_mode == "compact" else BLOCK
        self._on_error = on_error if callable(on_error) else None
        self.blob_low = np.zeros((0,), dtype=np.uint16)
        self.blob_orig = np.zeros((0,), dtype=np.uint16)
        self.off = np.zeros((0,), dtype=np.uint64)
        self.ln = np.zeros((0,), dtype=np.uint32)
        self.parent = np.zeros((0,), dtype=np.int32)
        self.flags = np.zeros((0,), dtype=np.uint8)
        self.frn = np.zeros((0,), dtype=np.uint64)
        self.size = np.zeros((0,), dtype=np.int64)
        self.mtime = np.zeros((0,), dtype=np.float64)
        self._root_of = np.zeros((0,), dtype=np.int32)
        self._depth = np.zeros((0,), dtype=np.int32)
        self._roots = []
        self._bloom = None
        self._bloom_generation = 0
        self._typing_blocks = None
        self._raw = None
        self._map = None
        self._path_cache = {}
        self._lru = collections.OrderedDict()
        self._ext_use = {}
        self._dir_use = {}
        self._journal_state = []
        self._base_n = 0
        self._delta_n = 0
        self._layout_dirty = False
        self._lock = threading.RLock()
        self._turbo_revision = 0
        self._turbo_layout = 0
        self._turbo_touched = {}
        self._turbo_lineage = object()
        self._turbo_last_touch = 0.0
        self._fuzzy_cache = None

    def _pack_memory(self):
        """Narrow columns safely and publish immutable, lossless name storage.

        Base metadata uses 16 B/row, before lazy metadata and USN identities.
        Dictionary-coded UTF-16 uses one B/unit plus sparse original spelling;
        >256-unit alphabets automatically retain the UTF-16 representation.
        Fast/compact Bloom remains 32/16 B/row. Starts use 4 B where safe.
        """
        for column, narrow, wide in (("off", np.uint32, np.uint64),
                                     ("ln", np.uint16, np.uint32),
                                     ("_depth", np.uint8, np.int32),
                                     ("parent", np.int32, np.int64)):
            setattr(self, column, _integer_column(getattr(self, column), narrow, wide))
        self.blob_low, self.blob_orig = _pack_name_arenas(self.blob_low, self.blob_orig)
        if not isinstance(self.blob_low, _NameArena):
            same_names = self.blob_orig.shape == self.blob_low.shape and all(
                np.array_equal(self.blob_orig[a:a + 65536], self.blob_low[a:a + 65536])
                for a in range(0, len(self.blob_low), 65536))
            if same_names:
                raw = self.blob_orig.tobytes()
                self.blob_low = np.frombuffer(raw, dtype=np.uint16)
                self.blob_orig = self.blob_low
                self._raw = raw
        else:
            self._raw = None
        for name, dtype, default in (("frn", np.uint64, 0),
                                     ("size", np.int64, -1),
                                     ("mtime", np.float64, -1)):
            column = getattr(self, name)
            if not isinstance(column, _DefaultColumn):
                count = sum(int(np.count_nonzero(column[a:a + 65536] != default))
                            for a in range(0, len(column), 65536))
                if count == 0:
                    setattr(self, name, _DefaultColumn(len(self), dtype, default))
                elif name != "frn" and count * 128 < column.nbytes:
                    values = {}
                    for a in range(0, len(column), 65536):
                        part = column[a:a + 65536]
                        values.update((a + int(k), part[k]) for k in np.flatnonzero(part != default))
                    setattr(self, name, _DefaultColumn(len(self), dtype, default, values))

    def _materialize_metadata(self):
        for name in ("size", "mtime"):
            value = getattr(self, name)
            if isinstance(value, _DefaultColumn):
                setattr(self, name, np.asarray(value))

    def _turbo_touch(self, rows):
        self._fuzzy_cache = None
        self._turbo_revision += 1
        self._turbo_last_touch = time.monotonic()
        self._turbo_touched.update((int(k), self._turbo_revision) for k in rows)

    def __len__(self):
        try:
            return int(self.off.shape[0])
        except Exception:
            return 0

    def _err(self, exc):
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:
                pass

    def _invalidate(self, keep_event_tables=False, preserve_turbo=False):
        with self._lock:
            self._directory_terms = None
            self._directory_terms_bytes = 0
            ready = getattr(self, '_search_ready', None)
            if ready is not None:
                ready.clear()
            self._fuzzy_cache = None
            if not keep_event_tables and not preserve_turbo:
                self._turbo_layout += 1
                self._turbo_touched.clear()
            if not keep_event_tables:
                self._event_base = None
            self._csr_cache = None
            self._starts_cache = None
            self._offset_order = None
            self._name_layout = None
            self._typing_blocks = None
            self._bloom_generation += 1
            self._bloom = None
            self._operator_ext = None
            self._operator_lengths = None
            self._reachable = None
            self._operator_positions = None
            self._directory_lower = None
            self._raw = None
            self._map = None
            self._path_cache = {}
            try:
                self._lru.clear()
            except Exception:
                pass

            if getattr(self, '_warm_requested', False):
                self.start_search_warmup()

    def start_search_warmup(self):
        """Prepare a generation privately; searches never wait on construction.

        Returns an Event for benchmarks/owners that need publication readiness.
        Derived arrays own their storage. Mutation invalidates the event and the
        worker retries rather than publishing caches for an obsolete generation.
        """
        with self._lock:
            self._warm_requested = True
            if not hasattr(self, '_search_ready'):
                self._search_ready = threading.Event()
            if (self._bloom is not None and getattr(self, '_operator_ext', None) is not None
                    and getattr(self, '_operator_lengths', None) is not None
                    and getattr(self, '_directory_terms', None) is not None):
                self._search_ready.set()
            if self._search_ready.is_set() or getattr(self, '_warm_thread', None) is not None:
                return self._search_ready
            self._warm_thread = threading.Thread(target=self._warm_search,
                name='localdocs-search-warmup', daemon=True)
            self._warm_thread.start()
            return self._search_ready

    def _warm_search(self):
        try:
            _parse_search_query('ext:pdf')  # optional filter imports, off the query path
            while True:
                with self._lock:
                    generation = self._bloom_generation
                self._ensure_bloom()
                with self._lock:
                    if generation != self._bloom_generation:
                        continue
                    snapshot = NameIndex(on_error=self._on_error)
                    snapshot._memory_mode, snapshot._block = self._memory_mode, self._block
                    snapshot._roots = list(self._roots)
                    # Names are replace-only arenas. Every retained derived
                    # cache below is detached, including UTF-16 bytes/starts.
                    snapshot.blob_low, snapshot.blob_orig = self.blob_low, self.blob_orig
                    for key in ('off', 'ln', 'parent', 'flags', '_root_of'):
                        setattr(snapshot, key, getattr(self, key).copy())
                if self._memory_mode == 'compact':
                    # Compact is an explicit memory/latency tradeoff: retain
                    # only the Bloom and byte starts until a broad query needs
                    # the optional T5 derived columns.
                    with self._lock:
                        if generation != self._bloom_generation:
                            continue
                        self._raw_bytes()
                        self._starts_np()
                        self._search_ready.set()
                        self._warm_thread = None
                        return
                snapshot._position_rows(np.empty(0, dtype=np.int64))
                snapshot._raw_bytes()
                snapshot._starts_np()
                snapshot._operator_extensions()
                snapshot._reachable_rows()
                snapshot._operator_path_lengths()
                snapshot._build_directory_terms()
                with self._lock:
                    if generation != self._bloom_generation:
                        continue
                    for key in ('_raw', '_offset_order', '_operator_ext',
                                '_operator_has_expansion', '_operator_lengths', '_reachable',
                                '_operator_positions', '_directory_lower',
                                '_directory_terms', '_directory_terms_bytes'):
                        setattr(self, key, getattr(snapshot, key, None))
                    self._starts_cache = (len(self), self.off, snapshot._starts_np())
                    self._search_ready.set()
                    self._warm_thread = None
                    return
        except Exception as exc:
            self._err(exc)
            with self._lock:
                self._warm_thread = None

    def _build_directory_terms(self):
        """Directory-only paths prepared after publication, never paths per file."""
        dirs = np.flatnonzero(((self.flags & FLAG_DIR) != 0) & self._reachable_rows())
        # Component joins can reset paths for absolute names. Empty names are
        # skipped by path_of. Preserve the scalar fallback for such graphs.
        unusual = np.flatnonzero((self.ln == 0) & (self.parent >= 0))
        tiny = np.flatnonzero((self.ln <= 2) & (self.parent >= 0))
        if any(self._orig_name(int(k)) in ('.', '..') for k in tiny):
            unusual = [1]
        for token in ('/', chr(92), ':'):
            found = self._operator_name_ids(token)
            if np.any(self.parent[found] >= 0):
                unusual = [1]
                break
        if not len(dirs) or len(unusual):
            self._directory_terms = None
            self._directory_terms_bytes = 0
            return
        # Memoize each directory once. Rewalking every ancestor for every
        # directory was O(directories * depth) Python path construction.
        paths = [None] * len(dirs)
        slots = {int(k): j for j, k in enumerate(dirs)}
        for row in dirs:
            chain, seen, cur = [], set(), int(row)
            while cur >= 0 and cur in slots and paths[slots[cur]] is None:
                if cur in seen or len(chain) >= 4095:
                    self._directory_terms = None
                    self._directory_terms_bytes = 0
                    return
                seen.add(cur)
                chain.append(cur)
                cur = int(self.parent[cur])
            for k in reversed(chain):
                p = int(self.parent[k])
                if p < 0:
                    path = str(self._roots[int(self._root_of[k])])
                elif p in slots and paths[slots[p]] is not None:
                    path = os.path.join(paths[slots[p]], self._orig_name(k))
                else:
                    path = self.path_of(k, _cache=False)
                paths[slots[k]] = path
        folders = tuple(ntpath.basename(path).lower() for path in paths)
        self._directory_terms = (dirs.copy(), tuple(paths), folders)
        self._directory_lower = tuple(path.lower() for path in paths)
        self._directory_terms_bytes = (dirs.nbytes +
            sum(sys.getsizeof(p) + sys.getsizeof(f) + sys.getsizeof(low) + 24
                for p, f, low in zip(paths, folders, self._directory_lower)))

    def _directory_parent_mask(self, terms, folder=False, rows=None):
        table = getattr(self, '_directory_terms', None)
        if table is None:
            return None
        dirs, paths, folders = table
        values = folders if folder else getattr(self, '_directory_lower', None)
        if values is None:
            values = tuple(path.lower() for path in paths)
        # Only examine parents of files surviving the extension/metadata masks.
        # Complete paths of these directory rows already include all ancestors.
        parents = self.parent if rows is None else self.parent[rows]
        candidates = np.unique(parents[parents >= 0])
        slots = np.searchsorted(dirs, candidates)
        valid = slots < len(dirs)
        candidates, slots = candidates[valid], slots[valid]
        valid = dirs[slots] == candidates
        candidates, slots = candidates[valid], slots[valid]
        matches = np.fromiter((all(t in values[int(j)] for t in terms)
                               for j in slots), dtype=bool, count=len(slots))
        selected = np.zeros(len(self), dtype=bool)
        selected[candidates[matches]] = True
        all_parents = self.parent
        valid_parent = (all_parents >= 0) & (all_parents < len(self))
        keep = np.zeros(len(self), dtype=bool)
        keep[valid_parent] = selected[all_parents[valid_parent]]
        # Preserve top-level files and unusual non-directory parent semantics.
        check_rows = np.arange(len(self)) if rows is None else rows
        p = all_parents[check_rows]
        slots = np.searchsorted(dirs, np.maximum(p, 0))
        ordinary = (p >= 0) & (slots < len(dirs))
        ordinary &= dirs[np.minimum(slots, len(dirs) - 1)] == p
        fallback = check_rows[~ordinary & ((self.flags[check_rows] & (FLAG_DIR | FLAG_TOMB)) == 0)]
        for k in fallback:
            path = self.path_of(int(k), _cache=False)
            value = ntpath.basename(ntpath.dirname(path)) if folder else path
            keep[k] = all(t in value.lower() for t in terms)
        return keep

    def _directory_prefix_ids(self, ids, prefixes):
        # A whole-root constraint is answered by the root-id column alone.
        covered = np.asarray([any(self._contains_path(root, p) for p in prefixes)
                              for root in self._roots], dtype=bool)
        if len(ids) == 0 or np.all(covered[self._root_of[ids]]):
            return ids, None
        table = getattr(self, '_directory_terms', None)
        if table is None:
            return ids, prefixes
        dirs, paths, _ = table
        certain, possible = [], []
        for k, path in zip(dirs, paths):
            if any(self._contains_path(path, p) for p in prefixes):
                certain.append(k)
            elif any(self._contains_path(p, path) for p in prefixes):
                possible.append(k)
        parents = self.parent[ids]
        keep = np.isin(parents, certain)
        check = ~keep & (np.isin(parents, possible) | ~np.isin(parents, dirs))
        for j in np.flatnonzero(check):
            path = self.path_of(int(ids[j]))
            keep[j] = any(self._contains_path(path, p) for p in prefixes)
        return ids[keep], None

    @classmethod
    def from_arrays(cls, blob_low, blob_orig, off, ln, parent, flags, roots,
                    root_of, frn=None, size=None, mtime=None, depth=None,
                    on_error=None):
        idx = cls(on_error=on_error)
        try:
            n = int(np.asanyarray(off).shape[0])
            idx.blob_low = np.ascontiguousarray(blob_low, dtype=np.uint16)
            idx.blob_orig = np.ascontiguousarray(blob_orig, dtype=np.uint16)
            idx.off = _integer_column(off, np.uint32, np.uint64)
            idx.ln = _integer_column(ln, np.uint16, np.uint32)
            idx.parent = _integer_column(parent, np.int32, np.int64)
            idx.flags = np.ascontiguousarray(flags, dtype=np.uint8)
            idx._root_of = np.ascontiguousarray(root_of, dtype=np.int32)
            if frn is None:
                idx.frn = _DefaultColumn(n, np.uint64, 0)
            else:
                idx.frn = np.ascontiguousarray(frn, dtype=np.uint64)
            if size is None:
                idx.size = _DefaultColumn(n, np.int64, -1)
            else:
                idx.size = np.ascontiguousarray(size, dtype=np.int64)
            if mtime is None:
                idx.mtime = _DefaultColumn(n, np.float64, -1)
            else:
                idx.mtime = np.ascontiguousarray(mtime, dtype=np.float64)
            if depth is None:
                d = np.zeros((n,), dtype=np.int32)
                par = idx.parent
                for i in range(n):
                    try:
                        p = int(par[i])
                        if p >= 0 and p < i:
                            d[i] = d[p] + 1
                    except Exception:
                        pass
                idx._depth = d
            else:
                idx._depth = _integer_column(depth, np.uint8, np.int32)
            idx._roots = [str(r) for r in (roots or [])]
            idx._base_n = n
            idx._delta_n = 0
            idx._pack_memory()
        except Exception as exc:
            idx._err(exc)
            return cls(on_error=on_error)
        return idx

    @classmethod
    def from_names(cls, names, root=None, parent=None, flags=None, frn=None,
                   on_error=None):
        try:
            items = names if isinstance(names, (list, tuple)) else list(names or [])
        except Exception:
            return cls(on_error=on_error)
        try:
            base = str(root) if root else os.path.abspath("memroot")
        except Exception:
            base = "memroot"
        n = len(items)
        par = (parent if isinstance(parent, (list, tuple, np.ndarray)) else
               list(parent) if parent is not None else ())
        fl = (flags if isinstance(flags, (list, tuple, np.ndarray)) else
              list(flags) if flags is not None else ())
        try:
            packed = _names_arenas(items)
            if packed is None:
                arena = bytearray()
                lengths = np.empty(n, dtype=np.uint32)
                for i, name in enumerate(items):
                    raw = str(name).encode("utf-16-le", "surrogatepass")
                    lengths[i] = len(raw) // 2
                    arena.extend(raw)
                orig = np.frombuffer(bytes(arena), dtype=np.uint16)
                del arena
                low = _lower_units(orig)
            else:
                low, orig, lengths = packed
            offsets = np.empty(n, dtype=np.uint32 if int(lengths.sum(dtype=np.uint64)) < 2 ** 32 else np.uint64)
            if n:
                offsets[0] = 0
                np.cumsum(lengths[:-1], dtype=np.uint64, out=offsets[1:])
            parent_dtype = np.int32
            for p in par:
                try:
                    if not np.iinfo(np.int32).min <= int(p) <= np.iinfo(np.int32).max:
                        parent_dtype = np.int64
                except Exception:
                    pass
            parents = np.full(n, -1, dtype=parent_dtype)
            file_flags = np.zeros(n, dtype=np.uint8)
            for i in range(n):
                try:
                    parents[i] = int(par[i]) if i < len(par) else -1
                except Exception:
                    pass
                try:
                    file_flags[i] = int(fl[i]) & 255 if i < len(fl) else 0
                except Exception:
                    pass
            parents[(parents < 0) | (parents >= n)] = -1
            depth = np.zeros(n, dtype=np.int32)
            for _ in range(64):
                next_depth = np.where(parents >= 0, depth[np.maximum(parents, 0)] + 1, 0).astype(np.int32)
                if np.array_equal(next_depth, depth):
                    break
                depth = next_depth
            del next_depth
            if isinstance(low, _NameArena):
                idx = cls(on_error=on_error)
                idx.blob_low, idx.blob_orig = low, orig
                idx.off, idx.ln = offsets, lengths
                idx.parent, idx.flags = parents, file_flags
                idx._roots, idx._root_of, idx._depth = [base], np.zeros(n, dtype=np.int32), depth
                idx.frn = _DefaultColumn(n, np.uint64, 0)
                idx.size = _DefaultColumn(n, np.int64, -1)
                idx.mtime = _DefaultColumn(n, np.float64, -1)
                idx._base_n = n
                idx._pack_memory()
            else:
                idx = cls.from_arrays(low, orig, offsets, lengths,
                                      parents, file_flags, [base],
                                      np.zeros(n, dtype=np.int32), depth=depth, on_error=on_error)
            del low, orig, packed, offsets, lengths, parents, file_flags, depth
        except Exception as exc:
            idx = cls(on_error=on_error)
            idx._err(exc)
            return idx
        if frn is not None:
            try:
                refs = np.asarray(frn, dtype=np.uint64)
                if refs.shape == (n,):
                    idx.frn = refs.copy()
            except Exception as exc:
                idx._err(exc)
        return idx

    @staticmethod
    def _assemble_fast(records, roots, on_error=None):
        """Bulk (vectorized) construction: one join/encode for all names instead of one numpy call per entry."""
        import operator
        # Most producers already supply ordered records. Borrow that sequence;
        # sorting unordered walker batches needs only one list of references.
        ordered = isinstance(records, (list, tuple)) and all(
            records[k - 1][0] <= records[k][0] for k in range(1, len(records)))
        recs = records if ordered else sorted(records, key=operator.itemgetter(0))
        n = len(recs)
        if n == 0:
            return None
        names = [str(r[1]) for r in recs]
        packed = _names_arenas(names)
        if packed is not None:
            low, orig, lens = packed
        else:
            arena = bytearray()
            lens = np.empty(n, dtype=np.uint32)
            for i, name in enumerate(names):
                raw = name.encode("utf-16-le", "surrogatepass")
                lens[i] = len(raw) // 2
                arena.extend(raw)
            orig = np.frombuffer(bytes(arena), dtype=np.uint16)
            del arena
            low = _lower_units(orig)
        ids = np.fromiter((r[0] for r in recs), dtype=np.int64, count=n)
        pid = np.fromiter((r[2] for r in recs), dtype=np.int64, count=n)
        flg = np.fromiter((int(r[3]) & 0xFF for r in recs), dtype=np.uint8, count=n)
        rot = np.fromiter((max(int(r[4]), 0) for r in recs), dtype=np.int32, count=n)
        parent_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
        par = np.full((n,), -1, dtype=parent_dtype)
        has = pid >= 0
        if has.any():
            where = np.searchsorted(ids, pid[has])
            where = np.minimum(where, n - 1)
            good = ids[where] == pid[has]
            tmp = np.full(int(has.sum()), -1, dtype=np.int64)
            tmp[good] = where[good]
            par[has] = tmp
            del where, good, tmp
        del ids, pid, has
        depth = np.zeros((n,), dtype=np.int32)
        for _ in range(64):
            nd = np.where(par >= 0, depth[np.maximum(par, 0)] + 1, 0).astype(np.int32)
            if np.array_equal(nd, depth):
                break
            depth = nd
        del nd
        idx = NameIndex(on_error=on_error)
        off = np.empty(n, dtype=np.uint32 if int(lens.sum(dtype=np.uint64)) < 2 ** 32 else np.uint64)
        off[0] = 0
        np.cumsum(lens[:-1], dtype=np.uint64, out=off[1:])
        idx.blob_orig = orig
        idx.blob_low = low
        idx.off = off
        idx.ln = _integer_column(lens, np.uint16, np.uint32)
        del lens, names
        idx.parent = par
        idx.flags = flg
        idx.frn = _DefaultColumn(n, np.uint64, 0)
        # Explicit producer metadata is authoritative even for large batches.
        # Unknown columns stay sparse; packing narrows sparse known exceptions.
        for key, position, dtype in (('size', 5, np.int64), ('mtime', 6, np.float64)):
            if any(len(record) > position and record[position] != -1 for record in recs):
                value = np.fromiter((record[position] if len(record) > position else -1
                                     for record in recs), dtype=dtype, count=n)
            else:
                value = _DefaultColumn(n, dtype, -1)
            setattr(idx, key, value)
        idx._root_of = rot
        idx._depth = depth
        idx._roots = [str(r) for r in (roots or [])]
        idx._base_n = n
        idx._delta_n = 0
        # Do not pin wide build columns or the original arena while packing.
        del depth, orig, low, packed, par, flg, rot, off, recs
        idx._pack_memory()
        return idx

    @staticmethod
    def _assemble(records, roots, on_error=None):
        idx = NameIndex(on_error=on_error)
        try:
            quick = NameIndex._assemble_fast(records, roots, on_error)
            if quick is not None:
                return quick
        except Exception as exc:
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
        try:
            recs = sorted(records, key=lambda r: int(r[0]))
        except Exception:
            return idx
        n = len(recs)
        if n == 0:
            idx._roots = [str(r) for r in (roots or [])]
            idx._pack_memory()
            return idx
        low_parts = []
        orig_parts = []
        off = np.zeros((n,), dtype=np.uint64)
        ln = np.zeros((n,), dtype=np.uint32)
        par = np.zeros((n,), dtype=np.int32 if n <= np.iinfo(np.int32).max else np.int64)
        fl = np.zeros((n,), dtype=np.uint8)
        ro = np.zeros((n,), dtype=np.int32)
        depth = np.zeros((n,), dtype=np.int32)
        sizes = np.full(n, -1, dtype=np.int64)
        mtimes = np.full(n, -1.0, dtype=np.float64)
        pos = 0
        new_order = {}
        for k, rec in enumerate(recs):
            try:
                old_id, nm, p, f, r = rec[0], str(rec[1]), int(rec[2]), int(rec[3]), int(rec[4])
            except Exception:
                continue
            if len(rec) > 5:
                sizes[k] = int(rec[5])
            if len(rec) > 6:
                mtimes[k] = float(rec[6])
            new_order[int(rec[0])] = k
            low = _lower_codes(nm)
            orig = _encode_one(nm)
            low_parts.append(low)
            orig_parts.append(orig)
            off[k] = np.uint64(pos)
            ln[k] = np.uint32(low.shape[0])
            pos += int(low.shape[0])
            fl[k] = np.uint8(f & 0xFF)
            if r < 0:
                r = 0
            ro[k] = np.int32(r)
            par[k] = np.int32(-1)
            try:
                if isinstance(p, int) and p in new_order and p != int(rec[0]):
                    par[k] = new_order[p]
                    depth[k] = np.int32(depth[new_order[p]] + 1)
            except Exception:
                pass
        try:
            if low_parts:
                blob_low = np.concatenate(low_parts)
                blob_orig = np.concatenate(orig_parts)
                del low_parts, orig_parts
            else:
                blob_low = np.zeros((0,), dtype=np.uint16)
                blob_orig = np.zeros((0,), dtype=np.uint16)
        except Exception:
            return idx
        idx.blob_low = np.ascontiguousarray(blob_low, dtype=np.uint16)
        idx.blob_orig = np.ascontiguousarray(blob_orig, dtype=np.uint16)
        idx.off = off
        idx.ln = ln
        idx.parent = par
        idx.flags = fl
        idx.frn = _DefaultColumn(n, np.uint64, 0)
        idx.size = sizes
        idx.mtime = mtimes
        idx._root_of = ro
        idx._depth = depth
        idx._roots = [str(r) for r in (roots or [])]
        idx._base_n = n
        idx._delta_n = 0
        idx._pack_memory()
        return idx

    @classmethod
    def from_usn(cls, vol, root_path, skip_names=None, on_error=None):
        """Build the index from an NTFS MFT enumeration (localdocs_usn.enum_volume result).

        A synthetic root entry (the drive) is prepended. Volume-root children
        and legacy synthetic records with parent zero hang below it; other
        unreachable branches are tombstoned. Folder-name junk (node_modules,
        .git, AppData, ...) is flagged, not dropped, so it is
        still searchable but ranked last.
        """
        try:
            n0 = int(vol["frn"].shape[0])
        except Exception:
            return cls(on_error=on_error)
        if n0 == 0:
            return cls(on_error=on_error)
        mask = np.uint64(0x0000FFFFFFFFFFFF)
        frn = np.asarray(vol["frn"], dtype=np.uint64)
        parent_ids = frn & mask
        pfrn = np.asarray(vol["parent"], dtype=np.uint64) & mask
        order = np.argsort(parent_ids, kind="stable")
        fs = parent_ids[order]
        pos = np.minimum(np.searchsorted(fs, pfrn), n0 - 1)
        found = fs[pos] == pfrn
        par = np.where(found, order[pos] + 1, -1).astype(np.int64)
        own = np.arange(1, n0 + 1, dtype=np.int64)
        par[par == own] = -1
        # Zero is the parent sentinel in legacy/synthetic enumeration data.
        # Slot five is the NTFS volume root, even with a sequence number.
        par[(pfrn == 0) | (pfrn == 5)] = 0
        root_name = np.frombuffer(str(root_path).rstrip(chr(92) + "/").encode("utf-16-le"), dtype=np.uint16)
        shift = int(root_name.shape[0])
        blob_orig = np.concatenate([root_name, np.asarray(vol["blob"], dtype=np.uint16)])
        off = np.concatenate([np.zeros((1,), dtype=np.uint64), np.asarray(vol["name_off"], dtype=np.uint64) + np.uint64(shift)])
        ln = np.concatenate([np.full((1,), shift, dtype=np.uint32), np.asarray(vol["name_len"], dtype=np.uint32)])
        parent = _integer_column(
            np.concatenate([np.full((1,), -1, dtype=np.int64), par]),
            np.int32, np.int64)
        del parent_ids, pfrn, order, fs, pos, found, par, own
        attrs = np.concatenate([np.full((1,), 0x10, dtype=np.uint32), np.asarray(vol["attrs"], dtype=np.uint32)])
        flags = np.zeros((n0 + 1,), dtype=np.uint8)
        flags[(attrs & np.uint32(0x10)) != 0] |= FLAG_DIR
        flags[(attrs & np.uint32(0x6)) != 0] |= FLAG_HIDDEN
        # Missing parents and cycles never reach the synthetic root. Pointer
        # doubling also handles branches deeper than the depth scoring limit.
        ancestor = np.where(parent >= 0, parent, np.arange(n0 + 1))
        for _ in range((n0 + 1).bit_length()):
            ancestor = ancestor[ancestor]
        flags[ancestor != 0] |= FLAG_TOMB
        names_skip = {str(x).casefold() for x in (skip_names if skip_names is not None else _SKIP_NAMES)}
        junk_dir = np.zeros((n0 + 1,), dtype=bool)
        dir_ids = np.flatnonzero((attrs & np.uint32(0x10)) != 0)
        for k in dir_ids.tolist():
            a = int(off[k])
            nm = blob_orig[a:a + int(ln[k])].tobytes().decode("utf-16-le", "replace").casefold()
            if nm in names_skip or nm.startswith("$"):
                junk_dir[k] = True
        junk = junk_dir.copy()
        for _ in range(64):
            inherit = junk | np.where(parent >= 0, junk[np.maximum(parent, 0)], False)
            if np.array_equal(inherit, junk):
                break
            junk = inherit
        flags[junk] |= FLAG_JUNK
        depth = np.zeros((n0 + 1,), dtype=np.int32)
        cur = parent.astype(np.int64)
        for _ in range(64):
            active = cur >= 0
            if not active.any():
                break
            depth += active.astype(np.int32)
            cur = np.where(active, parent[np.maximum(cur, 0)], -1)
        del ancestor, attrs, junk_dir, dir_ids, junk, inherit, cur, active
        idx = cls(on_error=on_error)
        idx.blob_orig = np.ascontiguousarray(blob_orig, dtype=np.uint16)
        idx.blob_low = np.ascontiguousarray(_lower_units(blob_orig), dtype=np.uint16)
        idx.off = off
        idx.ln = ln
        idx.parent = parent
        idx.flags = flags
        idx.frn = np.concatenate([np.full((1,), 5, dtype=np.uint64), frn])
        idx.size = _DefaultColumn(n0 + 1, np.int64, -1)
        idx.mtime = _DefaultColumn(n0 + 1, np.float64, -1)
        idx._root_of = np.zeros((n0 + 1,), dtype=np.int32)
        idx._depth = depth
        idx._roots = [str(root_path)]
        idx._base_n = n0 + 1
        idx._delta_n = 0
        del blob_orig, off, ln, parent, flags, depth
        idx._pack_memory()
        return idx

    @classmethod
    def from_walk(cls, roots, workers=8, skip=None, max_entries=600000,
                  on_batch=None, on_error=None):
        def err(exc):
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
        try:
            normed = _dedup_roots(roots)
        except Exception as exc:
            err(exc)
            return cls(on_error=on_error)
        try:
            w = max(1, min(int(workers), 32))
        except Exception:
            w = 8
        try:
            cap = max(1, int(max_entries))
        except Exception:
            cap = 600000
        skipset = set()
        if skip:
            try:
                for s in skip:
                    try:
                        skipset.add(str(s).casefold())
                    except Exception:
                        pass
            except Exception:
                pass
        if not normed:
            return cls(on_error=on_error)
        if os.environ.get("LOCALDOCS_NT_WALK", "1") != "0":
            try:
                import localdocs_ntwalk
                if localdocs_ntwalk.available():
                    import inspect
                    status = {}
                    options = {"on_error": on_error}
                    if "status" in inspect.signature(localdocs_ntwalk.walk).parameters:
                        options["status"] = status
                    recs = localdocs_ntwalk.walk(normed, w, cap, set(_SKIP_NAMES) | skipset, _SKIP_EXT, _PENALTY_EXT,
                                                 FLAG_DIR, FLAG_HIDDEN, FLAG_JUNK, **options)
                    status = getattr(recs, "status", status)
                    if recs or status:
                        idx = cls._assemble(recs, normed, on_error=on_error)
                        idx._walk_status = status
                        return idx
            except Exception as exc:
                err(exc)  # fall back to the portable scandir walker
        try:
            return cls._swarm(normed, w, cap, skipset, on_batch, on_error)
        except Exception as exc:
            err(exc)
            return cls(on_error=on_error)

    @classmethod
    def _swarm(cls, normed, workers, cap, skipset, on_batch, on_error):
        def err(exc):
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
        tasks = queue.Queue()
        guard = threading.Lock()
        counter = [0]
        pending = [0]
        seen_real = set()
        visited = set()
        shared = []
        status = {root: {"state": "ok", "error": ""} for root in normed}
        for rp, root in enumerate(normed):
            with guard:
                my = counter[0]
                counter[0] += 1
            try:
                base = os.path.basename(root) or root
            except Exception:
                base = root
            shared.append((my, base, -1, FLAG_DIR, rp))
            tasks.put((root, my, rp))
            pending[0] += 1
            try:
                st = os.stat(root)
                visited.add((st.st_dev, st.st_ino))
                seen_real.add(os.path.normcase(os.path.realpath(root)))
            except OSError:
                pass
        last_pub = [len(shared)]

        def maybe_publish(force=False):
            if not callable(on_batch):
                return
            with guard:
                n0 = len(shared)
            if force or (n0 - last_pub[0] >= 2000):
                last_pub[0] = n0
                try:
                    with guard:
                        snap_recs = list(shared)
                    snap = cls._assemble(snap_recs, normed, on_error=None)
                    on_batch(snap)
                except Exception as exc:
                    err(exc)

        def worker():
            priv = []
            while True:
                try:
                    cur, pidx, rpos = tasks.get(timeout=0.2)
                except Exception:
                    with guard:
                        done = pending[0] <= 0
                    if done:
                        break
                    continue
                try:
                    with guard:
                        capped = counter[0] >= cap
                    if capped:
                        with guard:
                            status[normed[rpos]]["state"] = "partial"
                        continue
                    try:
                        with os.scandir(cur) as it:
                            dents = list(it)
                    except OSError as exc:
                        with guard:
                            state = ("denied" if isinstance(exc, PermissionError) else
                                     "offline" if isinstance(exc, FileNotFoundError) else "partial")
                            status[normed[rpos]] = {"state": state if cur == normed[rpos] else "partial",
                                                   "error": str(exc)}
                        continue
                    for ent in dents:
                        with guard:
                            if counter[0] >= cap:
                                status[normed[rpos]]["state"] = "partial"
                                break
                        try:
                            nm = ent.name
                        except OSError:
                            continue
                        try:
                            lnk = ent.is_symlink()
                        except OSError:
                            continue
                        try:
                            idr = ent.is_dir(follow_symlinks=False)
                        except OSError:
                            continue
                        if idr:
                            if nm.startswith("."):
                                continue
                            try:
                                cf = nm.casefold()
                            except Exception:
                                continue
                            if cf in _SKIP_NAMES or cf in skipset:
                                continue
                            if lnk:
                                continue
                            full = ent.path
                            try:
                                rk = os.path.normcase(os.path.realpath(full))
                            except Exception:
                                continue
                            with guard:
                                if rk in seen_real:
                                    continue
                            try:
                                st = ent.stat(follow_symlinks=False)
                                dk = (st.st_dev, st.st_ino) if st.st_ino else None
                            except OSError:
                                continue
                            if dk is not None:
                                with guard:
                                    if dk in visited:
                                        continue
                                    visited.add(dk)
                            with guard:
                                seen_real.add(rk)
                                my = counter[0]
                                counter[0] += 1
                            priv.append((my, nm, pidx, FLAG_DIR, rpos))
                            tasks.put((full, my, rpos))
                            with guard:
                                pending[0] += 1
                        else:
                            try:
                                isf = ent.is_file(follow_symlinks=False)
                            except OSError:
                                continue
                            if not isf:
                                if not lnk:
                                    continue
                                try:
                                    if not ent.is_file(follow_symlinks=True):
                                        continue
                                except OSError:
                                    continue
                            try:
                                ext = os.path.splitext(nm)[1].lower()
                            except Exception:
                                ext = ""
                            if ext in _SKIP_EXT:
                                continue
                            fl = 0
                            if nm.startswith("."):
                                fl |= FLAG_HIDDEN
                            if ext in _PENALTY_EXT:
                                fl |= FLAG_JUNK
                            with guard:
                                my = counter[0]
                                counter[0] += 1
                            priv.append((my, nm, pidx, fl, rpos))
                        if len(priv) >= 1000:
                            with guard:
                                shared.extend(priv)
                            priv = []
                            maybe_publish(False)
                finally:
                    try:
                        tasks.task_done()
                    except Exception:
                        pass
                    with guard:
                        pending[0] -= 1
            if priv:
                with guard:
                    shared.extend(priv)
                maybe_publish(False)

        threads = []
        for _ in range(workers):
            try:
                t = threading.Thread(target=worker, daemon=True)
                threads.append(t)
                t.start()
            except Exception as exc:
                err(exc)
        for t in threads:
            try:
                t.join(timeout=600)
            except Exception:
                pass
        maybe_publish(True)
        try:
            with guard:
                recs = shared  # workers have joined; no second list of N pointers
            idx = cls._assemble(recs, normed, on_error=on_error)
            idx._walk_status = status
            return idx
        except Exception as exc:
            err(exc)
            return cls(on_error=on_error)

    def path_of(self, i, _cache=True):
        try:
            n = int(self.off.shape[0])
            k = int(i)
        except Exception:
            return ""
        if k < 0 or k >= n:
            return ""
        try:
            hit = self._path_cache.get(k) if _cache else None
        except Exception:
            hit = None
        if hit:
            return hit
        try:
            parts = []
            cur = k
            guard = 0
            root_pos = 0
            while cur >= 0 and guard < 4096:
                guard += 1
                try:
                    rp = int(self._root_of[cur])
                except Exception:
                    rp = 0
                root_pos = rp
                try:
                    p = int(self.parent[cur])
                except Exception:
                    p = -1
                if p < 0:
                    break
                try:
                    s = int(self.off[cur])
                    e = s + int(self.ln[cur])
                    nm = self.blob_orig[s:e].tobytes().decode("utf-16-le")
                except Exception:
                    nm = ""
                parts.append(nm)
                cur = p
            try:
                base_root = str(self._roots[root_pos])
            except Exception:
                base_root = ""
            segs = [base_root]
            for nm in reversed(parts):
                if nm:
                    try:
                        segs = [os.path.join(segs[0], nm)]
                    except Exception:
                        pass
            out = segs[0]
            if len(parts) == 0:
                try:
                    s = int(self.off[k])
                    e = s + int(self.ln[k])
                    leaf = self.blob_orig[s:e].tobytes().decode("utf-16-le")
                except Exception:
                    leaf = ""
                try:
                    if leaf and leaf != os.path.basename(base_root):
                        out = os.path.join(base_root, leaf)
                    else:
                        out = base_root
                except Exception:
                    out = base_root
            try:
                if _cache and len(self._path_cache) < 200000:
                    self._path_cache[k] = out
            except Exception:
                pass
            return out
        except Exception as exc:
            self._err(exc)
            return ""

    def _low_name(self, k):
        try:
            s = int(self.off[k])
            e = s + int(self.ln[k])
            return self.blob_low[s:e].tobytes().decode("utf-16-le")
        except Exception:
            return ""

    def _orig_name(self, k):
        try:
            s = int(self.off[k])
            e = s + int(self.ln[k])
            return self.blob_orig[s:e].tobytes().decode("utf-16-le")
        except Exception:
            return ""

    def build_trigrams(self):
        try:
            self._ensure_bloom()
            # Keep the public Bloom builder bounded. T5's full preparation
            # still runs through _start_search_preparation after publication.
            with self._lock:
                self._raw_bytes()
                self._starts_np()
            return self._bloom is not None
        except Exception as exc:
            self._err(exc)
            return False

    def _contiguous_names(self):
        layout = getattr(self, "_name_layout", None)
        if layout is None:
            layout = not len(self.off) or (int(self.off[0]) == 0 and
                not np.any(self.off[1:] != self.off[:-1] + self.ln[:-1]))
            self._name_layout = bool(layout)
        return self._name_layout

    def _ensure_bloom(self):
        try:
            n = int(self.off.shape[0])
        except Exception:
            return
        try:
            if self._bloom is not None and self._bloom.shape == ((n + self._block - 1) // self._block, BLOOM_U64):
                return
        except Exception:
            pass
        if n == 0:
            self._bloom = np.zeros((0, BLOOM_U64), dtype=np.uint64)
            return
        nb = (n + self._block - 1) // self._block
        if not self._contiguous_names():
            # Repointed names leave harmless old bytes in the arena. An all-hit
            # filter keeps search correct until a snapshot is compacted.
            self._bloom = np.full((nb, BLOOM_U64), np.iinfo(np.uint64).max, dtype=np.uint64)
            return
        try:
            with self._lock:
                generation = self._bloom_generation
                n = len(self)
                nb = (n + self._block - 1) // self._block
                blob = self.blob_low  # blobs are replaced, never edited in place
                # Snapshot replace-only columns; cast at chunk granularity.
                off = self.off
                ln = self.ln
        except Exception:
            return
        bloom = np.zeros((nb, BLOOM_U64), dtype=np.uint64)
        step = 64  # blocks per chunk keeps the temporaries small
        for b0 in range(0, nb, step):
            chunk_blocks = min(step, nb - b0)
            seen = np.zeros(chunk_blocks * BLOOM_BITS, dtype=bool)
            s0 = b0 * self._block
            e0 = min(n, (b0 + step) * self._block)
            lo = int(off[s0])
            hi = int(off[e0 - 1] + ln[e0 - 1])
            if hi - lo < 3:
                continue
            codes = blob[lo:hi].astype(np.uint32)
            lens = ln[s0:e0].astype(np.int64)
            name_of = np.repeat(np.arange(e0 - s0, dtype=np.int64), lens)
            ends = off[s0:e0].astype(np.int64) - lo + lens
            posn = np.arange(hi - lo, dtype=np.int64)
            ok = (posn + 2) < ends[name_of]
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                continue
            h = (codes[idx] * np.uint32(33) + codes[idx + 1]) * np.uint32(33) + codes[idx + 2]
            row = (name_of[idx] // self._block).astype(np.int64) * BLOOM_BITS
            for probe in _probe_arrays(h):
                seen[row + probe.astype(np.int64)] = True
            packed = np.packbits(seen.reshape(chunk_blocks, BLOOM_BITS),
                                 axis=1, bitorder="little")
            bloom[b0:b0 + chunk_blocks] = np.ascontiguousarray(packed).view(np.uint64)
        with self._lock:
            if generation == self._bloom_generation:
                self._bloom = bloom

    def _candidate_blocks(self, token):
        try:
            n = int(self.off.shape[0])
        except Exception:
            return np.zeros((0,), dtype=np.int64)
        nb = (n + self._block - 1) // self._block
        if nb == 0:
            return np.zeros((0,), dtype=np.int64)
        try:
            tok = str(token)
        except Exception:
            return np.arange(nb, dtype=np.int64)
        if len(tok) < 3:
            return np.arange(nb, dtype=np.int64)
        try:
            # Search never waits for construction: scan all blocks until ready.
            bloom = self._bloom
            if bloom is None or bloom.shape != (nb, BLOOM_U64):
                return np.arange(nb, dtype=np.int64)
        except Exception:
            return np.arange(nb, dtype=np.int64)
        try:
            hashes = _trigram_hashes_str(tok)
        except Exception:
            return np.arange(nb, dtype=np.int64)
        if not hashes:
            return np.arange(nb, dtype=np.int64)
        req = {}
        for h in hashes[:12]:
            for p in _probes(h):
                w = int(p) // 64
                m = np.uint64(1) << np.uint64(int(p) % 64)
                req[w] = req.get(w, np.uint64(0)) | m
        try:
            # Only Bloom candidates are cached, never the capped/ranked hits.
            # Extending one token can only remove candidates. Invalidation drops
            # the cache on any name/layout mutation, including same-size renames.
            cached = self._typing_blocks
            found = (cached[2].copy() if cached is not None and
                     cached[0] is bloom and tok.startswith(cached[1]) else
                     np.arange(nb, dtype=np.int64))
            for w, m in req.items():
                found = found[(bloom[found, int(w)] & m) == m]
                if not found.size:
                    break
            self._typing_blocks = (bloom, tok, found.copy())
            return found
        except Exception:
            return np.arange(nb, dtype=np.int64)

    def _raw_bytes(self):
        try:
            if self._raw is not None:
                return self._raw
            if isinstance(self.blob_low, _NameArena):
                self._raw = _ArenaBytes(self.blob_low)
                return self._raw
            if isinstance(self.blob_low, np.memmap):
                self._raw = _MappedBytes(self.blob_low)
                return self._raw
            raw = self.blob_low.base
            if not isinstance(raw, bytes) or len(raw) != self.blob_low.nbytes:
                raw = self.blob_low.tobytes()
            self._raw = raw
            return raw
        except Exception:
            return b""

    def _starts_np(self):
        """Byte offset (in the lowercase UTF-16 blob) of every entry; cached, rebuilt when the arrays change."""
        n = int(self.off.shape[0])
        cache = getattr(self, "_starts_cache", None)
        if cache is None or cache[0] != n or cache[1] is not self.off:
            dtype = np.uint32 if not n or int(self.off.max()) <= (2 ** 32 - 1) // 2 else np.uint64
            cache = (n, self.off, self.off.astype(dtype) * dtype(2))
            self._starts_cache = cache
        return cache[2]

    def _name_match_ids(self, tokens):
        """Entry ids whose lowercase NAME contains every token (Bloom block filter + one C-speed find per block)."""
        try:
            n = int(self.off.shape[0])
            if n == 0 or not tokens:
                return []
            raw = self._raw_bytes()
            if not self._contiguous_names():
                return self._arena_name_match_ids(tokens, raw)
            enc = [_lower_codes(t).tobytes() for t in tokens]
            if not all(enc):
                return []
            if len(tokens) == 1:
                ids = self._operator_name_ids(tokens[0])
                return ids[self._reachable_rows()[ids]].tolist()
            per_blocks = [self._candidate_blocks(t) for t in tokens]
            sizes = [int(b.size) for b in per_blocks]
            if min(sizes) == 0:
                return []
            drv_i = sizes.index(min(sizes))  # the most selective token (fewest Bloom-passing blocks) drives the scan
            drv = enc[drv_i]
            blocks = per_blocks[drv_i]
            starts = self._starts_np()
            positions = []
            # Coalesce adjacent passing blocks, especially for short/broad tokens.
            cuts = np.flatnonzero(np.diff(blocks) != 1) + 1
            run_first = blocks[np.r_[0, cuts]]
            run_last = blocks[np.r_[cuts - 1, blocks.size - 1]]
            for first_block, last_block in zip(run_first.tolist(), run_last.tolist()):
                first = int(first_block) * self._block
                last = min(n, (int(last_block) + 1) * self._block) - 1
                lo = int(starts[first])
                hi = int(starts[last]) + int(self.ln[last]) * 2
                at = raw.find(drv, lo, hi)
                while at >= 0:
                    if not at & 1:
                        positions.append(at)
                    at = raw.find(drv, at + 2, hi)
            if not positions:
                return []
            positions = np.asarray(positions, dtype=np.int64)
            slots = np.searchsorted(starts, positions, side="right") - 1
            valid = slots >= 0
            slots, positions = slots[valid], positions[valid]
            # A byte match may straddle two adjacent names in the shared blob.
            valid = positions + len(drv) <= starts[slots] + self.ln[slots].astype(np.int64) * 2
            ids = np.unique(slots[valid])
            ids = ids[(ids >= 0) & (ids < n)]
            ids = ids[self._reachable_rows()[ids]]
            if len(tokens) > 1:
                keep = []
                for k in ids.tolist():
                    nm = self._low_name(k)
                    if all(str(t) in nm for t in tokens):
                        keep.append(k)
                return keep
            return ids.tolist()
        except Exception as exc:
            self._err(exc)
            return []

    def _arena_name_match_ids(self, tokens, raw):
        """Search an append-only name arena, ignoring retired/gap bytes."""
        order = getattr(self, "_offset_order", None)
        if order is None:
            order = np.argsort(self.off, kind="stable")
            self._offset_order = order
        starts = self.off[order] * np.uint64(2)
        needle = _lower_codes(tokens[0]).tobytes()
        if not needle:
            return []
        positions = []
        at = raw.find(needle)
        while at >= 0:
            if not at & 1:
                positions.append(at)
            at = raw.find(needle, at + 1)
        if not positions:
            return []
        positions = np.asarray(positions, dtype=np.int64)
        slots = np.searchsorted(starts, positions, side="right") - 1
        valid = slots >= 0
        positions, slots = positions[valid], slots[valid]
        rows = order[slots]
        valid = positions + len(needle) <= (self.off[rows] + self.ln[rows]) * 2
        ids = np.unique(rows[valid])
        ids = ids[self._reachable_rows()[ids]]
        if len(tokens) > 1:
            return [int(k) for k in ids if all(
                str(t) in self._low_name(k) for t in tokens)]
        return ids.tolist()

    def _children_csr(self):
        """Children adjacency (CSR over the parent array); cached."""
        n = int(self.off.shape[0])
        cache = getattr(self, "_csr_cache", None)
        if cache is None or cache[0] != n or cache[1] is not self.parent:
            par = self.parent.astype(np.int64)
            order = np.argsort(par, kind="stable")
            ptr = np.searchsorted(par[order], np.arange(-1, n + 1), side="left")  # ptr[j + 1] = first row with parent >= j
            cache = (n, self.parent, order, ptr)
            self._csr_cache = cache
        return cache[2], cache[3]

    def _descendants(self, dir_ids, cap=200000):
        """All entries below the given directories (BFS over the CSR, cost proportional to the result)."""
        order, ptr = self._children_csr()
        out = []
        total = 0
        frontier = np.asarray(dir_ids, dtype=np.int64)
        for _ in range(64):
            if frontier.size == 0:
                break
            lo = ptr[frontier + 1]
            hi = ptr[frontier + 2]
            counts = hi - lo
            m = int(counts.sum())
            if m == 0:
                break
            rows = np.repeat(lo - np.cumsum(counts) + counts, counts) + np.arange(m)
            children = order[rows]
            out.append(children)
            total += m
            if total > cap:
                return None
            frontier = children[(self.flags[children] & FLAG_DIR) != 0]
        return np.concatenate(out) if out else np.zeros((0,), dtype=np.int64)

    def _path_only_ids(self, toks, exclude, need):
        """Entries where some token is only matched by an ANCESTOR folder name (e.g. 'invoices 2024' -> invoices/2024-03.pdf)."""
        try:
            if not np.any(self.flags & FLAG_DIR):
                return []
            n = len(self)
            cand = None
            for t in toks:
                own = np.asarray(self._name_match_ids([t]), dtype=np.int64)
                mask = np.zeros(n, dtype=bool)
                if own.size:
                    mask[own] = True
                    dirs = own[(self.flags[own] & FLAG_DIR) != 0]
                    if dirs.size:
                        d = self._descendants(dirs.tolist())
                        if d is None:
                            return []
                        mask[np.asarray(d, dtype=np.int64)] = True
                cand = mask if cand is None else (cand & mask)
            if cand is None:
                return []
            exc = np.asarray(sorted(exclude), dtype=np.int64) if len(exclude) else np.empty(0, dtype=np.int64)
            if exc.size:
                cand[exc[(exc >= 0) & (exc < n)]] = False
            return np.flatnonzero(cand)[:need].tolist()
        except Exception as exc:
            self._err(exc)
            return []

    def _rank_one(self, low, qfull, tokens, full_low):
        try:
            if low == qfull:
                return 0
            if low.startswith(qfull):
                return 1
            hit = False
            for t in tokens:
                if low.startswith(t):
                    hit = True
                    break
                i = low.find(t, 1)
                while i >= 0:
                    if low[i - 1] in _WORD_SEPS:
                        hit = True
                        break
                    i = low.find(t, i + 1)
                if hit:
                    break
            if hit:
                return 2
            if all(t in low for t in tokens):
                return 3
            if full_low and all(t in full_low for t in tokens):
                return 4
            return 9
        except Exception:
            return 9

    def search(self, query, limit=50, only_prefixes=None, kinds=None):
        try:
            return self._search_inner(query, limit, only_prefixes, kinds)
        except Exception as exc:
            self._err(exc)
            return []

    def _fuzzy_path(self, k):
        """Reconstruct a result path without populating the identity/path caches.

        Caller holds _lock. Strings and row IDs in the fuzzy cache are detached
        from all index columns; fuzzy matching never receives an array view.
        """
        parts = []
        cur = k
        visited = set()
        while cur >= 0:
            if cur in visited or cur >= len(self) or self.flags[cur] & FLAG_TOMB:
                return ""
            visited.add(cur)
            root_pos = int(self._root_of[cur])
            if not 0 <= root_pos < len(self._roots):
                return ""
            parent = int(self.parent[cur])
            if parent < 0:
                root = self._roots[root_pos]
                leaf = self._orig_name(cur)
                if not parts and leaf and leaf != os.path.basename(root):
                    parts.append(leaf)
                return os.path.join(root, *reversed(parts))
            parts.append(self._orig_name(cur))
            cur = parent
        return ""

    def _append_fuzzy(self, hits, query, limit, only_prefixes=None, kinds=None):
        # The common exact path does not import, parse, read settings, or build
        # arrays. Keep the exact-only LRU independent of the runtime switch.
        if len(hits) >= 3:
            return hits
        try:
            lim = max(1, min(int(limit), 200))
        except Exception:
            lim = 50
        if len(hits) >= lim:
            return hits
        try:
            clean, spec = _parse_search_query(str(query or ""))
            tokens = clean.split()
            if (spec is not None or not tokens or max(map(len, tokens)) < 4
                    or any(c in clean for c in (":", "*", "?", "[", "]", "^", "$", "|",
                                               "(", ")", "{", "}", _BS, _SL))):
                return hits
            import localdocs_settings
            if not localdocs_settings.get().get("fuzzy_names", True):
                return hits
            if os.environ.get("LOCALDOCS_FUZZY_NAMES", "1") == "0":
                return hits
            prefixes = None
            if only_prefixes is not None:
                prefixes = [os.path.normcase(os.path.normpath(str(p))) for p in only_prefixes]
                if not prefixes:
                    return hits
            kindset = {str(k).lower() if str(k).startswith(".") else "." + str(k).lower()
                       for k in (kinds or ())}
            try:
                fuzzy_cap = int(os.environ.get('LOCALDOCS_FUZZY_MAX', '150000'))
            except ValueError:
                fuzzy_cap = 150000
            if len(self) > fuzzy_cap:
                return hits  # typo-tolerant fallback is a small-index feature: its cache costs seconds and ~1 GB at volume scale
            import localdocs_fuzzy as fuzzy
            # Large indexes: building the fuzzy name cache costs seconds (6-9 s measured at 530k names) - never on the
            # query path. Serve exact hits now and prepare the cache on a background thread; small indexes stay synchronous.
            if len(self) > 200000 and threading.current_thread().name != "localdocs-fuzzy-warm":
                gen_now = (self._bloom_generation, self._turbo_layout, self._turbo_revision, len(self))
                cache0 = getattr(self, "_fuzzy_cache", None)
                if cache0 is None or cache0["generation"] != gen_now:
                    if (not getattr(self, "_fuzzy_warming", False) and
                            time.monotonic() - getattr(self, "_fuzzy_warm_at", -1e9) >= 60.0):  # live volumes change constantly: rebuild at most once a minute
                        self._fuzzy_warming = True
                        self._fuzzy_warm_at = time.monotonic()
                        def _warm_fuzzy(self=self, q=query, op=only_prefixes, kd=kinds):
                            try:
                                self._append_fuzzy([], q, 1, op, kd)
                            except Exception as exc:
                                self._err(exc)
                            finally:
                                self._fuzzy_warming = False
                        threading.Thread(target=_warm_fuzzy, name="localdocs-fuzzy-warm", daemon=True).start()
                    return hits
            if threading.current_thread().name == "localdocs-fuzzy-warm":
                # Background preparation: never hold the index lock while converting millions of names.
                with self._lock:
                    gen = (self._bloom_generation, self._turbo_layout, self._turbo_revision, len(self))
                    c0 = getattr(self, "_fuzzy_cache", None)
                    if c0 is not None and c0["generation"] == gen:
                        return hits
                    rows = np.flatnonzero((self.flags & FLAG_TOMB) == 0)
                names = []
                for lo in range(0, len(rows), 20000):
                    names.extend(ntpath.basename(self._orig_name(int(k))) for k in rows[lo:lo + 20000])
                    time.sleep(0.002)  # let query threads in between chunks
                with self._lock:
                    if gen == (self._bloom_generation, self._turbo_layout, self._turbo_revision, len(self)):
                        self._fuzzy_cache = {"generation": gen, "rows": rows, "names": names, "stems": None,
                                             "arrays": {}, "runners": {}}
                return hits
            with self._lock:
                generation = (self._bloom_generation, self._turbo_layout, self._turbo_revision, len(self))
                cache = getattr(self, "_fuzzy_cache", None)
                if cache is None or cache["generation"] != generation:
                    rows = np.flatnonzero((self.flags & FLAG_TOMB) == 0)
                    names = [ntpath.basename(self._orig_name(int(k))) for k in rows]
                    cache = {"generation": generation, "rows": rows,
                             "names": names, "stems": None, "arrays": {}, "runners": {}}
                    self._fuzzy_cache = cache
                # A name query without a suffix should also find report.pdf
                # from 'reprot'; an explicit suffix is compared in full.
                mode = "names" if ntpath.splitext(clean)[1] else "stems"
                if cache[mode] is None:
                    cache[mode] = [ntpath.splitext(name)[0] for name in cache["names"]]
                names = cache[mode]
                if mode not in cache["runners"]:
                    # Execute F1's unmodified functions with a private cache.
                    # Swapping fuzzy._CACHE would race with direct F1 callers.
                    # Only _get_index writes globals; its helpers are pure.
                    namespace = dict(vars(fuzzy))
                    namespace["_CACHE"] = {"ref": None, "id": None, "n": -1, "data": None}
                    namespace["_get_index"] = types.FunctionType(
                        fuzzy._get_index.__code__, namespace, "_get_index",
                        fuzzy._get_index.__defaults__)
                    cache["runners"][mode] = types.FunctionType(
                        fuzzy.fuzzy_search.__code__, namespace, "fuzzy_search",
                        fuzzy.fuzzy_search.__defaults__)
                    cache["arrays"][mode] = namespace["_CACHE"]
                # Filter before the final limit: denied/deleted/duplicate
                # rows must never crowd an allowed match out of the result.
                matches = cache["runners"][mode](names, clean, limit=len(names))
                out = list(hits)
                seen = {os.path.normcase(os.path.normpath(h["path"])) for h in hits}
                for row, distance in matches:
                    k = int(cache["rows"][row])
                    if int(self.flags[k]) & FLAG_TOMB:
                        continue
                    name = self._orig_name(k)
                    if kindset and os.path.splitext(name)[1].lower() not in kindset:
                        continue
                    path = self._fuzzy_path(k)
                    if not path:
                        continue
                    pk = os.path.normcase(os.path.normpath(path))
                    if pk in seen:
                        continue
                    if prefixes is not None and not any(
                            (self._contains_path(pk, p) if self._journal_state
                             else pk == p or pk.startswith(p + os.sep)) for p in prefixes):
                        continue
                    root_id = int(self._root_of[k])
                    root = self._roots[root_id] if 0 <= root_id < len(self._roots) else ""
                    out.append({"path": path, "name": name,
                                "folder": os.path.dirname(path),
                                "size": max(0, int(self.size[k])),
                                "mtime": max(0.0, float(self.mtime[k])),
                                "flags": int(self.flags[k]), "root": root,
                                "fuzzy": True, "distance": int(distance)})
                    seen.add(pk)
                    if len(out) >= lim:
                        break
                return out
        except Exception as exc:
            self._err(exc)
            return hits

    def _search_inner(self, query, limit, only_prefixes, kinds, candidate_rows=None):
        try:
            clean, spec = _parse_search_query(str(query or ""))
            toks = clean.lower().split()
        except Exception:
            return []
        if not toks and spec is None:
            return []
        qfull = " ".join(toks)
        try:
            lim = max(1, min(int(limit), 200))
        except Exception:
            lim = 50
        prefixes = None
        if only_prefixes is not None:
            prefixes = []
            try:
                for p in only_prefixes:
                    try:
                        prefixes.append(os.path.normcase(os.path.normpath(str(p))))
                    except Exception:
                        pass
            except Exception:
                return []
            if not prefixes:
                return []
        kindset = None
        if kinds:
            kindset = set()
            try:
                for k in kinds:
                    try:
                        s = str(k).lower()
                        if s and not s.startswith("."):
                            s = "." + s
                        kindset.add(s)
                    except Exception:
                        pass
            except Exception:
                kindset = None
        try:
            n = int(self.off.shape[0])
        except Exception:
            return []
        if n == 0:
            return []
        if spec is not None:
            if (spec.get("sort") or any(spec.get(k) is not None for k in
                    ("min_size", "max_size", "min_t", "max_t"))):
                self._materialize_metadata()
            return self._search_spec(spec, toks, limit, prefixes, kindset, candidate_rows)
        try:
            key = (str(query), lim, tuple(prefixes) if prefixes else None,
                   tuple(sorted(kindset)) if kindset else None)
            hit = self._lru.get(key) if candidate_rows is None else None
            if hit is not None and spec is None:
                try:
                    self._lru.move_to_end(key)
                except Exception:
                    pass
                return [dict(r) for r in hit]
        except Exception:
            pass
        if not self._dir_use and not getattr(self, '_operator_has_expansion', False):
            column_hits = self._search_plain_columns(toks, lim, prefixes, kindset, candidate_rows)
            if column_hits is not None:
                self._lru[key] = [dict(h) for h in column_hits]
                while len(self._lru) > _LRU_SIZE:
                    self._lru.popitem(last=False)
                return column_hits
        if candidate_rows is not None and spec is None and len(toks) > 1:
            # Historical ancestor fallback has its own candidate budget/order.
            return self._search_inner(query, limit, only_prefixes, kinds)
        try:
            # Filtered queries enumerate the complete snapshot, including path-only
            # matches and extensionless files, before any candidate budget.
            name_ids = (self._name_match_ids(toks) if candidate_rows is None else
                        [k for k in candidate_rows if all(t in self._low_name(k) for t in toks)])
            name_set = set(name_ids)
        except Exception:
            name_ids = []
            name_set = set()
        cands = list(name_ids)
        if candidate_rows is None and spec is None and len(cands) < lim and len(toks) > 1:
            cands.extend(self._path_only_ids(toks, name_set, lim * 4))
        # Phase 1: filtered queries check complete records before candidate budgets;
        # ordinary queries retain the cheap name-only pre-score.
        pre = []
        for k in cands:
            try:
                if not self._reachable_rows()[k]:
                    continue
                if spec is not None and int(self.flags[k]) & FLAG_DIR:
                    continue
                low = self._low_name(k)
                rank0 = self._rank_one(low, qfull, toks, None)
                if rank0 > 4:
                    rank0 = 4
                orig = self._orig_name(k)
                ext = os.path.splitext(orig)[1].lower()
                if kindset and ext not in kindset:
                    continue
                if spec is not None:
                    p = self.path_of(k)
                    if prefixes is not None and not any(self._contains_path(p, root) for root in prefixes):
                        continue
                    if toks and not all(t in p.lower() for t in toks):
                        continue
                    hit = {"path": p, "name": orig, "size": int(self.size[k]),
                           "mtime": float(self.mtime[k])}
                    from localdocs_filters import matches
                    if not matches(spec, hit):
                        continue
                    if hit["size"] >= 0 and hit["mtime"] >= 0:
                        self.size[k], self.mtime[k] = hit["size"], hit["mtime"]
                fl = int(self.flags[k])
                score0 = float(rank0) * 100.0 + min(max(int(self._depth[k]), 0), 20) * 0.5
                if ext in _DOC_EXT:
                    score0 -= 8.0
                if (fl & FLAG_JUNK) or (ext in _PENALTY_EXT):
                    score0 += 20.0
                if fl & FLAG_HIDDEN:
                    score0 += 10.0
                pre.append((score0, len(low), k))
            except Exception:
                continue
        pre.sort()
        # phase 2: full records (path, use-based boosts) only for the best candidates, in chunks until enough survive
        scored = []
        want = max(lim * 3, 60)
        pos = 0
        while pos < len(pre):
            for score0, _ln, k in pre[pos:pos + 200]:
                try:
                    low = self._low_name(k)
                    orig = self._orig_name(k)
                    p = self.path_of(k)
                    if prefixes:
                        pk = os.path.normcase(os.path.normpath(p))
                        if not any((self._contains_path(pk, q) if self._journal_state
                                    else pk == q or pk.startswith(q + os.sep)) for q in prefixes):
                            continue
                    rank = self._rank_one(low, qfull, toks, p.lower())
                    if rank > 4:
                        continue
                    ext = os.path.splitext(orig)[1].lower()
                    fl = int(self.flags[k])
                    dep = int(self._depth[k])
                except Exception:
                    continue
                score = float(rank) * 100.0
                if ext in _DOC_EXT:
                    score -= 8.0
                if (fl & FLAG_JUNK) or (ext in _PENALTY_EXT):
                    score += 20.0
                if fl & FLAG_HIDDEN:
                    score += 10.0
                score += min(max(dep, 0), 20) * 0.5
                folder = os.path.dirname(p)
                try:
                    score -= min(float(self._ext_use.get(ext, 0)), 5.0) * 0.4
                    score -= min(float(self._dir_use.get(os.path.normcase(folder), 0)), 5.0) * 0.2
                except Exception:
                    pass
                try:
                    mt = float(self.mtime[k])
                except Exception:
                    mt = -1.0
                scored.append((score, len(p), -mt if mt >= 0 else 0.0, k, p, orig, folder, fl))
            pos += 200
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        top = scored[:min(_TOP_STAT, max(lim, 20))]
        for item in top:
            k = item[3]
            try:
                if float(self.mtime[k]) < 0:
                    p = item[4]
                    try:
                        st = os.stat(p)
                        sz, mt = int(st.st_size), float(st.st_mtime)
                    except OSError:
                        continue
                    with self._lock:
                        try:
                            self.size[k] = np.int64(sz)
                            self.mtime[k] = np.float64(mt)
                            self._turbo_touch([k])
                        except Exception:
                            pass
            except Exception:
                pass
        rescored = []
        for score, plen, negmt, k, p, orig, folder, fl in scored:
            try:
                mt = float(self.mtime[k])
            except Exception:
                mt = -1.0
            try:
                sz = int(self.size[k])
                if sz < 0:
                    sz = 0
            except Exception:
                sz = 0
            if mt < 0:
                mt = 0.0
            rescored.append((score, plen, -mt, k, p, orig, folder, fl, sz, mt))
        rescored.sort(key=lambda t: (t[0], t[1], t[2]))
        out = []
        for score, plen, negmt, k, p, orig, folder, fl, sz, mt in rescored[:lim]:
            try:
                root = str(self._roots[int(self._root_of[k])])
            except Exception:
                root = ""
            out.append({"path": p, "name": orig, "folder": folder, "size": sz,
                        "mtime": mt, "flags": int(fl), "score": float(score),
                        "root": root})
        try:
            if spec is None and candidate_rows is None:
                self._lru[key] = [dict(r) for r in out]
            while len(self._lru) > _LRU_SIZE:
                self._lru.popitem(last=False)
        except Exception:
            pass
        return out

    def _offset_rows(self):
        """Row ids in arena order, shared by operator lookup and renames."""
        order = getattr(self, '_offset_order', None)
        if order is None:
            order = np.argsort(self.off, kind='stable')
            self._offset_order = order
        return order

    def _operator_extensions(self):
        """Generation-local suffix ids, extracted in bulk from UTF-16 columns."""
        cached = getattr(self, "_operator_ext", None)
        if cached is not None:
            return cached
        n = len(self)
        starts = self.off.astype(np.int64)
        order = self._offset_rows()
        ends = starts + self.ln
        dots = np.flatnonzero(self.blob_low == ord("."))
        last = np.full(n, -1, dtype=np.int64)
        if dots.size:
            slots = np.searchsorted(starts[order], dots, side="right") - 1
            owners = order[np.maximum(slots, 0)]
            good = (slots >= 0) & (dots < ends[owners])
            np.maximum.at(last, owners[good], dots[good])
        lengths = np.where(last >= 0, ends - last - 1, 0)
        ids = np.zeros(n, dtype=np.int32)
        values = [""]
        # One packed-key unique pass for ordinary suffixes. Long suffixes are
        # grouped only in the exceptional tail, avoiding max-length*N scratch.
        common = np.flatnonzero((lengths > 0) & (lengths <= 32))
        if common.size:
            keys = np.zeros((common.size, 33), dtype=np.uint16)
            keys[:, 0] = lengths[common]
            for first in range(0, common.size, 65536):
                batch = common[first:first + 65536]
                columns = np.arange(32)
                valid = columns < lengths[batch, None]
                positions = last[batch, None] + 1 + columns
                keys[first:first + len(batch), 1:] = np.where(
                    valid, self.blob_low[np.minimum(positions, len(self.blob_low) - 1)], 0)
            records = keys.view(np.dtype(('V', 66))).reshape(-1)
            unique, inverse = np.unique(records, return_inverse=True)
            packed = unique.view(np.uint16).reshape(-1, 33)
            values.extend(row[1:1 + int(row[0])].tobytes().decode(
                'utf-16-le', 'surrogatepass') for row in packed)
            ids[common] = inverse + 1
        for length in np.unique(lengths[lengths > 32]):
            length = int(length)
            rows = np.flatnonzero(lengths == length)
            units = self.blob_low[last[rows, None] + 1 + np.arange(length)]
            records = np.ascontiguousarray(units).view(
                np.dtype(('V', length * 2))).reshape(-1)
            unique, inverse = np.unique(records, return_inverse=True)
            offset = len(values)
            values.extend(v.tobytes().decode('utf-16-le', 'surrogatepass') for v in unique)
            ids[rows] = inverse + offset
        # Python lower() may expand a BMP character; the compact lowercase blob
        # deliberately keeps offsets. Repair just those exceptional suffixes.
        # U+0130 is stored as i by _LOWER, so detect it in the original column.
        unusual = np.flatnonzero(self.blob_orig == 0x0130)
        self._operator_has_expansion = bool(unusual.size) or any(
            "\u0130" in root for root in self._roots)
        if unusual.size:
            slots = np.searchsorted(starts[order], unusual, side="right") - 1
            rows = np.unique(order[slots[slots >= 0]])
            table = {v: i for i, v in enumerate(values)}
            for k in rows:
                name = self._orig_name(int(k))
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext not in table:
                    table[ext] = len(values)
                    values.append(ext)
                ids[k] = table[ext]
        # splitext differs from rsplit for dotfiles. Keep both semantics.
        split_ids = ids.copy()
        trailing = (last >= 0) & (lengths == 0)
        if trailing.any():
            # A dot cannot be a last-dot suffix, so it is a safe sentinel
            # for splitext('name.') == ('name', '.').
            split_ids[trailing] = len(values)
            values.append(".")
        leading = np.flatnonzero((self.ln > 0) &
                                (self.blob_orig[np.minimum(starts, max(0, len(self.blob_orig) - 1))]
                                 == ord("."))) if self.blob_orig.size else []
        for k in leading:
            ext = os.path.splitext(self._orig_name(int(k)))[1].lower()
            if not ext:
                split_ids[k] = 0
        # These are derived columns, never aliases of published metadata.
        dtype = np.uint16 if len(values) <= 65536 else np.uint32
        ids = ids.astype(dtype)
        split_ids = ids if np.array_equal(ids, split_ids) else split_ids.astype(dtype)
        self._operator_ext = (ids, split_ids, values)
        return self._operator_ext

    def _position_rows(self, positions):
        """Map arena positions to names without log(N) work per occurrence.

        Ordinary arenas use a row checkpoint per 32 UTF-16 units (about
        one sixteenth of the name bytes). A lookup needs at most 32 advances;
        typical Windows names need one or two. Gapped arenas use sorted offsets.
        This cache is derived, generation-local, and never persisted.
        """
        if not self._contiguous_names():
            order = self._offset_rows()
            slots = np.searchsorted(self.off[order], positions, side='right') - 1
            return np.where(slots >= 0, order[np.maximum(slots, 0)], -1)
        checkpoint = getattr(self, '_operator_positions', None)
        if checkpoint is None:
            checkpoint = np.searchsorted(self.off,
                np.arange(0, len(self.blob_low), 32, dtype=np.uint64),
                side='right') - 1
            checkpoint = checkpoint.astype(np.int32 if len(self) < 2 ** 31 else np.int64)
            self._operator_positions = checkpoint
        if not positions.size:
            return np.empty(0, dtype=np.int64)
        owners = checkpoint[positions // 32].astype(np.int64)
        for _ in range(32):
            following = owners + 1
            advance = (following < len(self)) & (self.off[np.minimum(following, len(self) - 1)] <= positions)
            if not advance.any():
                return owners
            owners[advance] += 1
        # Runs of empty names can contain more than 32 rows in 32 units.
        return np.searchsorted(self.off, positions, side='right') - 1

    def _token_occurrences(self, token, rows=None):
        """Vector UTF-16 comparisons, optionally on already filtered names."""
        needle = _lower_codes(token)
        empty = np.empty(0, dtype=np.int64)
        if not needle.size or not len(self):
            return empty, empty
        owners_out, positions_out = [], []
        width = len(needle)
        if rows is None and width >= 3 and self._contiguous_names():
            blocks = self._candidate_blocks(token)
            total_blocks = (len(self) + self._block - 1) // self._block
            if len(blocks) < max(1, total_blocks // 4):
                raw, starts = self._raw_bytes(), self._starts_np()
                encoded, found = needle.tobytes(), []
                cuts = np.flatnonzero(np.diff(blocks) != 1) + 1
                if len(blocks):
                    firsts = blocks[np.r_[0, cuts]]
                    lasts = blocks[np.r_[cuts - 1, len(blocks) - 1]]
                    for first, last in zip(firsts, lasts):
                        a = int(first) * self._block
                        b = min(len(self), (int(last) + 1) * self._block) - 1
                        stop = int(starts[b]) + int(self.ln[b]) * 2
                        at = raw.find(encoded, int(starts[a]), stop)
                        while at >= 0:
                            if not at & 1:
                                found.append(at // 2)
                            at = raw.find(encoded, at + 2, stop)
                if not found:
                    return empty, empty
                positions = np.asarray(found, dtype=np.int64)
                owners = np.searchsorted(self.off, positions, side='right') - 1
                valid = owners >= 0
                owners, positions = owners[valid], positions[valid]
                valid = positions + width <= self.off[owners] + self.ln[owners]
                return owners[valid], positions[valid]
        if rows is not None:
            rows = np.asarray(rows, dtype=np.int64)
            # Work proportional to selected name bytes, not the volume blob.
            for first in range(0, len(rows), 32768):
                batch = rows[first:first + 32768]
                counts = np.maximum(self.ln[batch].astype(np.int64) - width + 1, 0)
                total = int(counts.sum())
                if not total:
                    continue
                owners = np.repeat(batch, counts)
                positions = (np.repeat(self.off[batch].astype(np.int64) -
                    (np.cumsum(counts) - counts), counts) + np.arange(total))
                for j, unit in enumerate(needle):
                    valid = self.blob_low[positions + j] == unit
                    positions, owners = positions[valid], owners[valid]
                    if not positions.size:
                        break
                owners_out.append(owners)
                positions_out.append(positions)
        else:
            for first in range(0, max(0, len(self.blob_low) - width + 1), 1048576):
                stop = min(len(self.blob_low) - width + 1, first + 1048576)
                positions = np.flatnonzero(self.blob_low[first:stop] == needle[0]) + first
                for j in range(1, width):
                    positions = positions[self.blob_low[positions + j] == needle[j]]
                    if not positions.size:
                        break
                if not positions.size:
                    continue
                owners = self._position_rows(positions)
                valid = owners >= 0
                owners, positions = owners[valid], positions[valid]
                valid = positions + width <= self.off[owners] + self.ln[owners]
                owners_out.append(owners[valid])
                positions_out.append(positions[valid])
        if not owners_out:
            return empty, empty
        return np.concatenate(owners_out), np.concatenate(positions_out)

    def _operator_name_ids(self, token):
        if not token:
            return np.arange(len(self), dtype=np.int64)
        owners, _ = self._token_occurrences(token)
        return np.unique(owners)

    def _name_rank_columns(self, tokens, all_ranks=False):
        """Complete name tiers: exact, prefix, word boundary, substring."""
        if len(tokens) == 1:
            needle = _lower_codes(tokens[0])
            if needle.size == 1 and self._contiguous_names():
                return self._single_unit_rank_columns(needle[0], all_ranks)
        ranks = np.full(len(self), 3, dtype=np.uint8)
        keep = (self.flags & FLAG_TOMB) == 0
        separators = np.asarray([ord(c) for c in _WORD_SEPS], dtype=np.uint16)
        selected = None
        ordered_tokens = sorted(tokens, key=lambda token: len(self._candidate_blocks(token)))
        for token in ordered_tokens:
            owners, positions = self._token_occurrences(token, rows=selected)
            selected = np.unique(owners)
            present = np.zeros(len(self), dtype=bool)
            present[owners] = True
            keep &= present
            boundary = (positions == self.off[owners]) | np.isin(
                self.blob_low[np.maximum(positions - 1, 0)], separators)
            ranks[owners[boundary]] = 2
        ids = np.flatnonzero(keep)
        if len(tokens) != 1:
            # Prefix/exact tiers only need the beginning of surviving names.
            # A second full-arena phrase scan repeats token matching and owner
            # lookup, even when no name can start with the complete phrase.
            phrase = _lower_codes(' '.join(tokens))
            prefix = (ids[self.ln[ids] >= len(phrase)] if phrase.size else
                      np.empty(0, dtype=np.int64))
            for j, unit in enumerate(phrase):
                prefix = prefix[self.blob_low[self.off[prefix].astype(np.int64) + j] == unit]
                if not prefix.size:
                    break
        else:
            prefix = owners[positions == self.off[owners]]
        ranks[prefix] = 1
        width = len(_lower_codes(' '.join(tokens)))
        ranks[prefix[self.ln[prefix] == width]] = 0
        if all_ranks:
            ranks[~keep] = 4
            return ranks
        return ids, ranks[ids]

    def _single_unit_rank_columns(self, unit, all_ranks=False):
        """Reduce all occurrences by name, without mapping each hit to a row.

        Broad one-unit tokens can occur several times per name. A segmented
        minimum scans the contiguous arena once, retaining every name and tier;
        it needs neither an occurrence-owner vector nor repeated checkpoints.
        Empty rows are excluded from reduceat (duplicate/end offsets are not
        valid segments). Gapped/renamed arenas use the general matcher above.
        """
        ranks = np.full(len(self), 4, dtype=np.uint8)
        rows = np.flatnonzero(self.ln > 0)
        if rows.size:
            starts = self.off[rows].astype(np.int64)
            positions = np.flatnonzero(self.blob_low == unit)
            tiers = np.full(len(self.blob_low), 4, dtype=np.uint8)
            tiers[positions] = 3
            separators = np.asarray([ord(c) for c in _WORD_SEPS], dtype=np.uint16)
            boundary = np.isin(self.blob_low[np.maximum(positions - 1, 0)], separators)
            tiers[positions[boundary]] = 2
            ranks[rows] = np.minimum.reduceat(tiers, starts)
            prefix = rows[self.blob_low[starts] == unit]
            ranks[prefix] = 1
            ranks[prefix[self.ln[prefix] == 1]] = 0
        ranks[(self.flags & FLAG_TOMB) != 0] = 4
        if all_ranks:
            return ranks
        ids = np.flatnonzero(ranks < 4)
        return ids, ranks[ids]

    def _selected_path_lengths(self, rows):
        """Rank selective hits without allocating volume-sized derived columns."""
        cached = getattr(self, '_operator_lengths', None)
        if cached is not None:
            return cached[rows]
        result = np.empty(len(rows), dtype=np.int64)
        for slot, row in enumerate(rows):
            cur, parts = int(row), []
            for _ in range(4096):
                parent = int(self.parent[cur])
                if parent < 0:
                    break
                try:
                    name = self.blob_orig[int(self.off[cur]):int(self.off[cur]) + int(self.ln[cur])].tobytes().decode('utf-16-le')
                except UnicodeError:
                    name = ''
                parts.append(name)
                cur = parent
            root = str(self._roots[int(self._root_of[cur])])
            if not parts:
                try:
                    name = self.blob_orig[int(self.off[cur]):int(self.off[cur]) + int(self.ln[cur])].tobytes().decode('utf-16-le')
                except UnicodeError:
                    name = ''
                if name and name != os.path.basename(root):
                    parts.append(name)
            length, trailing = len(root), bool(root.endswith((os.sep, os.altsep or os.sep)))
            for name in reversed(parts):
                if not name:
                    continue
                if os.path.isabs(name) or os.path.splitdrive(name)[0]:
                    # Exceptional synthetic segments must retain os.path semantics.
                    length = len(self.path_of(int(row)))
                    break
                length += len(name) + (0 if trailing or not length else 1)
                trailing = name.endswith((os.sep, os.altsep or os.sep))
            result[slot] = length
        return result

    def _search_plain_columns(self, tokens, limit, prefixes, kindset, candidate_rows=None):
        """Rank every eligible name before selecting; decode only final rows."""
        if candidate_rows is None:
            ids, ranks = self._name_rank_columns(tokens)
        else:
            ids = np.asarray([k for k in candidate_rows if all(
                t in self._low_name(k) for t in tokens)], dtype=np.int64)
            ranks = np.asarray([self._rank_one(self._low_name(int(k)),
                ' '.join(tokens), tokens, None) for k in ids], dtype=np.int8)
        keep = self._reachable_rows()[ids]
        ids, ranks = ids[keep], ranks[keep]
        # Preserve the historical path-only fallback for multiword queries.
        if len(ids) < limit and len(tokens) > 1:
            return None
        if not len(ids):
            return []
        sparse = len(ids) <= 4096 and getattr(self, '_operator_ext', None) is None
        if sparse:
            names = [self._orig_name(int(k)) for k in ids]
            if any(chr(0x0130) in name for name in names) or any(chr(0x0130) in root for root in self._roots):
                return None
            suffixes = [os.path.splitext(name)[1].lower() for name in names]
            extensions = list(dict.fromkeys(suffixes))
            table = {ext: i for i, ext in enumerate(extensions)}
            class SelectedSuffixes:
                def __init__(self):
                    self.values = {int(k): table[ext] for k, ext in zip(ids, suffixes)}
                def __getitem__(self, rows):
                    return np.asarray([self.values[int(k)] for k in rows], dtype=np.int32)
            split = SelectedSuffixes()
        else:
            _, split, values = self._operator_extensions()
            if getattr(self, '_operator_has_expansion', False):
                return None
            extensions = [v if v == '.' else '.' + v if v else '' for v in values]
        if kindset:
            allowed = np.asarray([v in kindset for v in extensions])
            keep = allowed[split[ids]]
            ids, ranks = ids[keep], ranks[keep]
        if prefixes is not None:
            filtered, remaining = self._directory_prefix_ids(ids, prefixes)
            if remaining is not None:
                filtered = np.asarray([k for k in filtered if any(
                    self._contains_path(self.path_of(int(k)), p) for p in remaining)], dtype=np.int64)
            if filtered is not ids:
                keep = np.isin(ids, filtered)
                ids, ranks = ids[keep], ranks[keep]
        scores = ranks.astype(np.float64) * 100 + np.minimum(np.maximum(self._depth[ids], 0), 20) * .5
        docs = np.asarray([v in _DOC_EXT for v in extensions])
        penalties = np.asarray([v in _PENALTY_EXT for v in extensions])
        scores -= np.where(docs[split[ids]], 8., 0.)
        scores += np.where(((self.flags[ids] & FLAG_JUNK) != 0) | penalties[split[ids]], 20., 0.)
        scores += np.where((self.flags[ids] & FLAG_HIDDEN) != 0, 10., 0.)
        use = np.asarray([min(float(self._ext_use.get(v, 0)), 5.) * .4 for v in extensions])
        scores -= use[split[ids]]
        if not len(ids):
            return []
        lengths = (self._selected_path_lengths(ids) if sparse else
                   self._operator_path_lengths()[ids])
        # Start with complete score ties; refine using global secondary keys.
        stat_count = min(_TOP_STAT, max(limit, 20))
        if len(ids) > max(limit * 4, 4096):
            boundary = np.partition(scores, stat_count - 1)[stat_count - 1]
            better = np.flatnonzero(scores < boundary)
            ties = np.flatnonzero(scores == boundary)
            needed = stat_count - len(better)
            # A broad token often leaves hundreds of thousands of identical
            # score ties. Refine only that complete tie tier, retaining every
            # strictly better key. This is global top-k, never a row-order cap.
            if len(ties) > max(stat_count * 4, 4096):
                cutoff = np.partition(lengths[ties], needed - 1)[needed - 1]
                better = np.concatenate((better, ties[lengths[ties] < cutoff]))
                ties = ties[lengths[ties] == cutoff]
                needed = stat_count - len(better)
            # Unknown mtimes must remain eligible for the stat-on-demand pass.
            if len(ties) > max(stat_count * 4, 4096) and np.all(self.mtime[ids] >= 0):
                times = self.mtime[ids[ties]]
                cutoff = np.partition(times, len(times) - needed)[len(times) - needed]
                ties = ties[times >= cutoff]
            pool = np.concatenate((better, ties))
        else:
            pool = np.arange(len(ids))
        # Preserve stable legacy ties after global score/path/mtime ranking.
        def select(count):
            return pool[np.lexsort((ids[pool], self.ln[ids[pool]],
                -np.maximum(self.mtime[ids[pool]], 0), lengths[pool], scores[pool]))[:count]]
        initial = select(stat_count)
        metadata_changed = False
        for j in initial:
            k = int(ids[j])
            if self.mtime[k] < 0:
                try:
                    st = os.stat(self.path_of(k))
                    self.size[k], self.mtime[k] = st.st_size, st.st_mtime
                    self._turbo_touch([k])
                    metadata_changed = True
                except OSError:
                    pass
        selected = select(limit) if metadata_changed else initial[:limit]
        out = []
        for j in selected:
            k = int(ids[j])
            path = self.path_of(k)
            out.append(dict(path=path, name=self._orig_name(k), folder=os.path.dirname(path),
                size=max(int(self.size[k]), 0), mtime=max(float(self.mtime[k]), 0.),
                flags=int(self.flags[k]), score=float(scores[j]),
                root=str(self._roots[int(self._root_of[k])])))
        out.sort(key=lambda h: (h['score'], len(h['path']), -h['mtime']))
        return out

    def _reachable_rows(self):
        """Resolve live ancestry in columns, without constructing any paths.

        Pointer doubling reaches a terminal in O(log N) passes. Cycles retain
        nonnegative pointers and are rejected, together with their descendants.
        Cache lifetime follows the same mutation generation as path lengths.
        """
        cached = getattr(self, '_reachable', None)
        if cached is not None:
            return cached
        n = len(self)
        jump = self.parent.astype(np.int64).copy()
        tops = jump < 0
        roots = (self._root_of >= 0) & (self._root_of < len(self._roots))
        jump[tops] = np.where(roots[tops], -1, -2)
        jump[jump >= n] = -2
        jump[(self.flags & FLAG_TOMB) != 0] = -2
        for _ in range(n.bit_length() + 1):
            active = np.flatnonzero(jump >= 0)
            if not active.size:
                break
            jump[active] = jump[jump[active]]
        self._reachable = jump == -1
        return self._reachable

    def _operator_path_lengths(self):
        """Exact path_of lengths for ordinary trees without materializing paths."""
        cached = getattr(self, "_operator_lengths", None)
        if cached is not None:
            return cached
        n = len(self)
        lengths = self.ln.astype(np.int64).copy()
        # Python len counts an astral character once, not two UTF-16 units.
        pairs = np.flatnonzero((self.blob_orig[:-1] >= 0xD800) &
                               (self.blob_orig[:-1] <= 0xDBFF) &
                               (self.blob_orig[1:] >= 0xDC00) &
                               (self.blob_orig[1:] <= 0xDFFF))
        if pairs.size:
            order = self._offset_rows()
            slots = np.searchsorted(self.off[order], pairs, side="right") - 1
            owners = order[np.maximum(slots, 0)]
            good = (slots >= 0) & (pairs + 1 < self.off[owners] + self.ln[owners])
            np.add.at(lengths, owners[good], -1)
        parent = self.parent
        reachable = self._reachable_rows()
        tops = np.flatnonzero((parent < 0) & reachable)
        prefix = np.full(n, -1, dtype=np.int64)
        result = np.full(n, -1, dtype=np.int64)
        # A top row contributes the root, not its own name, to children.
        for rp, root in enumerate(self._roots):
            rows = tops[self._root_of[tops] == rp]
            prefix[rows] = len(root)
            result[rows] = len(root) + lengths[rows] + (0 if root.endswith(
                (os.sep, os.altsep or os.sep)) or not root else 1)
            result[rows[lengths[rows] == 0]] = len(root)
            basename = os.path.basename(root)
            if basename:
                equal = self._operator_name_ids(basename)
                equal = equal[(parent[equal] < 0) & (self._root_of[equal] == rp) &
                              (lengths[equal] == len(basename))]
                # Only potential basename equals need original-case decoding.
                for k in equal:
                    if self._orig_name(int(k)) == basename:
                        result[k] = len(root)
        pending = np.flatnonzero((parent >= 0) & reachable)
        for _ in range(4096):
            if not pending.size:
                break
            ready = prefix[parent[pending]] >= 0
            rows = pending[ready]
            if not rows.size:
                break
            p = parent[rows]
            # Nonempty child names are joined; empty names are skipped.
            prefix[rows] = prefix[p] + np.where(lengths[rows] > 0, lengths[rows] + 1, 0)
            # A root ending in a separator needs no extra first separator.
            for rp, root in enumerate(self._roots):
                if root.endswith((os.sep, os.altsep or os.sep)) or not root:
                    first = rows[(parent[rows] >= 0) &
                                 (self.parent[parent[rows]] < 0) &
                                 (self._root_of[rows] == rp) & (lengths[rows] > 0)]
                    prefix[first] -= 1
            result[rows] = prefix[rows]
            pending = pending[~ready]
        # Unreachable rows never participate in ranking or consume result slots.
        for k in np.flatnonzero((result < 0) & reachable):
            result[k] = len(self.path_of(int(k)))
        result[~reachable] = 0
        self._operator_lengths = _integer_column(result, np.uint32, np.int64)
        return self._operator_lengths

    def _selected_extensions(self, rows):
        """Suffix ids for a bounded native candidate set, without an arena scan."""
        values, table = [""], {"": 0}
        ids, split = [], []
        expansion = any("\u0130" in root for root in self._roots)
        for row in rows:
            name = self._orig_name(int(row))
            expansion |= "\u0130" in name
            suffix = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
            extension = os.path.splitext(name)[1].lower()
            for value in (suffix, extension[1:] if extension != '.' else '.'):
                if value not in table:
                    table[value] = len(values)
                    values.append(value)
            ids.append(table[suffix])
            split.append(table[extension[1:] if extension != '.' else '.'])
        return np.asarray(ids, dtype=np.int32), np.asarray(split, dtype=np.int32), values, expansion

    def _search_spec(self, spec, tokens, limit, prefixes, kindset, candidate_rows=None):
        """Column masks first; decode/stat only survivors, slice after global sort."""
        from localdocs_filters import (FilterResults, matches, _regex_indices,
                                       _REGEX_SECONDS, _REGEX_CANDIDATES)
        deadline = time.monotonic() + _REGEX_SECONDS if spec["regex"] else None
        incomplete = False
        rows = (slice(None) if candidate_rows is None else
                np.unique(np.asarray(candidate_rows, dtype=np.int64)))
        if candidate_rows is not None and not rows.size:
            return FilterResults(invalid_regex=spec.get('invalid_regex', False))
        mask = (self.flags[rows] & (FLAG_TOMB | FLAG_DIR)) == 0
        mask &= self._reachable_rows()[rows]
        if candidate_rows is None:
            ext_ids, split_ids, extensions = self._operator_extensions()
            has_expansion = self._operator_has_expansion
        else:
            ext_ids, split_ids, extensions, has_expansion = self._selected_extensions(rows)
        if spec["ext"]:
            wanted = [i for i, ext in enumerate(extensions) if ext in spec["ext"]]
            mask &= np.isin(ext_ids, wanted)
        if kindset:
            wanted = [i for i, ext in enumerate(extensions)
                      if (ext if ext == "." else "." + ext if ext else "") in kindset]
            mask &= np.isin(split_ids, wanted)
        wildcards = []
        if spec["wildcards"]:
            mask &= self.ln[rows] > 0
        for token in spec["wildcards"]:
            if (token.startswith("*.") and len(token) > 2 and not any(
                    c in token[2:] for c in "*?.") and not
                    (has_expansion and "i" in token)):
                wanted = [i for i, ext in enumerate(extensions) if ext == token[2:]]
                # ext_ids describe only the last suffix: *.zip excludes .zip.bak.
                mask &= np.isin(ext_ids, wanted)
            else:
                wildcards.append(token)
        bounds = (("size", "min_size", "max_size"), ("mtime", "min_t", "max_t"))
        for column, lower, upper in bounds:
            values = getattr(self, column)[rows]
            known = values >= 0 if column == "size" else values != -1
            if spec[lower] is not None:
                exclusive = spec.get("min_size_exclusive", False) if column == "size" else False
                mask &= ~known | (values > spec[lower] if exclusive else values >= spec[lower])
            if spec[upper] is not None:
                exclusive = spec.get("max_size_exclusive", False) if column == "size" else True
                mask &= ~known | (values < spec[upper] if exclusive else values <= spec[upper])
        # Every free token must occur in the full path. Name matches and their
        # descendants give an uncapped fast superset; separator-spanning tokens
        # fall back to checking only the already masked candidates.
        path_checks = []
        text_checks = []
        directory_folder = self._directory_parent_mask(spec['folder'], folder=True,
            rows=(np.flatnonzero(mask) if candidate_rows is None else rows[mask])) if spec['folder'] else None
        if directory_folder is not None:
            mask &= directory_folder[rows]
        for token in list(tokens) + list(spec["path"]):
            if (any(c in token for c in (os.sep, chr(92), "/", "\u0307")) or
                    (has_expansion and "i" in token)):
                if token in tokens:
                    text_checks.append(token)
                if token in spec["path"]:
                    path_checks.append(token)
                continue
            eligible = np.flatnonzero(mask) if candidate_rows is None else rows[mask]
            directory_keep = self._directory_parent_mask([token], rows=eligible)
            if directory_keep is not None:
                remaining = eligible[~directory_keep[eligible]]
                own, _ = self._token_occurrences(token, rows=remaining)
                directory_keep[own] = True
                mask &= directory_keep[rows]
                continue
            own = self._operator_name_ids(token)
            keep = np.zeros(len(self), dtype=bool)
            keep[own] = True
            for rp, root in enumerate(self._roots):
                if token in root.lower():
                    keep[self._root_of == rp] = True
            # Top row names are omitted from descendant paths by path_of().
            frontier = keep & (self.parent >= 0)
            for _ in range(4096):
                valid_parent = (self.parent >= 0) & (self.parent < len(self))
                children = np.zeros(len(self), dtype=bool)
                children[valid_parent] = frontier[self.parent[valid_parent]] & ~keep[valid_parent]
                if not children.any():
                    break
                keep |= children
                frontier = children
            mask &= keep[rows]
        ids = np.flatnonzero(mask) if candidate_rows is None else rows[np.flatnonzero(mask)]
        if prefixes is not None:
            ids, prefixes = self._directory_prefix_ids(ids, prefixes)
        if deadline is not None and len(ids) > _REGEX_CANDIDATES:
            incomplete = True
            ids = ids[:_REGEX_CANDIDATES]
        # Check path/folder/glob before regex and unknown metadata resolution.
        folder_checks = spec['folder'] if directory_folder is None else []
        cheap = dict(spec, ext=set(), regex=[], wildcards=wildcards, path=path_checks, folder=folder_checks,
                     min_size=None, max_size=None, min_t=None, max_t=None)
        if prefixes is not None or text_checks or path_checks or folder_checks or wildcards:
            kept = []
            for k in ids:
                k = int(k)
                if deadline is not None and time.monotonic() >= deadline:
                    incomplete = True
                    break
                path = self.path_of(k)
                if prefixes is not None and not any(self._contains_path(path, p) for p in prefixes):
                    continue
                if text_checks and not all(t in path.lower() for t in text_checks):
                    continue
                if matches(cheap, {"path": path, "name": self._orig_name(k)}):
                    kept.append(k)
            ids = np.asarray(kept, dtype=np.int64)
        if deadline is not None:
            if len(ids) > _REGEX_CANDIDATES:
                incomplete = True
                ids = ids[:_REGEX_CANDIDATES]
            found, timed_out = _regex_indices(spec["regex"],
                [self._orig_name(int(k)) for k in ids], deadline) if ids.size else ([], False)
            ids = ids[np.asarray(found, dtype=np.int64)]
            incomplete |= timed_out
        metadata_predicate = any(spec[key] is not None for key in
                                 ("min_size", "max_size", "min_t", "max_t"))
        needs_meta = metadata_predicate or (spec["sort"] or "").lstrip("-") in ("size", "date")
        if needs_meta:
            unknown = ids[(self.size[ids] < 0) | (self.mtime[ids] == -1)]
            rejected = []
            if len(unknown) > 20000:
                incomplete = True
                rejected.extend(unknown[20000:])
            for k in unknown[:20000]:
                try:
                    st = os.stat(self.path_of(int(k)))
                    self.size[k], self.mtime[k] = int(st.st_size), float(st.st_mtime)
                    self._turbo_touch([k])
                except OSError:
                    if metadata_predicate:
                        rejected.append(k)
            if rejected:
                ids = ids[~np.isin(ids, rejected)]
            # Reapply numeric bounds to resolved metadata, without stat or regex.
            numeric = dict(spec, ext=set(), regex=[], wildcards=[], path=[], folder=[])
            if metadata_predicate:
                # Only unknown rows require scalar validation; known rows have
                # already passed the column comparisons above.
                failed = [int(k) for k in unknown[:20000] if not matches(numeric,
                          {"size": int(self.size[k]), "mtime": float(self.mtime[k])})]
                if failed:
                    ids = ids[~np.isin(ids, failed)]
        try:
            default = int(limit) if limit is not None else 50
            count = spec["limit"] if spec["limit"] is not None and spec["limit"] > 0 else (
                min(default, 1000) if default > 0 else 50)
        except (TypeError, ValueError):
            count = 50
        # No free text means _rank_one returns 1 for every nonempty name.
        # Rank and tie-break in columns, constructing only the final hit records.
        names = {}
        paths = {}
        ranks = np.where(self.ln[ids] == 0, 0.0, 1.0)
        order = spec["sort"]
        field = (order or "").lstrip("-")
        if tokens and not has_expansion:
            ranks = self._name_rank_columns(tokens, all_ranks=True)[ids].astype(np.float64)
        if (tokens and has_expansion) or self._dir_use or field in ("name", "ext"):
            for j, k in enumerate(ids):
                k = int(k)
                name, path = self._orig_name(k), self.path_of(k)
                names[k], paths[k] = name, path
                if tokens and has_expansion:
                    ranks[j] = self._rank_one(name.lower(), " ".join(tokens), tokens, path.lower())
        scores = 100.0 * ranks + np.minimum(np.maximum(self._depth[ids], 0), 20) * 0.5
        split_extensions = [ext if ext == "." else "." + ext if ext else ""
                            for ext in extensions]
        suffix_ids = split_ids[ids] if candidate_rows is None else split_ids[np.searchsorted(rows, ids)]
        doc_ext = np.asarray([ext in _DOC_EXT for ext in split_extensions])
        scores -= np.where(doc_ext[suffix_ids], 8.0, 0.0)
        penalties = np.asarray([ext in _PENALTY_EXT for ext in split_extensions])
        scores += np.where(((self.flags[ids] & FLAG_JUNK) != 0) |
                           penalties[suffix_ids], 20.0, 0.0)
        scores += np.where((self.flags[ids] & FLAG_HIDDEN) != 0, 10.0, 0.0)
        ext_use = np.asarray([min(float(self._ext_use.get(ext, 0)), 5.0) * 0.4
                              for ext in split_extensions])
        scores -= ext_use[suffix_ids]
        if self._dir_use:
            for j, k in enumerate(ids):
                k = int(k)
                scores[j] -= min(float(self._dir_use.get(
                    os.path.normcase(os.path.dirname(paths[k])), 0)), 5.0) * 0.2
        if field in ("size", "date"):
            values = self.size[ids] if field == "size" else self.mtime[ids]
            if field == "date":
                primary = values == -1
                values = np.where(primary, 0, values)
            else:
                primary = (values >= 0).astype(np.int8)
            if order.startswith("-"):
                values = -values
                if field == "size":
                    primary = -primary
            selected = np.lexsort((ids, values, primary))[:count]
        elif order:
            def key(j):
                k = int(ids[j])
                name = names[k].lower()
                return (ntpath.splitext(name)[1], name) if field == "ext" else name
            selected = sorted(range(len(ids)), key=key, reverse=order.startswith("-"))[:count]
        else:
            lengths = (self._operator_path_lengths()[ids] if candidate_rows is None else
                       np.asarray([len(self.path_of(int(k))) for k in ids]))
            # Partition on the leading key, keeping *all* boundary ties for
            # the exact stable lexicographic sort. Never discard score ties.
            if len(ids) > max(count * 4, 4096):
                boundary = np.partition(scores, min(count, len(ids)) - 1)[min(count, len(ids)) - 1]
                pool = np.flatnonzero(scores <= boundary)
            else:
                pool = np.arange(len(ids))
            selected = pool[np.lexsort((ids[pool], -self.mtime[ids[pool]], lengths[pool], scores[pool]))[:count]]
        hits = []
        for j in selected:
            k = int(ids[j])
            name = names.get(k)
            path = paths.get(k)
            if name is None:
                name = self._orig_name(k)
            if path is None:
                path = self.path_of(k)
            hits.append({"name": name, "path": path, "folder": os.path.dirname(path),
                         "size": int(self.size[k]), "mtime": float(self.mtime[k]),
                         "flags": int(self.flags[k]), "root": str(self._roots[int(self._root_of[k])]),
                         "score": float(scores[j])})
        result = FilterResults(hits, incomplete, spec.get("invalid_regex", False))
        spec["truncated"] = incomplete
        self.last_search_truncated = incomplete
        return result

    def search_parallel(self, query, limit=50, only_prefixes=None, kinds=None,
                        workers=4):
        if _parse_search_query(str(query or ''))[1] is not None:
            return self.search(query, limit, only_prefixes, kinds)
        try:
            toks = str(query or "").lower().split()
        except Exception:
            return []
        if not toks:
            return []
        try:
            w = max(1, min(int(workers), 8))
        except Exception:
            w = 4
        try:
            n = int(self.off.shape[0])
        except Exception:
            return []
        if n == 0:
            return []
        try:
            drv = min(toks, key=len)
            blocks = self._candidate_blocks(drv if len(drv) >= 3 else toks[0])
            nb = int(blocks.shape[0])
        except Exception:
            return self.search(query, limit, only_prefixes, kinds)
        if nb < 64 or w <= 1:
            return self.search(query, limit, only_prefixes, kinds)
        chunks = [blocks[i::w] for i in range(w)]
        raw = self._raw_bytes()
        try:
            drv_b = toks[0].encode("utf-16-le")
        except Exception:
            return self.search(query, limit, only_prefixes, kinds)
        try:
            byte_start = [int(x) * 2 for x in self.off.tolist()]
        except Exception:
            return self.search(query, limit, only_prefixes, kinds)
        import bisect as _bisect

        def verify(sub):
            got = set()
            try:
                for b in sub.tolist():
                    s = int(b) * self._block
                    e = min(n, s + self._block)
                    try:
                        lo = int(self.off[s]) * 2
                        hi = int(self.off[e - 1]) * 2 + int(self.ln[e - 1]) * 2
                    except Exception:
                        continue
                    at = raw.find(drv_b, lo, hi)
                    while at >= 0:
                        k = _bisect.bisect_right(byte_start, at) - 1
                        if 0 <= k < n:
                            got.add(k)
                        try:
                            at = raw.find(drv_b, at + 2, hi)
                        except Exception:
                            break
            except Exception:
                pass
            return got

        found = set()
        try:
            with ThreadPoolExecutor(max_workers=w) as ex:
                for got in ex.map(verify, chunks):
                    found |= set(got)
        except Exception:
            return self.search(query, limit, only_prefixes, kinds)
        return self.search(query, limit, only_prefixes, kinds)

    def touch(self, path):
        try:
            p = str(path)
        except Exception:
            return False
        try:
            with self._lock:
                try:
                    ext = os.path.splitext(p)[1].lower()
                    self._ext_use[ext] = self._ext_use.get(ext, 0) + 1
                except Exception:
                    pass
                try:
                    folder = os.path.normcase(os.path.dirname(p))
                    self._dir_use[folder] = self._dir_use.get(folder, 0) + 1
                    if len(self._dir_use) > 512:
                        items = sorted(self._dir_use.items(), key=lambda kv: kv[1])
                        for kk, _ in items[:128]:
                            self._dir_use.pop(kk, None)
                except Exception:
                    pass
                try:
                    self._lru.clear()
                except Exception:
                    pass
            return True
        except Exception as exc:
            self._err(exc)
            return False

    def _ensure_map(self):
        try:
            if self._map is not None:
                return self._map
            m = {}
            n = int(self.off.shape[0])
            for k in range(n):
                try:
                    if int(self.flags[k]) & FLAG_TOMB:
                        continue
                except Exception:
                    continue
                try:
                    p = self.path_of(k)
                    m[os.path.normcase(os.path.normpath(p))] = k
                except Exception:
                    pass
            self._map = m
            return m
        except Exception:
            return {}


    def _append_record(self, name, parent_idx, fl, root_pos):
        try:
            low = _lower_codes(name)
            orig = _encode_one(name)
        except Exception:
            return -1
        try:
            with self._lock:
                n = int(self.off.shape[0])
                start = int(self.blob_low.shape[0])
                self.blob_low = np.concatenate([self.blob_low, low])
                self.blob_orig = np.concatenate([self.blob_orig, orig])
                self.off = np.concatenate([self.off, np.array([start], dtype=np.uint64)])
                self.ln = np.concatenate([self.ln, _integer_column([len(low)], np.uint16, np.uint32)])
                self.parent = np.concatenate([self.parent, _integer_column([parent_idx], np.int32, np.int64)])
                self.flags = np.concatenate([self.flags, np.array([int(fl) & 0xFF], dtype=np.uint8)])
                self.frn = np.concatenate([self.frn, np.array([0], dtype=np.uint64)])
                self.size = np.concatenate([self.size, np.array([-1], dtype=np.int64)])
                self.mtime = np.concatenate([self.mtime, np.array([-1.0], dtype=np.float64)])
                self._root_of = np.concatenate([self._root_of, np.array([int(root_pos)], dtype=np.int32)])
                try:
                    dep = int(self._depth[int(parent_idx)]) + 1 if int(parent_idx) >= 0 else 0
                except Exception:
                    dep = 0
                self._depth = np.concatenate([self._depth, _integer_column([dep], np.uint8, np.int32)])
                self._delta_n += 1
                self._turbo_touch([n])
                self._invalidate(preserve_turbo=True)
                return n
        except Exception as exc:
            self._err(exc)
            return -1

    def _contains_path(self, path, root):
        if not self._journal_state:
            return _prefix_match(path, root)
        a = os.path.normcase(os.path.normpath(path))
        b = os.path.normcase(os.path.normpath(root))
        return a == b or a.startswith(b.rstrip(os.sep) + os.sep)

    def _root_for_path(self, full):
        try:
            best, best_i = -1, -1
            for i, r in enumerate(self._roots):
                try:
                    if self._contains_path(full, r):
                        if len(str(r)) > best:
                            best, best_i = len(str(r)), i
                except Exception:
                    pass
            return best_i
        except Exception:
            return -1

    @staticmethod
    def _event_hashes(blob, off, lengths):
        # Hash every UTF-16 unit. Bounded numpy chunks avoid both Python work
        # per filename and degenerate buckets for long shared name prefixes.
        out = lengths.astype(np.uint64).copy()
        for lo in range(0, len(off), 32768):
            hi = min(len(off), lo + 32768)
            lens = lengths[lo:hi].astype(np.int64)
            starts = np.cumsum(lens) - lens
            total = int(lens.sum())
            if not total:
                continue
            which = np.repeat(np.arange(hi - lo), lens)
            local = np.arange(total, dtype=np.int64) - starts[which]
            positions = off[lo:hi][which].astype(np.int64) + local
            values = blob[positions].astype(np.uint64)
            values += (local.astype(np.uint64) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
            values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            values ^= values >> np.uint64(31)
            nonempty = np.flatnonzero(lens)
            out[lo + nonempty] ^= np.add.reduceat(values, starts[nonempty])
        return out

    @staticmethod
    def _event_hash_name(name):
        codes = _lower_codes(name) if os.path.normcase("A") == "a" else _encode_one(name)
        mask, total = 0xFFFFFFFFFFFFFFFF, 0
        for j, code in enumerate(codes):
            value = (int(code) + (j + 1) * 0x9E3779B97F4A7C15) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            total = (total + (value ^ (value >> 31))) & mask
        return total ^ len(codes)

    def _build_event_base(self):
        """Lookup tables for journal events (built without the index lock when called from the prebuild thread)."""
        dtype = np.dtype([("root", "i4"), ("parent", self.parent.dtype), ("hash", "u8")])
        keys = np.empty(len(self), dtype=dtype)
        keys["root"], keys["parent"] = self._root_of, self.parent
        blob = self.blob_low if os.path.normcase("A") == "a" else self.blob_orig
        keys["hash"] = self._event_hashes(blob, self.off, self.ln)
        # lexsort on plain columns instead of a structured argsort: the structured sort held the GIL and the index lock
        # for ~1.6 s at 2.6M rows on the first live event (measured stall), starving every query.
        order = np.lexsort((keys["hash"], keys["parent"], keys["root"]))
        refs = np.argsort(self.frn, kind="stable")
        # Immutable numpy tables can be shared by publication clones.
        return (keys[order], order, self.frn[refs].copy(), refs)

    def _event_tables(self):
        if getattr(self, "_event_base", None) is not None:
            return
        self._event_base = self._build_event_base()
        self._event_names = {}
        self._event_refs = {}

    def prebuild_event_tables(self):
        """Background: build the event tables outside the lock so the first journal event does not stall queries."""
        try:
            with self._lock:
                if getattr(self, "_event_base", None) is not None:
                    return
                gen, n = self._bloom_generation, len(self)
            base = self._build_event_base()
            with self._lock:
                if getattr(self, "_event_base", None) is None and gen == self._bloom_generation and n == len(self):
                    self._event_base = base
                    self._event_names = {}
                    self._event_refs = {}
        except Exception as exc:
            self._err(exc)

    def apply_events(self, adds=None, removes=None, renames=None):
        """Stage only touched rows; publish each appended column once.

        Lookup tables contain numpy rows plus a bounded overlay. No global path
        cache or per-file Python records are constructed on the update path.
        """
        result = {"added": 0, "removed": 0, "renamed": 0}
        if not adds and not removes and not renames:
            return result
        try:
            with self._lock:
                self._event_tables()
                keys, order, sorted_refs, ref_order = self._event_base
                n = len(self)
                changed, fresh = {}, []
                metadata_changed = False
                names = self._event_names
                references = self._event_refs
                base_hits = {}

                def row(k):
                    if k >= n:
                        return fresh[k - n]
                    if k not in changed:
                        changed[k] = [k, self.blob_orig[int(self.off[k]):int(self.off[k]) + int(self.ln[k])]
                                      .tobytes().decode("utf-16-le", "surrogatepass"), int(self.parent[k]),
                                      int(self.flags[k]), int(self._root_of[k]),
                                      int(self.size[k]), float(self.mtime[k]), int(self.frn[k])]
                    return changed[k]

                def alive(k):
                    return not (row(k)[3] & FLAG_TOMB)

                def namekey(rp, parent, name):
                    return (rp, parent, os.path.normcase(name))

                def remember(k):
                    r = row(k)
                    nk = namekey(r[4], r[2], r[1])
                    names[nk] = tuple(set(names.get(nk, ())) | {k})
                    if r[7]:
                        references[r[7]] = tuple(set(references.get(r[7], ())) | {k})

                def children(rp, parent, name):
                    nk = namekey(rp, parent, name)
                    cached = base_hits.get(nk)
                    if cached is None:
                        h = self._event_hash_name(name)
                        target = np.array((rp, parent, h), dtype=keys.dtype)
                        lo, hi = np.searchsorted(keys, target, side="left"), np.searchsorted(keys, target, side="right")
                        cached = tuple(order[int(lo):int(hi)].tolist())
                        base_hits[nk] = cached
                    candidates = set(cached)
                    candidates.update(names.get(nk, ()))
                    return [k for k in candidates if namekey(row(k)[4], row(k)[2], row(k)[1]) == nk]

                def resolve(full):
                    rp = self._root_for_path(full)
                    if rp < 0:
                        return None
                    root = self._roots[rp]
                    rel = os.path.relpath(full, root)
                    # A USN volume root is stored as "C:", whereas basename
                    # of the drive root (with its trailing separator) is empty.
                    root_name = os.path.basename(root) or str(root).rstrip(_BS + _SL)
                    base = children(rp, -1, root_name)
                    if rel in ("", "."):
                        hits = [k for k in base if alive(k)]
                        return max(hits) if hits else None
                    parents = [-1] + base
                    for component in rel.split(os.sep):
                        hits = set()
                        for p in parents:
                            hits.update(children(rp, p, component))
                        parents = list(hits)
                        if not parents:
                            return None
                    hits = [k for k in parents if alive(k)]
                    return max(hits) if hits else None

                def by_ref(reference, rp):
                    lo, hi = np.searchsorted(sorted_refs, np.uint64(reference), side="left"), np.searchsorted(sorted_refs, np.uint64(reference), side="right")
                    candidates = set(ref_order[int(lo):int(hi)].tolist()) if reference else set()
                    candidates.update(references.get(reference, ()))
                    candidates = [k for k in candidates if row(k)[7] == reference
                                  and (rp < 0 or row(k)[4] == rp)]
                    live = [k for k in candidates if alive(k)]
                    hit = max(live) if live and (rp >= 0 or len(live) == 1) else None
                    return hit, bool(candidates)

                def lookup(item, full, fallback=False):
                    if isinstance(item, dict) and item.get("frn") is not None:
                        reference = int(item["frn"])
                        hit, known = by_ref(reference, self._root_for_path(full) if full else -1)
                        if hit is not None or known or not fallback:
                            return hit
                        hit = resolve(full)
                        if hit is not None and reference and row(hit)[7]:
                            mask = 0x0000FFFFFFFFFFFF
                            if reference & mask == row(hit)[7] & mask:
                                return None
                        return hit
                    return resolve(full)

                def append(name, parent, flags, rp, reference=0):
                    k = n + len(fresh)
                    fresh.append([k, name, parent, flags, rp, -1, -1.0, reference])
                    remember(k)
                    return k

                def ensure_dir(folder, rp):
                    hit = resolve(folder)
                    if hit is not None:
                        return hit
                    root = self._roots[rp]
                    if not self._contains_path(folder, root) or _norm(folder) == _norm(root):
                        return -1
                    parent = ensure_dir(os.path.dirname(folder), rp)
                    if parent < 0:
                        root_name = os.path.basename(root) or str(root).rstrip(_BS + _SL)
                        parent = append(root_name, -1, FLAG_DIR, rp)
                    return append(os.path.basename(folder), parent, FLAG_DIR, rp)

                def parent_of(item, full, rp):
                    if isinstance(item, dict) and item.get("parent_frn") is not None:
                        hit, _ = by_ref(int(item["parent_frn"]), rp)
                        if hit is not None:
                            return hit
                    return ensure_dir(os.path.dirname(full), rp)

                for it in removes or ():
                    try:
                        full = str(it.get("path", "")) if isinstance(it, dict) else str(it)
                        k = int(it) if isinstance(it, int) else lookup(it, full, True)
                        if k is None or not 0 <= k < n + len(fresh) or not alive(k):
                            continue
                        row(k)[3] |= FLAG_TOMB
                        self._layout_dirty = True
                        result["removed"] += 1
                    except Exception as exc:
                        self._err(exc)
                for it in adds or ():
                    try:
                        full = str(it.get("path", "")) if isinstance(it, dict) else str(it)
                        rp = self._root_for_path(full)
                        if not full or rp < 0:
                            continue
                        reference = int(it.get("frn", 0)) if isinstance(it, dict) else 0
                        # Identity wins over spelling (a repeated create may
                        # arrive after a rename). Zero is not an identity.
                        hit = by_ref(reference, rp)[0] if reference else None
                        if hit is None:
                            hit = resolve(full)
                        if hit is not None:
                            if not reference or row(hit)[7] == reference:
                                if isinstance(it, dict):
                                    for field, slot, convert in (("size", 5, int), ("mtime", 6, float)):
                                        if field in it:
                                            row(hit)[slot] = convert(it[field])
                                            metadata_changed = True
                                            self._layout_dirty = True
                                continue
                            row(hit)[3] |= FLAG_TOMB
                            self._layout_dirty = True
                        parent = parent_of(it, full, rp)
                        name = os.path.basename(full) or full
                        flags = FLAG_DIR if isinstance(it, dict) and it.get("is_dir") else 0
                        if os.path.splitext(name)[1].lower() in _PENALTY_EXT:
                            flags |= FLAG_JUNK
                        k = append(name, parent, flags, rp, reference)
                        if isinstance(it, dict):
                            if "size" in it:
                                row(k)[5] = int(it["size"])
                            if "mtime" in it:
                                row(k)[6] = float(it["mtime"])
                        result["added"] += 1
                    except Exception as exc:
                        self._err(exc)
                moves = []
                for it in renames or ():
                    try:
                        if isinstance(it, (tuple, list)) and len(it) == 2:
                            old, new = str(it[0]), str(it[1])
                        elif isinstance(it, dict):
                            old = str(it.get("old", it.get("src", it.get("path", ""))))
                            new = str(it.get("new", it.get("dst", it.get("new_path", ""))))
                        else:
                            continue
                        k, rp = lookup(it, old), self._root_for_path(new)
                        if k is None or not old or not new or rp < 0:
                            continue
                        if self._contains_path(os.path.dirname(new), old):
                            continue
                        parent = parent_of(it, new, rp)
                        r = row(k)
                        previous_root = r[4]
                        r[1], r[2], r[4] = os.path.basename(new) or new, parent, rp
                        remember(k)
                        moves.append((k, previous_root, rp))
                        self._layout_dirty = True
                        result["renamed"] += 1
                    except Exception as exc:
                        self._err(exc)

                if any(result.values()) or metadata_changed:
                    # _assemble sees only the new rows, never the original N.
                    # This also preserves the one-bulk-builder-per-batch contract.
                    snap = self._assemble([r[:7] for r in fresh], self._roots,
                                          on_error=self._on_error)
                    renamed_rows = [k for k, r in changed.items() if
                        self._encode_event_name(r[1]) != self.blob_orig[
                            int(self.off[k]):int(self.off[k]) + int(self.ln[k])].tobytes()]
                    extra = [self._encode_event_name(row(k)[1]) for k in renamed_rows]
                    new_blob = snap.blob_orig.tobytes() + b"".join(extra)
                    if new_blob:
                        units = np.frombuffer(new_blob, dtype=np.uint16)
                        base = len(self.blob_orig)
                        self.blob_orig = np.concatenate((self.blob_orig, units))
                        self.blob_low = np.concatenate((self.blob_low, _lower_units(units)))
                    else:
                        base = len(self.blob_orig)
                    if max(base + len(new_blob) // 2, len(self.blob_orig)) > np.iinfo(self.off.dtype).max:
                        self.off = self.off.astype(np.uint64)
                    if fresh:
                        snap.off = _integer_column(snap.off.astype(np.uint64) + np.uint64(base), np.uint32, np.uint64)
                        snap.parent = _integer_column([r[2] for r in fresh], np.int32, np.int64)
                        snap.frn = np.asarray([r[7] for r in fresh], dtype=np.uint64)
                        for column in ("off", "ln", "parent", "flags", "frn", "size", "mtime", "_root_of", "_depth"):
                            setattr(self, column, np.concatenate((getattr(self, column), getattr(snap, column))))
                    pos = base + len(snap.blob_orig)
                    if extra and max(map(len, extra)) // 2 > np.iinfo(self.ln.dtype).max:
                        self.ln = self.ln.astype(np.uint32)
                    for k, raw in zip(renamed_rows, extra):
                        self.off[k], self.ln[k] = pos, len(raw) // 2
                        pos += len(raw) // 2
                    if any(r[2] > np.iinfo(self.parent.dtype).max for r in changed.values()):
                        self.parent = self.parent.astype(np.int64)
                    for k, r in changed.items():
                        self.parent[k], self.flags[k], self._root_of[k] = r[2], r[3], r[4]
                        self.size[k], self.mtime[k] = r[5], r[6]
                    # Depth/root propagation is needed only for moved branches.
                    # Membership and identities stay unchanged for ordinary moves.
                    for k in range(n, len(self)):
                        p = int(self.parent[k])
                        self._depth[k] = min(64, int(self._depth[p]) + 1) if p >= 0 else 0
                    for k, old_root, rp in moves:
                        p = int(self.parent[k])
                        self._depth[k] = min(64, int(self._depth[p]) + 1) if p >= 0 else 0
                        frontier = np.array([k], dtype=np.int64)
                        for _ in range(64):
                            if not frontier.size or not self.flags[k] & FLAG_DIR:
                                break
                            below = np.flatnonzero(np.isin(self.parent, frontier))
                            if not below.size:
                                break
                            self._depth[below] = np.minimum(64, self._depth[self.parent[below]].astype(np.int32) + 1)
                            self._root_of[below] = rp
                            self._turbo_touch(below)
                            frontier = below
                        if old_root != rp:
                            self._event_base = None
                    self._delta_n += len(fresh)
                    self._turbo_touch(list(changed) + list(range(n, len(self))))
                    self._pack_memory()
                    self._invalidate(keep_event_tables=True)
                    if len(names) + len(references) > 10000:
                        self._event_base = None
                return result
        except Exception as exc:
            self._err(exc)
            return {"added": 0, "removed": 0, "renamed": 0}

    @staticmethod
    def _encode_event_name(name):
        return name.encode("utf-16-le", "surrogatepass")

    def dedupe_live_rows(self):
        """Keep the first live sibling of each lowercase name, preserving ancestry.

        Exact UTF-16 keys, numpy sorting and pointer doubling avoid a Python
        object per row. Process parents before children so duplicates nested
        beneath duplicate directories are merged in the same pass.
        """
        try:
            with self._lock:
                n = len(self)
                if not n:
                    return 0
                parent = self.parent.astype(np.int64).copy()
                if np.any((parent < -1) | (parent >= n)):
                    raise ValueError("invalid parent in snapshot")
                ancestor = parent.copy()
                depth = (parent >= 0).astype(np.int64)
                for _ in range(n.bit_length()):
                    active = ancestor >= 0
                    if not active.any():
                        break
                    at = ancestor[active]
                    depth[active] += depth[at]
                    ancestor[active] = ancestor[at]
                if np.any(ancestor >= 0):
                    raise ValueError("cyclic snapshot ancestry")
                live = np.flatnonzero((self.flags & FLAG_TOMB) == 0)
                if not live.size:
                    return 0
                # NTFS names are bounded; pack once and compare exact bytes,
                # including lengths (embedded/trailing zero units are legal).
                width = max(1, int(self.ln[live].max()))
                packed = np.zeros((live.size, width), dtype=np.uint16)
                for lo in range(0, live.size, 32768):
                    hi = min(live.size, lo + 32768)
                    ids = live[lo:hi]
                    lengths = self.ln[ids].astype(np.int64)
                    which = np.repeat(np.arange(hi - lo), lengths)
                    starts = np.cumsum(lengths) - lengths
                    columns = np.arange(int(lengths.sum())) - starts[which]
                    packed[lo + which, columns] = self.blob_low[
                        self.off[ids][which].astype(np.int64) + columns]
                dtype = np.dtype([("root", "i4"), ("parent", "i8"),
                                  ("length", "u4"), ("name", "V%d" % (width * 2))])
                keys = np.empty(live.size, dtype=dtype)
                keys["root"] = self._root_of[live]
                keys["length"] = self.ln[live]
                keys["name"] = packed.view("V%d" % (width * 2)).reshape(-1)
                levels = np.argsort(depth[live], kind="stable")
                boundaries = np.r_[0, np.flatnonzero(np.diff(depth[live][levels])) + 1, live.size]
                canonical = np.arange(n, dtype=np.int64)
                duplicates = []
                for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                    slots = levels[lo:hi]
                    ids = live[slots]
                    p = parent[ids]
                    keys["parent"][slots] = np.where(p >= 0, canonical[np.maximum(p, 0)], -1)
                    _, first, inverse = np.unique(keys[slots], return_index=True, return_inverse=True)
                    survivors = ids[first[inverse]]
                    canonical[ids] = survivors
                    duplicates.append(ids[ids != survivors])
                dead = np.concatenate(duplicates)
                if dead.size:
                    # Repoint ALL children before marking duplicate directories.
                    has_parent = parent >= 0
                    parent[has_parent] = canonical[parent[has_parent]]
                    self.parent = _integer_column(parent, np.int32, np.int64)
                    self.flags[dead] |= FLAG_TOMB
                    self._layout_dirty = True
                    self._invalidate()
                return int(dead.size)
        except Exception as exc:
            self._err(exc)
            return 0

    def _compact(self):
        try:
            with self._lock:
                n = int(self.off.shape[0])
                keep_old = []
                for k in range(n):
                    try:
                        if not (int(self.flags[k]) & FLAG_TOMB):
                            keep_old.append(k)
                    except Exception:
                        pass
                if len(keep_old) == n:
                    self._layout_dirty = False
                    return True
                remap = {o: i for i, o in enumerate(keep_old)}
                recs = []
                for i, o in enumerate(keep_old):
                    try:
                        nm = self.blob_orig[int(self.off[o]):int(self.off[o]) + int(self.ln[o])].tobytes().decode("utf-16-le")
                    except Exception:
                        nm = ""
                    try:
                        p = int(self.parent[o])
                        np_ = remap.get(p, -1)
                    except Exception:
                        np_ = -1
                    try:
                        recs.append((i, nm, np_, int(self.flags[o]), int(self._root_of[o]),
                                     int(self.size[o]), float(self.mtime[o])))
                    except Exception:
                        pass
                roots = list(self._roots)
            snap = self._assemble(recs, roots, on_error=self._on_error)
            try:
                with self._lock:
                    self.blob_low = snap.blob_low
                    self.blob_orig = snap.blob_orig
                    self.off = snap.off
                    self.ln = snap.ln
                    self.parent = snap.parent
                    self.flags = snap.flags
                    self.frn = self.frn[keep_old].copy()
                    self.size = snap.size
                    self.mtime = snap.mtime
                    self._root_of = snap._root_of
                    self._depth = snap._depth
                    self._base_n = len(snap)
                    self._delta_n = 0
                    self._layout_dirty = False
                    self._invalidate()
                return True
            except Exception as exc:
                self._err(exc)
                return False
        except Exception as exc:
            self._err(exc)
            return False

    def stats(self):
        try:
            with self._lock:
                n = int(self.off.shape[0])
                try:
                    ndirs = int(np.count_nonzero((self.flags & np.uint8(FLAG_DIR)) != 0))
                except Exception:
                    ndirs = 0
                try:
                    tomb = int(np.count_nonzero((self.flags & np.uint8(FLAG_TOMB)) != 0))
                except Exception:
                    tomb = 0
                try:
                    blob_bytes = int(self.blob_low.shape[0]) * 2
                except Exception:
                    blob_bytes = 0
                try:
                    nb = (n + self._block - 1) // self._block if n else 0
                except Exception:
                    nb = 0
                try:
                    bloom_mb = float(self._bloom.nbytes) / 1048576.0 if self._bloom is not None else 0.0
                except Exception:
                    bloom_mb = 0.0
                return {"files": n - ndirs, "dirs": ndirs, "entries": n,
                        "bytes_names": blob_bytes, "blocks": nb,
                        "bloom_mb": bloom_mb, "tombstones": tomb,
                        "delta": int(self._delta_n), "roots": list(self._roots)}
        except Exception as exc:
            self._err(exc)
            return {"files": 0, "dirs": 0, "entries": 0, "bytes_names": 0,
                    "blocks": 0, "bloom_mb": 0.0, "tombstones": 0, "delta": 0, "roots": []}

    def save(self, path, on_error=None):
        try:
            p = os.fspath(path)
        except Exception as exc:
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
            return False
        try:
            self.build_trigrams()
            with self._lock:
                data = {"version": np.int64(2),
                        "memory_mode": np.array(self._memory_mode),
                        "block": np.int64(self._block),
                        "shared_names": np.int8(self.blob_orig is self.blob_low),
                        "blob_low": self.blob_low, "blob_orig": self.blob_orig,
                        "off": self.off, "ln": self.ln, "parent": self.parent,
                        "flags": self.flags, "frn": self.frn, "size": self.size,
                        "mtime": self.mtime, "root_of": self._root_of,
                        "depth": self._depth,
                        "starts": self._starts_np(),
                        "reachable": self._reachable_rows(),
                        "contiguous": np.int8(self._contiguous_names()),
                        "roots": np.array([json.dumps(list(self._roots), ensure_ascii=True).encode("utf-8")]),
                        "journal_letters": np.array([s[0] for s in self._journal_state], dtype="U1"),
                        "journal_id": np.array([s[1] for s in self._journal_state], dtype=np.uint64),
                        "next_usn": np.array([s[2] for s in self._journal_state], dtype=np.int64)}
                if isinstance(self.blob_low, _NameArena):
                    data.pop("blob_low")
                    data.pop("blob_orig")
                    data["name_encoding"] = np.array("utf16-alphabet-v1")
                    data["name_codes"] = self.blob_low.codes
                    data["name_alphabet"] = self.blob_low.alphabet
                    data["name_positions"] = self.blob_orig.positions
                    data["name_spelling"] = self.blob_orig.spelling
                n = int(self.off.shape[0])
                if self._bloom is not None and self._bloom.shape == ((n + self._block - 1) // self._block, BLOOM_U64):
                    data["bloom"] = self._bloom
                    data["bloom_layout"] = np.array([BLOOM_VERSION, self._block, BLOOM_BITS], dtype=np.int64)
                if self.blob_orig is self.blob_low:
                    data.pop("blob_orig", None)
                for name in ("frn", "size", "mtime"):
                    column = getattr(self, name)
                    if isinstance(column, _DefaultColumn):
                        del data[name]
                        items = list(column.values.items())
                        data[name + "_rows"] = np.fromiter((int(k) for k, _ in items), dtype=np.int64, count=len(items))
                        data[name + "_values"] = np.fromiter((v for _, v in items), dtype=column.dtype, count=len(items))
                _atomic_savez(p, data)
            with self._lock:
                self._base_n = n
                self._delta_n = 0
                self._layout_dirty = False
            try:
                dp = p + ".delta"
                if os.path.exists(dp):
                    os.remove(dp)
            except Exception:
                pass
            return True
        except Exception as exc:
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
            self._err(exc)
            return False

    def save_segmented(self, base_path, on_error=None):
        # A complete v2 generation also persists updates to existing rows and
        # the journal watermark. Legacy append-only deltas cannot express these.
        return self.save(base_path, on_error=on_error)


    @classmethod
    def _load_v2(cls, base, on_error=None):
        idx = cls(on_error=on_error)
        roots = json.loads(base["roots"].reshape(-1)[0].decode("utf-8"))
        if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
            raise ValueError("invalid snapshot roots")
        idx._roots = roots
        n = len(base["off"])
        coded = "name_encoding" in base
        if coded and str(base["name_encoding"]) != "utf16-alphabet-v1":
            raise ValueError("unsupported packed name encoding")
        for name, allowed in (("blob_low", "u"), ("off", "u"), ("ln", "u"),
                              ("parent", "i"), ("flags", "u"),
                              ("root_of", "i"), ("depth", "iu")):
            if coded and name == "blob_low":
                continue
            value = base[name]
            if value.ndim != 1 or value.dtype.kind not in allowed:
                raise ValueError("invalid snapshot column " + name)
            if name != "blob_low" and len(value) != n:
                raise ValueError("snapshot column length mismatch " + name)
            setattr(idx, "_" + name if name in ("root_of", "depth") else name, value)
        if coded:
            codes, alphabet = base["name_codes"], base["name_alphabet"]
            positions, spelling = base["name_positions"], base["name_spelling"]
            if (codes.ndim != 1 or codes.dtype != np.dtype(np.uint8) or
                    alphabet.ndim != 1 or alphabet.dtype != np.dtype(np.uint16) or
                    not 0 < len(alphabet) <= 256 or positions.ndim != 1 or
                    positions.dtype not in (np.dtype(np.uint32), np.dtype(np.uint64)) or
                    spelling.shape != positions.shape or spelling.dtype != np.dtype(np.uint16)):
                raise ValueError("invalid coded name arena")
            if len(positions) and (int(positions[-1]) >= len(codes) or
                                  np.any(positions[1:] <= positions[:-1])):
                raise ValueError("invalid spelling overlay")
            idx.blob_low = _NameArena(codes, alphabet)
            idx.blob_orig = (_NameArena(codes, alphabet, positions, spelling)
                             if len(positions) else idx.blob_low)
        else:
            if idx.blob_low.dtype != np.dtype(np.uint16):
                raise ValueError("unsupported name encoding")
            idx.blob_low.flags.writeable = False
            if bool(base["shared_names"]):
                idx.blob_orig = idx.blob_low
            else:
                idx.blob_orig = base["blob_orig"]
                if idx.blob_orig.dtype != np.dtype(np.uint16) or idx.blob_orig.shape != idx.blob_low.shape:
                    raise ValueError("invalid original name arena")
                idx.blob_orig.flags.writeable = False
        for name, dtype, default in (("frn", np.uint64, 0),
                                     ("size", np.int64, -1),
                                     ("mtime", np.float64, -1)):
            if name in base:
                value = base[name]
                if value.shape != (n,) or value.dtype != np.dtype(dtype):
                    raise ValueError("invalid metadata column " + name)
            else:
                rows, values = base[name + "_rows"], base[name + "_values"]
                if rows.ndim != 1 or rows.dtype != np.dtype(np.int64) or values.shape != rows.shape or values.dtype != np.dtype(dtype):
                    raise ValueError("invalid sparse column " + name)
                if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n):
                    raise ValueError("sparse row outside snapshot")
                value = _DefaultColumn(n, dtype, default, zip(rows.tolist(), values.tolist()))
            setattr(idx, name, value)
        idx._base_n = n
        idx._journal_state = list(zip(base["journal_letters"].tolist(),
                                      base["journal_id"].tolist(),
                                      base["next_usn"].tolist()))
        # Honour the selected mode. A different block layout must be rebuilt;
        # the common same-mode restart does no O(N) conversion or validation.
        stored_block = int(base["block"])
        if "bloom" in base and stored_block == idx._block:
            saved = base["bloom"]
            if (base["bloom_layout"].tolist() == [BLOOM_VERSION, idx._block, BLOOM_BITS]
                    and saved.dtype == np.dtype(np.uint64)
                    and saved.shape == ((n + idx._block - 1) // idx._block, BLOOM_U64)):
                saved.flags.writeable = False
                idx._bloom = saved
        if "reachable" in base:
            reachable = base["reachable"]
            if reachable.dtype != np.dtype(bool) or reachable.shape != (n,):
                raise ValueError("invalid snapshot reachability")
            reachable.flags.writeable = False
            idx._reachable = reachable
        idx._name_layout = bool(base["contiguous"])
        starts = base["starts"]
        if starts.dtype not in (np.dtype(np.int64), np.dtype(np.uint32), np.dtype(np.uint64)) or starts.shape != (n,):
            raise ValueError("invalid snapshot starts")
        starts.flags.writeable = False
        idx._starts_cache = (n, idx.off, starts)
        return idx

    @classmethod
    def load(cls, path, mmap_mode="r", on_error=None):
        try:
            p = os.fspath(path)
        except Exception as exc:
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
            return cls(on_error=on_error)
        base = None
        try:
            # Include the format probe: its ordinary file handle must also be
            # closed before a Windows writer can retire the mapped archive.
            with _snapshot_lock:
                generation = _snapshot_generation(p)
                base = np.load(generation, mmap_mode=mmap_mode, allow_pickle=False)
                if int(base["version"]) == 2:
                    base.close()
                    if generation == os.path.abspath(p):
                        # Legacy v2 archives used the logical path as data. Migrate
                        # by copying before mapping so a subsequent save can
                        # replace that path with an ordinary pointer on Windows.
                        import shutil
                        fd, detached = tempfile.mkstemp(prefix=os.path.basename(p) + '.v2.',
                            suffix='.npz', dir=os.path.dirname(os.path.abspath(p)))
                        os.close(fd)
                        shutil.copyfile(generation, detached)
                        generation = detached
                    loaded = cls._load_v2(_MappedNPZ(generation), on_error=on_error)
                    if loaded._journal_state:
                        loaded.dedupe_live_rows()  # same repair hook as the legacy loader (journal-built snapshots may hold duplicate paths)
                    return loaded
        except Exception as exc:
            if base is not None:
                try:
                    base.close()
                except Exception:
                    pass
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
            return cls(on_error=on_error)
        try:
            def g(k, dt):
                try:
                    return np.ascontiguousarray(base[k], dtype=dt)
                except Exception:
                    return np.zeros((0,), dtype=dt)
            n = int(np.asanyarray(base["off"]).shape[0])
            stored_roots = np.asanyarray(base["roots"])
            if stored_roots.dtype.kind == "S":
                roots = json.loads(stored_roots.reshape(-1)[0].decode("utf-8"))
            elif stored_roots.dtype.kind == "U":
                roots = stored_roots.tolist()  # Safe legacy Unicode arrays.
            else:
                raise ValueError("Unsupported roots encoding; rebuild snapshot")
            if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
                raise ValueError("Invalid snapshot roots; rebuild snapshot")
            idx = cls.from_arrays(g("blob_low", np.uint16), g("blob_orig", np.uint16),
                                  g("off", np.uint64), g("ln", np.uint32),
                                  _integer_column(base["parent"], np.int32, np.int64),
                                  g("flags", np.uint8),
                                  roots, g("root_of", np.int32),
                                  frn=g("frn", np.uint64), size=g("size", np.int64),
                                  mtime=g("mtime", np.float64),
                                  depth=g("depth", np.int32), on_error=on_error)
            if "journal_letters" in base:
                idx._journal_state = list(zip(base["journal_letters"].tolist(),
                                              base["journal_id"].tolist(),
                                              base["next_usn"].tolist()))
            try:
                saved = np.ascontiguousarray(base["bloom"], dtype=np.uint64)
                layout = np.asarray(base["bloom_layout"]).tolist()
                if (layout == [BLOOM_VERSION, idx._block, BLOOM_BITS] and
                        saved.shape == ((n + idx._block - 1) // idx._block, BLOOM_U64)):
                    idx._bloom = saved
            except Exception:
                pass
            try:
                base.close()
            except Exception:
                pass
            idx._base_n = n
            try:
                dp = p + ".delta"
                if os.path.exists(dp):
                    d = np.load(dp, allow_pickle=False)
                    try:
                        dn = int(np.asanyarray(d["off"]).shape[0])
                    except Exception:
                        dn = 0
                    if dn > 0:
                        try:
                            doff = np.ascontiguousarray(d["off"], dtype=np.uint64)
                            idx.blob_low = np.concatenate([idx.blob_low, np.ascontiguousarray(d["blob_low"], dtype=np.uint16)])
                            idx.blob_orig = np.concatenate([idx.blob_orig, np.ascontiguousarray(d["blob_orig"], dtype=np.uint16)])
                            idx.off = np.concatenate([idx.off, doff])
                            idx.ln = np.concatenate([idx.ln, np.ascontiguousarray(d["ln"], dtype=np.uint32)])
                            idx.parent = np.concatenate([idx.parent, _integer_column(d["parent"], np.int32, np.int64)])
                            idx.flags = np.concatenate([idx.flags, np.ascontiguousarray(d["flags"], dtype=np.uint8)])
                            idx.frn = np.concatenate([idx.frn, np.ascontiguousarray(d["frn"], dtype=np.uint64)])
                            idx.size = np.concatenate([idx.size, np.ascontiguousarray(d["size"], dtype=np.int64)])
                            idx.mtime = np.concatenate([idx.mtime, np.ascontiguousarray(d["mtime"], dtype=np.float64)])
                            idx._root_of = np.concatenate([idx._root_of, np.ascontiguousarray(d["root_of"], dtype=np.int32)])
                            idx._depth = np.concatenate([idx._depth, np.ascontiguousarray(d["depth"], dtype=np.int32)])
                            idx._delta_n = dn
                        except Exception:
                            pass
                    try:
                        d.close()
                    except Exception:
                        pass
            except Exception:
                pass
            saved_bloom = idx._bloom if idx._delta_n == 0 else None
            idx._invalidate()
            idx._pack_memory()
            idx._bloom = saved_bloom
            if idx._journal_state:
                idx.dedupe_live_rows()
            return idx
        except Exception as exc:
            if callable(on_error):
                try:
                    on_error(exc)
                except Exception:
                    pass
            try:
                base.close()
            except Exception:
                pass
            return cls(on_error=on_error)



class _CSharpIndex:
    """Owned LDI1 mmap worker. The pipe carries row IDs, never trusted paths.

    LDI1: <4sIQ> header, starts/lengths u32[N], parents i32[N], flags
    u8[N], suffix IDs u32[N], UTF-8 lowercase arena. A suffix ID is its
    arena byte offset plus one; zero denotes no suffix. Python owns
    spelling, roots, metadata, ACLs and the final global ranking.
    """
    def __init__(self, cache_dir=None):
        import atexit
        self.index = self.process = None
        self.layout = self.generation = self.n_exported = 0
        self._lock = threading.RLock()
        self._responses = queue.Queue()
        self._directory = tempfile.TemporaryDirectory(prefix='localdocs-cs-', dir=cache_dir)
        self._job = None
        self.export_s = 0.0
        atexit.register(self.stop)

    @staticmethod
    def _compiler():
        import shutil
        for framework in ('Framework64', 'Framework'):
            path = os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Microsoft.NET',
                                framework, 'v4.0.30319', 'csc.exe')
            if os.path.isfile(path):
                return path
        return shutil.which('csc.exe')

    def _own_process(self):
        # Kill-on-job-close also covers abrupt interpreter exit on Windows.
        if os.name != 'nt':
            return
        import ctypes
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', w.DWORD), ('minimum', ctypes.c_size_t),
                        ('maximum', ctypes.c_size_t), ('active', w.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', w.DWORD),
                        ('scheduling', w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ('read_ops', 'write_ops', 'other_ops', 'read', 'write', 'other')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        job = kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x2000
        if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            if job:
                kernel.CloseHandle(job)
            raise OSError('cannot create native ownership job')
        if not kernel.AssignProcessToJobObject(job, w.HANDLE(int(self.process._handle))):
            kernel.CloseHandle(job)
            raise OSError('cannot own native worker')
        self._job = (kernel, job)

    def export_and_load(self, idx):
        import struct
        import subprocess
        compiler = self._compiler()
        if not compiler:
            raise RuntimeError('csc unavailable')
        directory = self._directory.name
        exe = os.path.join(directory, 'ldeng.exe')
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native', 'ldeng.cs')
        result = subprocess.run([compiler, '-nologo', '-unsafe', '-optimize+', '-platform:x64',
                                 '-out:' + exe, source], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError('csc failed: ' + result.stdout.decode(errors='replace'))
        # Capture replace-only arenas and detached metadata under the mutation lock.
        with idx._lock:
            arena = idx.blob_low
            original = idx.blob_orig
            off, lengths = idx.off.copy(), idx.ln.copy()
            parents, flags = idx.parent.copy(), idx.flags.copy()
            layout, revision = idx._turbo_layout, idx._turbo_revision
        self.has_expansion = bool(np.any(original == 0x0130))
        started = time.perf_counter()
        units = np.asarray(arena)
        # One codec pass, no per-name Python loop or fixed-width string matrix.
        widths = np.where(units < 128, 1, np.where(units < 2048, 2, 3)).astype(np.uint32)
        pairs = np.flatnonzero((units[:-1] >= 0xD800) & (units[:-1] <= 0xDBFF) &
                               (units[1:] >= 0xDC00) & (units[1:] <= 0xDFFF))
        widths[pairs], widths[pairs + 1] = 4, 0
        total = int(widths.sum(dtype=np.uint64))
        if total >= 2 ** 32:
            raise ValueError('native arena exceeds u32')
        cumulative = np.empty(len(units) + 1, dtype=np.uint32)
        cumulative[0] = 0
        np.cumsum(widths, dtype=np.uint32, out=cumulative[1:])
        starts = cumulative[off]
        byte_lengths = cumulative[off + lengths] - starts
        data = units.tobytes().decode('utf-16-le', 'surrogatepass').encode('utf-8', 'surrogatepass')
        if len(data) != total:
            raise ValueError('native UTF-8 offset mismatch')
        snapshot = os.path.join(directory, 'names.bin')
        with open(snapshot, 'wb', buffering=0) as stream:
            stream.write(struct.pack('<4sIQ', b'LDI1', len(off), total))
            starts.astype('<u4', copy=False).tofile(stream)
            byte_lengths.astype('<u4', copy=False).tofile(stream)
            parents.astype('<i4').tofile(stream)
            flags.tofile(stream)
            # String-table IDs without decoding N extensions or allocating an
            # N-by-max-suffix matrix. Equal spellings may have distinct IDs.
            dots = np.flatnonzero(units == ord('.'))
            for first in range(0, len(off), 65536):
                ends = off[first:first + 65536] + lengths[first:first + 65536]
                ids = np.zeros(len(ends), dtype='<u4')
                if dots.size:
                    at = np.searchsorted(dots, ends, side='left') - 1
                    last = dots[np.maximum(at, 0)]
                    valid = ((at >= 0) & (last >= off[first:first + len(ends)]) &
                             (last + 1 < ends))
                    ids[valid] = cumulative[last[valid] + 1] + 1
                ids.tofile(stream)
            stream.write(data)
        self.export_s = time.perf_counter() - started
        del units, widths, cumulative, data, starts, byte_lengths
        self.process = self._spawn_worker(exe, snapshot)
        self._own_process()
        process = self.process
        buffer = bytearray()
        def take(count):
            # One blocking read hands over up to 64 KB: a reply of thousands of rows would otherwise
            # re-acquire the GIL twice per row, which stalls for seconds while another thread is busy.
            while len(buffer) < count:
                part = process.stdout.read(65536)  # raw pipe (bufsize=0): returns what is available
                if not part:
                    raise EOFError('native worker disconnected')
                buffer.extend(part)
            out = bytes(buffer[:count])
            del buffer[:count]
            return out
        def reader():
            try:
                while True:
                    status, ws, count = struct.unpack('<iqi', take(16))
                    if not 0 <= count <= 4097:
                        raise ValueError('invalid native count')
                    rows = []
                    for _ in range(count):
                        length, = struct.unpack('<i', take(4))
                        if not 0 <= length <= 4096:
                            raise ValueError('invalid native item')
                        rows.append(take(length).decode('utf-8'))
                    self._responses.put((status, rows))
            except Exception as exc:
                self._responses.put(exc)
        threading.Thread(target=reader, name='localdocs-cs-pipe', daemon=True).start()
        self._response(10.0)
        self.index = types.SimpleNamespace(_turbo_lineage=idx._turbo_lineage)
        self.layout = layout
        self.revision = revision
        self.n_exported = len(off)
        self.generation = time.monotonic_ns()

    @staticmethod
    def _spawn_worker(exe, snapshot):
        import subprocess
        try:
            # The child inherits the error mode: a broken/blocked exe must fail silently, never show a
            # 'This application could not be started' dialog to the user.
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)  # FAILCRITICALERRORS | NOGPFAULTERRORBOX | NOOPENFILEERRORBOX
        except Exception:
            pass
        return subprocess.Popen([exe, snapshot], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def _response(self, timeout):
        answer = self._responses.get(timeout=timeout)
        if isinstance(answer, Exception):
            raise answer
        status, rows = answer
        if status:
            raise RuntimeError('native error: ' + '; '.join(rows))
        return rows

    def search(self, query, skip):
        import struct
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError('native worker stopped')
            # A single request includes the complete small overlay skip set.
            data = query.encode('utf-8', 'surrogatepass')
            packet = (struct.pack('<ii', -3, len(skip)) +
                      np.asarray(sorted(skip), dtype='<i4').tobytes() +
                      struct.pack('<ii', len(data), 4097) + data)
            deadline = time.monotonic() + float(os.environ.get('LOCALDOCS_NATIVE_TIMEOUT', '2.0'))  # generous: the reader threads share the GIL with background preparation
            # Writes are bounded too: a stuck worker must not block the caller.
            done = queue.Queue(maxsize=1)
            process = self.process
            def send():
                try:
                    view = memoryview(packet)
                    while view:
                        count = process.stdin.write(view)
                        if not count:
                            raise EOFError('native pipe closed')
                        view = view[count:]
                    done.put(None)
                except Exception as exc:
                    done.put(exc)
            if len(packet) <= 16384:
                # Small request (the normal case): write it from the calling thread. A helper thread per query adds a
                # GIL hand-off that stalls for seconds while background Python work holds the GIL.
                view = memoryview(packet)
                while view:
                    count = process.stdin.write(view)
                    if not count:
                        raise EOFError('native pipe closed')
                    view = view[count:]
            else:
                threading.Thread(target=send, name='localdocs-cs-send', daemon=True).start()
                error = done.get(timeout=max(.001, deadline - time.monotonic()))
                if error is not None:
                    raise error
            rows = self._response(max(.001, deadline - time.monotonic()))
            return [int(k) for k in rows]

    def stop(self):
        import atexit
        atexit.unregister(self.stop)
        process, self.process = self.process, None
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except Exception:
                pass
            for pipe in (process.stdin, process.stdout):
                try:
                    pipe.close()
                except Exception:
                    pass
        if self._job is not None:
            kernel, handle = self._job
            kernel.CloseHandle(handle)
            self._job = None
        # TemporaryDirectory cleanup is deferred until mappings/pipes are gone.
        try:
            self._directory.cleanup()
        except OSError:
            pass


class FastQuickIndex:
    """Thin adapter exposing the QuickIndex public API over NameIndex.

    Same methods: start, stop, rescan_now, search(query, limit, only_roots),
    status. Search dicts carry the same keys:
    path, name, folder, size, mtime, root.
    """

    def __init__(self, roots_provider, max_entries=600000, workers=8, cache_path=None, turbo=None):
        self._engine = os.environ.get('LOCALDOCS_ENGINE', 'auto').lower()
        if self._engine not in ('auto', 'cs', 'py'):
            self._engine = 'auto'
        self._turbo_enabled = turbo
        self._turbo = None
        self._turbo_thread = None
        self._turbo_guard = threading.RLock()
        self._turbo_until = 0.0
        self._turbo_error = None
        self._turbo_native_queries = 0
        self._turbo_fallbacks = 0
        self._roots_provider = roots_provider
        self._cache_path = cache_path
        try:
            self._max_entries = max(1, int(max_entries))
        except Exception:
            self._max_entries = 600000
        try:
            self._workers = max(1, min(int(workers), 32))
        except Exception:
            self._workers = 8
        self._lock = threading.RLock()
        self._scan_lock = threading.Lock()
        self._index = NameIndex()
        self._roots = []
        self._scanning = False
        self._done = False
        self._thread = None
        self._stop = threading.Event()
        self._usn_sessions = []
        self._restore_pending = False
        self._snapshot_validated = True
        self._start_mode = "walk"
        self._last_scan = None
        self._usn_errors = []
        self._watchers = []
        self._watch_roots = []
        self._watch_lock = threading.Lock()
        self._watch_generation = 0
        self._watch_events = queue.Queue(maxsize=1024)
        self._watch_reconcile = threading.Event()
        self._wake = threading.Event()

    def start(self):
        try:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._stop.clear()
            self._load_cache()
            t = threading.Thread(target=self._loop, name="localdocs-fastquick", daemon=True)
            self._thread = t
            t.start()
            return t
        except Exception:
            return None

    def stop(self):
        try:
            self._stop.set()
            with self._turbo_guard:
                if self._turbo is not None:
                    self._turbo.stop()
            self._wake.set()
            self._close_watchers()
            if self._thread is None or not self._thread.is_alive():
                with self._scan_lock:
                    self._close_usn()
            return True
        except Exception:
            return False

    def _load_cache(self):
        """Map cached columns; expose only a snapshot at the live journal head."""
        try:
            if not self._cache_path or not os.path.exists(str(self._cache_path)):
                return
            idx = NameIndex.load(self._cache_path)
            if idx is not None and len(idx) and idx._journal_state:
                valid = True
                letters = {ntpath.splitdrive(root)[0].upper() for root in self._resolve_roots()}
                if letters != {str(s[0]).upper() + ":" for s in idx._journal_state}:
                    return
                for letter, journal_id, cursor in idx._journal_state:
                    source = _usn_source(letter)
                    if source is None:
                        return
                    try:
                        info = source.query()
                        if not self._journal_valid(info, int(journal_id), int(cursor)):
                            return
                        valid &= int(info["next_usn"]) == int(cursor)
                    finally:
                        source.close()
                with self._lock:
                    self._snapshot_validated = valid
                    self._index = idx
                    self._restore_pending = bool(idx._journal_state)
                    self._done = True
                    if idx._journal_state:
                        self._roots = self._resolve_roots()
                self._native_maintenance(idx)
                threading.Thread(target=idx.prebuild_event_tables, name='localdocs-event-tables', daemon=True).start()
                if not (self._native_wanted(idx) and os.environ.get('LOCALDOCS_PREP', 'lazy') != 'eager'):
                    idx.start_search_warmup()
        except Exception:
            pass

    def _save_cache(self, idx):
        try:
            if self._cache_path and len(idx):
                idx.save(self._cache_path)
        except Exception:
            pass

    def _loop(self):
        try:
            while not self._stop.is_set():
                self._watch_reconcile.clear()
                try:
                    with self._scan_lock:
                        self._scan()
                except Exception:
                    pass
                # With a live USN journal the index is kept current record by record (a replaced/wrapped journal forces a
                # rebuild by itself): a full MFT re-enumeration every 10 minutes only burns CPU and ~1-2 GB of churn.
                deadline = time.monotonic() + (21600.0 if self._usn_sessions else 600.0)
                while not self._stop.is_set():
                    self._wake.clear()
                    if self._watch_reconcile.is_set():
                        break
                    with self._scan_lock:
                        self._drain_watch_events()
                    if self._watch_reconcile.is_set():
                        break
                    self._native_maintenance(self._index)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    delay = min(1.0, remaining) if self._usn_sessions or self._native_wanted(self._index) else remaining
                    if self._stop.is_set():
                        break
                    self._wake.wait(delay)
                    with self._scan_lock:
                        if self._usn_sessions:
                            self._poll_usn()
        finally:
            self._close_watchers()
            self._close_usn()

    def _close_watchers(self):
        with self._watch_lock:
            self._close_watchers_locked()

    def _close_watchers_locked(self):
        self._watch_generation += 1
        watchers, self._watchers = self._watchers, []
        self._watch_roots = []
        for watcher in watchers:
            try:
                watcher.stop()
            except Exception:
                pass
        while True:
            try:
                self._watch_events.get_nowait()
            except queue.Empty:
                break

    def _sync_watchers(self, roots):
        """Keep armed readers across reconciles; callbacks only enqueue work."""
        with self._watch_lock:
            desired = list(roots) if not self._usn_sessions else []
            if self._stop.is_set():
                desired = []
            if desired == self._watch_roots:
                return
            self._close_watchers_locked()
            self._watch_roots = desired
            if not desired:
                return
            try:
                from localdocs_dirwatch import DirWatcher
            except Exception:
                return  # Preserve the periodic walker if notifications are unavailable.
            generation = self._watch_generation

            def enqueue(root, events):
                if self._stop.is_set() or generation != self._watch_generation:
                    return
                if any(event.get("overflow") for event in events):
                    self._watch_reconcile.set()
                else:
                    try:
                        self._watch_events.put_nowait((generation, root, events))
                    except queue.Full:
                        self._watch_reconcile.set()
                self._wake.set()

            for root in desired:
                state = {"started": False, "failed": False}

                def failed(error, state=state):
                    state["failed"] = True
                    if (state["started"] and not self._stop.is_set()
                            and generation == self._watch_generation):
                        self._watch_reconcile.set()
                        self._wake.set()

                watcher = None
                try:
                    watcher = DirWatcher(
                        root, lambda events, root=root: enqueue(root, events),
                        recursive=True, debounce_s=0.25, on_error=failed)
                    if watcher.start() and not state["failed"]:
                        state["started"] = True
                        self._watchers.append(watcher)
                        continue
                except Exception:
                    pass
                if watcher is not None:
                    watcher.stop()
            if self._watchers:
                # Close the scan-to-arm gap once, with readers kept armed.
                self._watch_reconcile.set()
                self._wake.set()

    @staticmethod
    def _watch_path_allowed(path, root, is_dir=False):
        if not _prefix_match(path, root) or _norm(path) == _norm(root):
            return False
        parts = os.path.relpath(path, root).split(os.sep)
        directories = parts if is_dir else parts[:-1]
        if any(part.startswith(".") or part.casefold() in _SKIP_NAMES
               for part in directories):
            return False
        # Do not enter directory symlinks/junctions that the walker skips.
        current = root
        for part in directories:
            current = os.path.join(current, part)
            if os.path.islink(current) or (getattr(os.path, "isjunction", lambda p: False)(current)):
                return False
        return is_dir or os.path.splitext(path)[1].lower() not in _SKIP_EXT

    def _drain_watch_events(self):
        if self._usn_sessions or self._stop.is_set():
            return
        paths = {}
        # Bound queue draining even if producers never go idle. DirWatcher
        # already caps debounce with fixed 250ms windows.
        for _ in range(256):
            try:
                generation, root, events = self._watch_events.get_nowait()
            except queue.Empty:
                break
            if generation != self._watch_generation:
                continue
            for event in events:
                if event.get("action") not in ("added", "removed", "renamed_old", "renamed_new"):
                    continue
                path = event.get("path", "")
                if self._watch_path_allowed(path, root, os.path.isdir(path)):
                    # Last spelling wins for case-only renames. Check final
                    # filesystem state because debounce may merge delete/create.
                    paths[_norm(path)] = (path, root)
        if not self._watch_events.empty():
            self._wake.set()
        if not paths:
            return
        try:
            adds, missing, refreshed_dirs = {}, set(), []
            for key, (path, root) in paths.items():
                if not os.path.exists(path):
                    missing.add(key)
                    continue
                is_dir = os.path.isdir(path)
                adds[key] = {"path": path, "is_dir": is_dir}
                if is_dir:
                    refreshed_dirs.append(key)
                    def walk_error(error):
                        raise error
                    for folder, dirs, files in os.walk(path, onerror=walk_error):
                        if self._stop.is_set():
                            return
                        dirs[:] = [name for name in dirs if self._watch_path_allowed(
                            os.path.join(folder, name), root, True)]
                        for name in dirs + files:
                            full = os.path.join(folder, name)
                            directory = name in dirs
                            if self._watch_path_allowed(full, root, directory):
                                adds[_norm(full)] = {"path": full, "is_dir": directory}
            with self._lock:
                idx = self._index
                with idx._lock:
                    mapping = dict(idx._ensure_map())
                    subtrees = [path for path in missing if path in mapping
                                and int(idx.flags[mapping[path]]) & FLAG_DIR]
                    removes = [path for path in mapping if path in missing
                               or any(_prefix_match(path, folder) for folder in subtrees)
                               or (path not in adds and any(_prefix_match(path, folder)
                                                           for folder in refreshed_dirs))]
                    # A directory's children must retain their parent links
                    # when only its spelling changes.
                    renames = [(idx.path_of(mapping[key]), item["path"])
                               for key, item in adds.items() if key in mapping
                               and idx.path_of(mapping[key]) != item["path"]]
                    if renames:
                        idx.apply_events(renames=renames)
                    if removes:
                        idx.apply_events(removes=removes)
                    if adds:
                        idx.apply_events(adds=list(adds.values()))
        except Exception:
            self._watch_reconcile.set()
            self._wake.set()

    def _close_usn(self):
        sessions, self._usn_sessions = self._usn_sessions, []
        for session in sessions:
            try:
                session["source"].close()
            except Exception:
                pass

    @staticmethod
    def _journal_valid(info, journal_id, cursor):
        return (info is not None and int(info["journal_id"]) == journal_id
                and int(info["first_usn"]) <= cursor <= int(info["next_usn"]))

    def _build_usn(self, roots):
        """All-or-walker: never publish a partially enumerated set of roots."""
        sources, indexes = [], []
        try:
            letters = []
            for root in roots:
                drive = ntpath.splitdrive(root)[0]
                if len(drive) != 2 or drive[1] != ":":
                    return None
                letter = drive[0].upper()
                if letter not in letters:
                    letters.append(letter)
            for letter in letters:
                source = _usn_source(letter)
                if source is None:
                    return None
                session = {"source": source, "letter": letter}
                sources.append(session)
                info = source.query()
                journal_id, cursor = int(info["journal_id"]), int(info["next_usn"])
                vol = source.enumerate()
                if not len(vol["frn"]):
                    raise ValueError("empty MFT enumeration")
                if not self._journal_valid(source.query(), journal_id, cursor):
                    raise ValueError("journal changed during enumeration")
                volume_root = letter + ":" + os.sep
                errors = []
                idx = NameIndex.from_usn(vol, volume_root, on_error=errors.append)
                if errors or not len(idx):
                    raise ValueError("invalid MFT index")
                reader = source.reader(journal_id, cursor)
                if reader.last_error:
                    raise RuntimeError(reader.last_error)
                session.update(nodes=None, include_tombs=True, root_id=len(indexes), root=volume_root, reader=reader,
                               journal_id=journal_id)
                indexes.append(idx)
            if len(indexes) == 1:
                # The common one-volume case needs no full second column set.
                combined = indexes.pop()
                combined._on_error = self._usn_errors.append
                self._usn_sessions = sources
                sources = []
                del vol
                self._update_journal_state(combined)
                return combined
            # Merge columns, retaining parent and root offsets across volumes.
            columns = {k: [] for k in ("blob_low", "blob_orig", "off", "ln", "parent",
                                       "flags", "frn", "size", "mtime", "_root_of", "_depth")}
            count, units = 0, 0
            for root_id, idx in enumerate(indexes):
                for key in columns:
                    arr = getattr(idx, key)
                    if key == "off":
                        arr = arr + np.uint64(units)
                    elif key == "parent":
                        arr = _integer_column(
                            np.where(arr >= 0, arr.astype(np.int64) + count, -1),
                            np.int32, np.int64)
                    elif key == "_root_of":
                        arr = np.full(len(idx), root_id, dtype=np.int32)
                    columns[key].append(arr)
                count += len(idx)
                units += len(idx.blob_orig)
            combined = NameIndex(on_error=self._usn_errors.append)
            for key, parts in columns.items():
                setattr(combined, key, np.concatenate(parts))
            combined._roots = [s["root"] for s in sources]
            combined._base_n = count
            combined._pack_memory()
            del columns, indexes, vol  # merged columns own the published snapshot
            self._usn_sessions = sources
            sources = []  # ownership transferred
            self._update_journal_state(combined)
            return combined
        except Exception:
            return None
        finally:
            for session in sources:
                try:
                    session["source"].close()
                except Exception:
                    pass

    def _update_journal_state(self, idx):
        idx._journal_state = [(s["letter"], s["journal_id"], int(s["reader"].start_usn))
                              for s in self._usn_sessions]

    def _copy_usn_index(self, published):
        """Own mutable columns; share the replace-only name arenas."""
        idx = NameIndex(on_error=self._usn_errors.append)
        idx._memory_mode, idx._block = published._memory_mode, published._block
        with published._lock:
            for key in ("blob_low", "blob_orig", "off", "ln", "parent", "flags",
                        "frn", "size", "mtime", "_root_of", "_depth"):
                value = getattr(published, key)
                # Name arenas are only replaced on mutation, never written into.
                setattr(idx, key, value if key in ("blob_low", "blob_orig") else value.copy())
            idx._turbo_lineage = published._turbo_lineage
            idx._turbo_layout = published._turbo_layout
            idx._turbo_revision = published._turbo_revision
            idx._turbo_touched = dict(published._turbo_touched)
            idx._turbo_last_touch = published._turbo_last_touch
            idx._roots = list(published._roots)
            idx._journal_state = list(published._journal_state)
            idx._base_n = published._base_n
            idx._delta_n = published._delta_n
            idx._layout_dirty = published._layout_dirty
            idx._ext_use = dict(published._ext_use)
            idx._dir_use = dict(published._dir_use)
            # Arrays in the lookup base are immutable. Only bounded overlays
            # belong to each transaction; never copy a volume-sized dict.
            idx._event_base = getattr(published, "_event_base", None)
            idx._event_names = dict(getattr(published, "_event_names", {}))
            idx._event_refs = dict(getattr(published, "_event_refs", {}))
        return idx

    def _restore_usn(self, roots):
        """Reconstruct journal parent lookup from the snapshot, without MFT I/O."""
        sessions = []
        try:
            with self._lock:
                published = self._index
            # Replay is a transaction. A failed later batch must leave the
            # instant-start snapshot intact until _scan publishes a fallback.
            idx = self._copy_usn_index(published)
            letters = {ntpath.splitdrive(root)[0].upper() for root in roots}
            stored = [str(state[0]).upper() for state in idx._journal_state]
            if (letters != {letter + ":" for letter in stored}
                    or len(set(stored)) != len(stored)
                    or len(idx._roots) != len(stored)):
                return None
            for letter, journal_id, cursor in idx._journal_state:
                source = _usn_source(letter)
                if source is None:
                    raise ValueError("USN source unavailable")
                session = {"source": source, "letter": letter}
                sessions.append(session)
                info = source.query()
                journal_id, cursor = int(journal_id), int(cursor)
                if not self._journal_valid(info, journal_id, cursor):
                    raise ValueError("snapshot journal expired")
                root_id = next(k for k, root in enumerate(idx._roots)
                               if ntpath.splitdrive(root)[0].upper() == letter.upper() + ":")
                # Validate identities eagerly without constructing Python nodes.
                live_rows = np.flatnonzero((idx._root_of == root_id) &
                                            ((idx.flags & FLAG_TOMB) == 0))
                refs = idx.frn[live_rows] & np.uint64(0x0000FFFFFFFFFFFF)
                rows = live_rows[refs != 5]
                parents = idx.parent[rows]
                if (np.any(idx.frn[rows] == 0) or
                        np.any(idx.frn[parents[parents >= 0]] == 0)):
                    raise ValueError("snapshot lacks journal identities")
                del live_rows, refs, rows, parents
                reader = source.reader(journal_id, cursor)
                if reader.last_error or getattr(reader, "_fast_errors", []):
                    raise RuntimeError(reader.last_error or "reader initialization failed")
                session.update(nodes=None, root_id=root_id, root=idx._roots[root_id], reader=reader,
                               journal_id=journal_id)
                if not self._journal_valid(source.query(), journal_id, cursor):
                    raise ValueError("journal changed while opening reader")
                # Drain to the startup watermark; later events use live polling.
                target = int(info["next_usn"])
                while int(reader.start_usn) < target:
                    if self._stop.is_set():
                        raise RuntimeError("restore stopped")
                    self._read_usn(idx, session)
            if self._stop.is_set():
                raise RuntimeError("restore stopped")
            self._usn_sessions = sessions
            sessions = []
            self._update_journal_state(idx)
            return idx
        except Exception:
            return None
        finally:
            for session in sessions:
                try:
                    session["source"].close()
                except Exception:
                    pass

    @staticmethod
    def _node_path(session, frn):
        parts, seen = [], set()
        nodes = session["nodes"]
        while frn != 5:
            if frn in seen or frn not in nodes:
                raise ValueError("unresolved journal parent")
            seen.add(frn)
            parent, name, _ = nodes[frn]
            parts.append(name)
            frn = parent
        return os.path.join(session["root"], *reversed(parts))

    def _apply_usn_record(self, idx, session, record, blob):
        reference = int(record["frn"])
        frn = reference & 0x0000FFFFFFFFFFFF
        if frn == 5:
            return
        reason = int(record["reason"])
        if not reason & (0x100 | 0x200 | 0x2000):
            return
        # Idle volumes keep no per-file Python tuples/strings/dict. Rebuild
        # once, from the current columns, on the first actionable journal record.
        if session["nodes"] is None:
            nodes = {}
            root_id = session["root_id"]
            # Bulk rebuild: one local reference per column (kept alive for the whole pass) and one decoded
            # name buffer, instead of millions of per-row NumPy/arena calls that pin the GIL for ~20 s.
            n_rows = len(idx)
            root_of = np.asarray(idx._root_of)[:n_rows]
            flags_all = np.asarray(idx.flags)[:n_rows]
            frn_all = np.asarray(idx.frn)[:n_rows].astype(np.int64) & 0x0000FFFFFFFFFFFF
            parent_all = np.asarray(idx.parent)[:n_rows].astype(np.int64)
            off_all = np.asarray(idx.off)[:n_rows].astype(np.int64)
            ln_all = np.asarray(idx.ln)[:n_rows].astype(np.int64)
            keep = root_of == root_id
            if not session.get("include_tombs", False):
                keep &= (flags_all & FLAG_TOMB) == 0
            keep &= frn_all != 5
            rows = np.flatnonzero(keep)
            parent_rows = parent_all[rows]
            ref_rows = frn_all[rows]
            has_parent = parent_rows >= 0
            parent_frn = np.zeros(len(rows), dtype=np.int64)
            parent_frn[has_parent] = frn_all[parent_rows[has_parent]]
            if (ref_rows == 0).any() or (has_parent & (parent_frn == 0)).any():
                raise ValueError("snapshot lacks journal identities")
            parent_frn[~has_parent] = 5
            is_dir = (flags_all[rows] & FLAG_DIR) != 0
            raw = bytes(np.asarray(idx.blob_orig[0:len(idx.blob_orig)], dtype=np.uint16).tobytes())
            starts = (off_all[rows] * 2).tolist()
            ends = ((off_all[rows] + ln_all[rows]) * 2).tolist()
            for ref0, par0, dirflag, lo, hi in zip(ref_rows.tolist(), parent_frn.tolist(),
                                                    is_dir.tolist(), starts, ends):
                nodes[ref0] = (par0, raw[lo:hi].decode("utf-16-le", "surrogatepass"), dirflag)
            del raw
            session["nodes"] = nodes
        nodes = session["nodes"]
        old = nodes.get(frn)

        def emit(kind, event):
            pending = session.get("_event_batch")
            if pending is None:
                idx.apply_events(**{kind: [event]})
            else:
                # Preserve record order across operation kinds, while batching
                # the common create/noise stream into one column publication.
                if pending[0] != kind and pending[1]:
                    idx.apply_events(**{pending[0]: pending[1]})
                    pending[1] = []
                pending[0] = kind
                pending[1].append(event)

        if reason & 0x200:
            before = self._node_path(session, frn) if old else ""
            affected = {frn}
            if old and old[2]:
                while True:
                    more = {key for key, node in nodes.items() if node[0] in affected} - affected
                    if not more:
                        break
                    affected.update(more)
            # Descendant deletes have no supplied full FRN; use their old paths.
            for key in affected:
                path = self._node_path(session, key) if key in nodes else before
                emit("removes", {"path": path, "frn": reference} if key == frn else path)
            for key in affected:
                nodes.pop(key, None)
            return
        a, length = int(record["name_off"]), int(record["name_len"])
        name = blob[a:a + length].tobytes().decode("utf-16-le", "surrogatepass")
        parent = int(record["parent"]) & 0x0000FFFFFFFFFFFF
        # CREATE remains set on data/close records until the handle closes.
        # Deletes have already been handled; rename-new must still be applied.
        if (reason & 0x100 and not reason & 0x2000 and old is not None
                and old[:2] == (parent, name)):
            return
        before = self._node_path(session, frn) if old else ""
        nodes[frn] = (parent,
                      name, bool(int(record["attrs"]) & 0x10))
        after = self._node_path(session, frn)
        event = {"path": after, "is_dir": nodes[frn][2], "frn": reference,
                 "parent_frn": int(record["parent"])}
        if old and before != after and reason & 0x2000:
            event.update(old=before, new=after)
            emit("renames", event)
        elif not old or before != after or reason & 0x100:
            emit("adds", event)

    def _read_usn_batch(self, session):
        """Read and validate volume I/O without holding the publication lock."""
        reader, source = session["reader"], session["source"]
        info = source.query()
        cursor = int(reader.start_usn)
        if (reader.last_error or getattr(reader, "_fast_errors", [])
                or not self._journal_valid(info, session["journal_id"], cursor)):
            raise ValueError("journal replaced or overflowed")
        if int(info["next_usn"]) == cursor:
            return None
        records, blob = reader.read(timeout_ms=0)
        if (reader.last_error or getattr(reader, "_fast_errors", [])
                or int(reader.start_usn) <= cursor):
            raise RuntimeError(reader.last_error or "journal did not advance")
        if not self._journal_valid(source.query(), session["journal_id"],
                                   int(reader.start_usn)):
            raise ValueError("journal changed during read")
        return records, blob

    def _apply_usn_batch(self, idx, session, batch):
        records, blob = batch
        session["_event_batch"] = [None, []]
        try:
            for record in records:
                self._apply_usn_record(idx, session, record, blob)
            kind, events = session["_event_batch"]
            if events:
                idx.apply_events(**{kind: events})
        finally:
            session.pop("_event_batch", None)
        if self._usn_errors:
            raise ValueError("journal batch apply failed")

    def _read_usn(self, idx, session):
        batch = self._read_usn_batch(session)
        if batch is None:
            return False
        self._apply_usn_batch(idx, session, batch)
        if not self._journal_valid(session["source"].query(), session["journal_id"],
                                   int(session["reader"].start_usn)):
            raise ValueError("journal changed or batch apply failed")
        return True

    def _poll_usn(self):
        """Called under _scan_lock; a discontinuity immediately rebuilds by walk."""
        try:
            self._usn_errors.clear()
            with self._lock:
                published = self._index
            idx = None
            for session in self._usn_sessions:
                batch = self._read_usn_batch(session)
                if batch is None:
                    continue
                # Idle polls need no column copy. Pending batches are applied
                # privately so even searches holding the old reference are safe.
                if idx is None:
                    idx = self._copy_usn_index(published)
                self._apply_usn_batch(idx, session, batch)
                if not self._journal_valid(session["source"].query(), session["journal_id"],
                                           int(session["reader"].start_usn)):
                    raise ValueError("journal changed during batch apply")
            if self._usn_errors:
                raise RuntimeError(str(self._usn_errors[0]))
            if self._stop.is_set():
                raise RuntimeError("poll stopped")
            if idx is not None:
                self._update_journal_state(idx)
                with self._lock:
                    self._index = idx
                self._start_search_preparation(idx)
                self._save_cache(idx)
            return True
        except Exception:
            self._close_usn()
            self._scan(force_walk=True)
            return False

    def rescan_now(self):
        try:
            with self._scan_lock:
                return self._scan()
        except Exception:
            return {"files": 0, "roots": []}

    def root_status(self):
        """Return configured-root health, including roots omitted by the walker."""
        try:
            raw = self._roots_provider() if callable(self._roots_provider) else self._roots_provider
            roots = list(dict.fromkeys(os.path.normpath(os.path.realpath(str(r)))
                                       for r in (raw or [])))
            with self._lock:
                idx = self._index
                sessions = list(self._usn_sessions)
                mode, last_scan = self._start_mode, self._last_scan
                walk_status = getattr(idx, "_walk_status", {})
                if not isinstance(walk_status, dict):
                    walk_status = {}
                out = []
                with idx._lock:
                    live = (idx.flags & FLAG_TOMB) == 0
                    for root in roots:
                        volume = ntpath.splitdrive(root)[0].upper().rstrip(":")
                        backend = ("snapshot+journal" if mode == "snapshot+journal" else "usn") if any(
                            str(s["letter"]).upper() == volume for s in sessions) else "walk"
                        root_id = next((i for i, r in enumerate(idx._roots) if _norm(r) == _norm(root)), None)
                        if root_id is None and backend != "walk":
                            root_id = next((i for i, r in enumerate(idx._roots)
                                            if ntpath.splitdrive(r)[0].upper().rstrip(":") == volume), None)
                        entries = int(np.count_nonzero(live & (idx._root_of == root_id))) if root_id is not None else 0
                        info = next((v for k, v in walk_status.items() if _norm(k) == _norm(root)), {})
                        if isinstance(info, str):
                            info = {"state": info}
                        if not isinstance(info, dict):
                            info = {}
                        state, error = info.get("state", "ok"), str(info.get("error") or "")
                        if backend == "walk" and not info:
                            try:
                                os.stat(root)
                            except PermissionError as exc:
                                state, error = "denied", str(exc)
                            except OSError as exc:
                                state, error = "offline", str(exc)
                        if state not in ("ok", "denied", "offline", "partial"):
                            state = "partial"
                        out.append(dict(root=root, backend=backend, state=state,
                                        entries=entries, last_scan=last_scan, error=error))
                return out
        except Exception:
            return []

    def status(self):
        try:
            with self._lock:
                n = len(self._index)
                roots = list(self._roots)
                scanning = self._scanning
                done = self._done
            if scanning and not done:
                phase = "scanning"
            elif done:
                phase = "ready"
            else:
                phase = "idle"
            return {"phase": phase, "files": n, "roots": roots,
                    "start_mode": self._start_mode, "root_status": self.root_status(),
                    "turbo": self._turbo_status()}
        except Exception:
            return {"phase": "idle", "files": 0, "roots": [],
                    "start_mode": self._start_mode, "root_status": self.root_status(),
                    "turbo": self._turbo_status()}

    def _resolve_roots(self):
        try:
            raw = self._roots_provider() if callable(self._roots_provider) else self._roots_provider
        except Exception:
            return []
        try:
            items = list(raw or [])
        except Exception:
            return []
        return _dedup_roots(items)

    def _scan(self, force_walk=False):
        self._close_usn()
        restore, self._restore_pending = self._restore_pending, False
        try:
            roots = self._resolve_roots()
        except Exception:
            roots = []
        with self._lock:
            self._roots = list(roots)
            if not self._done:
                self._scanning = True
        if not roots:
            with self._lock:
                self._index = NameIndex()
                self._snapshot_validated = True
                self._done = True
                self._scanning = False
                self._last_scan = time.time()
            self._sync_watchers([])
            return {"files": 0, "roots": []}

        def on_batch(snap):
            try:
                with self._lock:
                    self._index = snap
                    self._snapshot_validated = True
                    self._roots = list(roots)
            except Exception:
                pass

        try:
            self._usn_errors.clear()
            self._index._on_error = self._usn_errors.append
            idx = self._restore_usn(roots) if restore and not force_walk else None
            self._start_mode = "snapshot+journal" if idx is not None else "mft"
            if idx is None and not force_walk:
                idx = self._build_usn(roots)
            if idx is None:
                self._start_mode = "walk"
                idx = NameIndex.from_walk(roots, workers=self._workers,
                                          max_entries=self._max_entries,
                                          on_batch=on_batch)
        except Exception:
            idx = NameIndex()
        with self._lock:
            self._index = idx
            self._snapshot_validated = True
            self._roots = list(roots)
            self._done = True
            self._scanning = False
            self._last_scan = time.time()
        try:
            self._start_search_preparation(idx)
        except Exception:
            pass  # The published index remains searchable without the filter.
        self._save_cache(idx)
        self._sync_watchers(roots)
        try:
            return {"files": len(idx), "roots": list(roots)}
        except Exception:
            return {"files": 0, "roots": list(roots)}

    def _start_search_preparation(self, idx):
        """Only call after publication; queries use lazy caches until ready."""
        self._native_maintenance(idx)
        threading.Thread(target=idx.prebuild_event_tables, name='localdocs-event-tables', daemon=True).start()
        if self._native_wanted(idx) and os.environ.get('LOCALDOCS_PREP', 'lazy') != 'eager':
            return  # native engine serves the queries: skip ~45 s of GIL-heavy Python preparation (built lazily if a query falls back)
        def prepare():
            try:
                idx.build_trigrams()
                idx.start_search_warmup()
            except Exception as exc:
                idx._err(exc)
        threading.Thread(target=prepare, name='localdocs-search-prepare',
                         daemon=True).start()

    def _native_enabled(self):
        return self._engine == 'cs' or (self._engine == 'auto' and self._turbo_enabled is not False)

    def _native_wanted(self, idx):
        """auto mode only pays for the native process on large indexes (small roots are already instant in Python)."""
        if not self._native_enabled():
            return False
        if self._engine == 'cs':
            return True
        try:
            floor = int(os.environ.get('LOCALDOCS_CS_MIN_ENTRIES', '100000'))
        except ValueError:
            floor = 100000
        try:
            return len(idx) >= floor
        except Exception:
            return False

    def _native_matches(self, worker, idx):
        return (worker is not None and worker.index is not None and
                worker.index._turbo_lineage is idx._turbo_lineage and
                worker.layout == idx._turbo_layout and worker.process is not None and
                worker.process.poll() is None and not self._stop.is_set() and
                time.monotonic() >= self._turbo_until)

    def _turbo_status(self):
        t, idx = self._turbo, self._index
        ready = self._native_enabled() and self._native_matches(t, idx)
        with idx._lock:
            overlay = sum(v > t.revision for v in idx._turbo_touched.values()) if ready else len(idx._turbo_touched)
        return dict(active=bool(ready), native_ready=bool(ready),
                    engine='cs' if ready else 'py', requested_engine=self._engine,
                    generation=t.generation if t else 0,
                    rows=t.n_exported if ready else 0,
                    delta_rows=max(0, len(idx) - t.n_exported) if ready else len(idx),
                    touched_rows=overlay, overlay_rows=overlay,
                    last_error=self._turbo_error, native_queries=self._turbo_native_queries,
                    python_fallbacks=self._turbo_fallbacks, fallbacks=self._turbo_fallbacks,
                    export_s=t.export_s if t else None,
                    fallback_reasons=dict(getattr(self, '_turbo_reasons', {}) or {}),
                    fast_error=getattr(self, '_turbo_fast_error', None), worker_error=getattr(self, '_turbo_last_exc', None),
                    prof=list(getattr(self, '_turbo_prof', []) or [])[:60])

    def _turbo_failed(self, exc, worker=None):
        self._turbo_last_exc = repr(exc)[:300]
        with self._turbo_guard:
            self._turbo_error = str(exc)
            self._turbo_until = time.monotonic() + 60
            if worker is not None:
                worker.stop()
            if self._turbo is worker:
                self._turbo = None

    def _turbo_reexport(self, idx, worker=None):
        if not self._native_enabled() or self._stop.is_set() or time.monotonic() < self._turbo_until:
            return
        with self._turbo_guard:
            if self._turbo_thread is not None and self._turbo_thread.is_alive():
                return
            def run():
                fresh = None
                try:
                    folder = os.path.dirname(os.path.abspath(self._cache_path)) if self._cache_path else None
                    if folder and not os.path.isdir(folder):
                        folder = None
                    fresh = _CSharpIndex(folder)
                    fresh.export_and_load(idx)
                    # Warm the native process BEFORE it serves: JIT of the scan paths and page-faulting of the mapped
                    # snapshot cost 400-1100 ms on the first real queries (measured), ~15 ms afterwards.
                    for _q in ('zzzzqqqqxx', 'e', 'a', '*.pdf', 'dll', chr(1) + 'windows' + chr(10) + 'dll', 'config'):
                        try:
                            fresh.search(_q, set())
                        except Exception:
                            break
                    with self._turbo_guard:
                        current = self._index
                        if (self._stop.is_set() or
                                current._turbo_lineage is not idx._turbo_lineage or
                                current._turbo_layout != fresh.layout):
                            fresh.stop()
                            return
                        old, self._turbo = self._turbo, fresh
                        with current._lock:
                            current._turbo_touched = {k: v for k, v in current._turbo_touched.items()
                                                      if v > fresh.revision}
                        self._turbo_error = None
                    if old is not None:
                        # Queries hold this same guard across their transport.
                        old.stop()
                except Exception as exc:
                    if fresh is not None:
                        fresh.stop()
                    self._turbo_failed(exc)
            self._turbo_thread = threading.Thread(target=run, name='localdocs-cs-export', daemon=True)
            self._turbo_thread.start()

    def _native_maintenance(self, idx):
        if not self._native_wanted(idx):
            return
        worker = self._turbo
        if worker is not None and worker.process is not None and worker.process.poll() is not None:
            self._turbo_failed(RuntimeError('native worker exited'), worker)
            return
        if not self._native_matches(worker, idx):
            self._turbo_reexport(idx)
            return
        with idx._lock:
            touched = sum(v > worker.revision for v in idx._turbo_touched.values())
            touched += max(0, len(idx) - worker.n_exported)  # rows appended after the export are overlay rows too (startup tail)
            idle = time.monotonic() - idx._turbo_last_touch
        # The overlay is filtered in Python (~30 us per touched row): keep it small by re-exporting soon (export ~0.2 s
        # at 2.6M rows, background, rate-limited), instead of waiting for 5000 rows / 60 s idle.
        if touched > 300 or (touched and idle >= 3):
            now = time.monotonic()
            if now - getattr(self, '_turbo_reexport_at', -1e9) >= 2.0:
                self._turbo_reexport_at = now
                self._turbo_reexport(idx, worker)

    def _native_fast_hits(self, idx, toks, rows, touched, limit, prefixes, kinds, spec=None, saturated=False):
        """Plain-query fast answer straight from native candidates (Everything-style): tier-rank only the first
        <=4097 native matches (+ overlay rows changed since the snapshot), build paths for the best `limit` rows.
        Returns a list of hit dicts or None when this path cannot answer exactly (caller falls back)."""
        qfull = " ".join(toks)
        n = len(idx)
        ids = np.fromiter((k for k in set(rows) if 0 <= k < n and k not in touched), dtype=np.int64)
        extra = [k for k in touched if 0 <= k < n and all(t in idx._low_name(k) for t in toks)]
        if extra:
            ids = np.unique(np.concatenate([ids, np.asarray(extra, dtype=np.int64)]))
        if not ids.size:
            return []
        # Per-candidate Python work (name decode + tiering) costs ~100 us on the compressed name arena: rank only the
        # first few hundred native matches (index order), like Everything which does not rank globally at all.
        ids = np.sort(ids)
        cap = max(4 * limit, 400)
        meta_spec = spec is not None and any(spec.get(k) is not None for k in ('min_size', 'max_size', 'min_t', 'max_t'))
        if spec is not None and (spec.get('path') or spec.get('folder') or meta_spec):
            cap = 4097  # path predicates reject many superset rows: verify the whole native sample (~120 ms) instead of falling back (~1 s)
        ids = ids[(idx.flags[ids] & FLAG_TOMB) == 0][:cap]
        kindset = None
        if kinds:
            kindset = {str(k).lower() if str(k).startswith(".") else "." + str(k).lower() for k in kinds}
        scored = []
        ext_fixed = None
        if spec is not None:
            spec_ext = list(spec.get('ext') or ())
            if len(spec_ext) == 1 and spec_ext[0]:
                ext_fixed = '.' + str(spec_ext[0]).lower().lstrip('.')
        flags_ids = idx.flags[ids]
        for k, fl in zip(ids.tolist(), flags_ids.tolist()):
            if spec is not None and fl & FLAG_DIR:
                continue  # operator queries list files only (same as the Python path)
            if toks:
                low = idx._low_name(k)
                rank0 = min(idx._rank_one(low, qfull, toks, None), 4)
                low_len = len(low)
            else:
                rank0, low_len = 0, int(idx.ln[k])  # operator-only query: every candidate ties on name rank
            ext = ext_fixed if ext_fixed is not None else os.path.splitext(idx._orig_name(k))[1].lower()
            if kindset and ext not in kindset:
                continue
            score = float(rank0) * 100.0 + min(max(int(idx._depth[k]), 0), 20) * 0.5
            if ext in _DOC_EXT:
                score -= 8.0
            if (fl & FLAG_JUNK) or (ext in _PENALTY_EXT):
                score += 20.0
            if fl & FLAG_HIDDEN:
                score += 10.0
            scored.append((score, low_len, k))
        scored.sort()
        out, seen = [], set()
        pref = None if prefixes is None else [os.path.normcase(os.path.normpath(str(p))) for p in prefixes]
        for _score, _ln, k in scored:
            path = idx._fuzzy_path(k)  # '' for tombstoned/orphaned/cyclic rows: no volume-wide reachability pass
            if not path:
                continue
            pk = os.path.normcase(os.path.normpath(path))
            if pk in seen:
                continue
            if pref is not None and not any(
                    (idx._contains_path(pk, p) if idx._journal_state else pk == p or pk.startswith(p + os.sep))
                    for p in pref):
                continue
            if spec is not None:
                if toks and not all(t in pk for t in toks):
                    continue
                from localdocs_filters import matches as _spec_matches
                sz, mt = int(idx.size[k]), float(idx.mtime[k])
                if meta_spec and (sz < 0 or mt < 0):
                    try:  # MFT enumeration carries no size/date: read them on demand for the sampled candidates only
                        st = os.stat(path)
                        sz, mt = int(st.st_size), float(st.st_mtime)
                        idx.size[k], idx.mtime[k] = sz, mt
                    except OSError:
                        continue
                if not _spec_matches(spec, {"path": path, "name": idx._orig_name(k), "size": sz, "mtime": mt}):
                    continue
            seen.add(pk)
            name = idx._orig_name(k)
            root_id = int(idx._root_of[k])
            out.append({"path": path, "name": name, "folder": os.path.dirname(path),
                        "size": max(0, int(idx.size[k])), "mtime": max(0.0, float(idx.mtime[k])),
                        "flags": int(idx.flags[k]), "root": idx._roots[root_id] if 0 <= root_id < len(idx._roots) else ""})
            if len(out) >= limit:
                break
        if saturated and len(out) < limit:
            return None  # the sample (first 4097 native rows) cannot prove completeness: let the Python path answer
        return out

    def _turbo_why(self, reason):
        """Count why a query went to the Python path (exposed in status()['turbo']['fallback_reasons'])."""
        d = getattr(self, "_turbo_reasons", None)
        if d is None:
            d = self._turbo_reasons = {}
        d[reason] = d.get(reason, 0) + 1
        return None

    def _turbo_search(self, idx, query, limit, prefixes, kinds=None):
        # Native is only a candidate accelerator. Never cap before product ACLs
        # or ranking: a saturated response requires the complete Python search.
        if not self._native_wanted(idx) or self._stop.is_set():
            return self._turbo_why('not_wanted')
        self._native_maintenance(idx)
        import re
        text = str(query or '').strip().lower()
        tokens = text.split()
        if (not tokens or any(c in text for c in '"\'[]^$|(){}?') or
                any(t.startswith(('-', '!')) or t in ('and', 'or', 'not') for t in tokens)):
            return None
        for token in tokens:
            if ':' in token and not token.startswith(('ext:', 'path:', 'folder:', 'size:', 'date:')):
                return None
            if '*' in token and not re.fullmatch(r'\*\.[\w]+', token):
                return None
        clean, spec = _parse_search_query(text)
        if not clean and spec is None:
            return None
        native_query = clean.lower() if spec is None else ''
        if spec is not None:
            # Operator semantics remain authoritative in localdocs_filters.
            if spec.get('sort') or spec.get('regex'):
                return self._turbo_why('unsupported_query')
            extensions = list(spec.get('ext') or ())
            wildcards = list(spec.get('wildcards') or ())
            if len(extensions) == 1 and extensions[0] and '.' not in extensions[0]:
                native_query = '*.' + extensions[0]
            elif len(wildcards) == 1:
                native_query = wildcards[0]
            path_tokens = (clean.lower().split() + list(spec.get('path') or ()) +
                           list(spec.get('folder') or ()))
            safe = [t for t in path_tokens if t and not any(c in t for c in
                    (os.sep, chr(92), '/', chr(0x0307))) and
                    not any(t in root.lower() for root in idx._roots)]
            if safe:
                native_query = chr(1) + max(safe, key=len) + chr(10) + native_query
            # Path predicates can match the configured root as well as ancestors;
            # the empty scan is a safe superset, including for multi-root indices.
        worker = None
        try:
            _t_lock0 = time.perf_counter()
            with self._turbo_guard:
                worker = self._turbo
                if not self._native_matches(worker, idx) or worker.has_expansion:
                    return self._turbo_why('not_ready' if not self._native_matches(worker, idx) else 'expansion')
                with idx._lock:
                    _lock_wait = (time.perf_counter() - _t_lock0) * 1000
                    touched = {k for k, v in idx._turbo_touched.items() if v > worker.revision}
                    touched.update(range(worker.n_exported, len(idx)))
                    if len(touched) > 20000:
                        return self._turbo_why('touched_too_many')
                    _t_n = time.perf_counter()
                    rows = worker.search(native_query, touched)
                    _t_f = time.perf_counter()
                    prof = getattr(self, '_turbo_prof', None)
                    if prof is None:
                        prof = self._turbo_prof = []
                    if os.environ.get('LOCALDOCS_NATIVE_FAST', '1') != '0':
                        try:
                            fast = self._native_fast_hits(idx, clean.lower().split(), rows, touched,
                                                          max(1, min(int(limit), 200)), prefixes, kinds,
                                                          spec=spec, saturated=len(rows) >= 4097)
                        except Exception as fast_exc:
                            # A Python-side bug in the fast path must NOT take the healthy native worker down.
                            self._turbo_fast_error = repr(fast_exc)[:300]
                            return self._turbo_why('fast_error')
                        if fast is not None:
                            self._turbo_native_queries += 1
                            self._turbo_timeouts = 0
                            if len(prof) < 60:
                                prof.append((query[:24], round((_t_f - _t_n) * 1000, 1), round((time.perf_counter() - _t_f) * 1000, 1),
                                             len(touched), len(rows), round(_lock_wait, 1)))
                            return fast
                    if len(rows) >= 4097:
                        return self._turbo_why('saturated')
                    if any(k < 0 or k >= worker.n_exported for k in rows):
                        raise ValueError('invalid native row')
                    candidates = sorted((set(rows) - touched) | touched)
                    result = idx._search_inner(query, limit, prefixes, kinds, candidate_rows=candidates)
                    self._turbo_native_queries += 1
                    return result
        except queue.Empty:
            # Transient stall (GIL/CPU contention): this query goes to Python; only repeated stalls retire the worker.
            self._turbo_timeouts = getattr(self, '_turbo_timeouts', 0) + 1
            if self._turbo_timeouts >= 3:
                self._turbo_timeouts = 0
                self._turbo_failed(RuntimeError('native worker unresponsive'), worker)
            return self._turbo_why('timeout')
        except Exception as exc:
            self._turbo_failed(exc, worker)
            return self._turbo_why('error')

    def search(self, query, limit=50, only_roots=None, only_prefixes=None, kinds=None):
        t0 = time.perf_counter()
        out = self._search_impl(query, limit, only_roots, only_prefixes, kinds)
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            if ms >= float(os.environ.get('LOCALDOCS_SLOW_MS', '150')):
                self._log_slow(query, limit, ms, len(out) if out is not None else -1)
        except Exception:
            pass
        return out

    def _log_slow(self, query, limit, ms, hits):
        """Slow-query log (logs/slow_queries.log, rotated at 2 MB): what was slow and what the process was doing."""
        folder = os.environ.get('LOCALDOCS_LOG_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'slow_queries.log')
        try:
            if os.path.getsize(path) > 2 * 1024 * 1024:
                os.replace(path, path + '.1')
        except OSError:
            pass
        prof = (getattr(self, '_turbo_prof', None) or [None])[-1]
        st = self._turbo_status() if hasattr(self, '_turbo_status') else {}
        busy = []
        for th in threading.enumerate():
            if th is threading.current_thread() or th.name.startswith(('Thread-', 'localdocs-cs-pipe')):
                continue
            busy.append(th.name)
        line = "%s %7.1f ms q=%r limit=%s hits=%s engine=%s native_q=%s fallbacks=%s reasons=%s last_native_prof=%s overlay=%s threads=%s" % (
            time.strftime('%Y-%m-%d %H:%M:%S'), ms, query, limit, hits, st.get('engine'), st.get('native_queries'),
            st.get('python_fallbacks'), st.get('fallback_reasons'), prof, st.get('overlay_rows'), ','.join(busy[:8]))
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + chr(10))

    def _search_impl(self, query, limit=50, only_roots=None, only_prefixes=None, kinds=None):
        try:
            prefixes = None
            if only_roots is not None:
                prefixes = []
                try:
                    for r in only_roots:
                        try:
                            prefixes.append(os.path.normpath(os.path.realpath(str(r))))
                        except Exception:
                            pass
                except Exception:
                    return []
                if not prefixes:
                    return []
            if only_prefixes is not None:
                supplied = list(only_prefixes)
                if not supplied:
                    return []
                if prefixes is None:
                    prefixes = supplied
                else:
                    prefixes = [p if _prefix_match(p, r) else r for p in supplied for r in prefixes
                                if _prefix_match(p, r) or _prefix_match(r, p)]
                    if not prefixes:
                        return []
            with self._lock:
                if not getattr(self, "_snapshot_validated", True):
                    return []
                idx = self._index
                if idx._journal_state:
                    allowed = list(self._roots)
                    if not allowed:
                        return []
                    if prefixes:
                        allowed = [p if idx._contains_path(p, r) else r
                                   for p in prefixes for r in allowed
                                   if idx._contains_path(p, r) or idx._contains_path(r, p)]
                        if not allowed:
                            return []
                    prefixes = allowed
            # Honour the same plain-query LRU before paying for a native round
            # trip. Cache validity is owned by NameIndex._invalidate().
            clean, spec = _parse_search_query(str(query or ''))
            if not clean.strip() and spec is None:
                return []
            key = None
            if spec is None and not kinds:
                try:
                    lim = max(1, min(int(limit), 200))
                except Exception:
                    lim = 50
                normalized = tuple(os.path.normcase(os.path.normpath(p)) for p in prefixes) if prefixes else None
                key = (str(query), lim, normalized, None)
            with idx._lock:
                worker = self._turbo
                cache_ready = True
                revision, layout = idx._turbo_revision, idx._turbo_layout
                hit = idx._lru.get(key) if key is not None and cache_ready else None
                if hit is not None:
                    idx._lru.move_to_end(key)
                    res = [dict(row) for row in hit]
                else:
                    res = None
            if res is None:
                try:
                    res = self._turbo_search(idx, query, limit, prefixes, kinds)
                except Exception as exc:
                    self._turbo_error = str(exc)
                    res = None
                if res is None:
                    self._turbo_fallbacks += 1
                    res = idx.search(query, limit=limit, only_prefixes=prefixes, kinds=kinds)
                elif key is not None:
                    with idx._lock:
                        if (revision, layout) == (idx._turbo_revision, idx._turbo_layout):
                            idx._lru[key] = [dict(row) for row in res]
                        while len(idx._lru) > _LRU_SIZE:
                            idx._lru.popitem(last=False)
            res = idx._append_fuzzy(res, query, limit, prefixes, kinds)
            out = _filter_results(truncated=getattr(res, 'truncated', False),
                                invalid_regex=getattr(res, 'invalid_regex', False))
            self.last_search_truncated = out.truncated
            for e in res:
                try:
                    out.append({"path": e["path"], "name": e["name"],
                                "folder": e["folder"], "size": e["size"],
                                "mtime": e["mtime"], "root": e.get("root", "")})
                    if e.get("fuzzy"):
                        out[-1].update(fuzzy=True, distance=e["distance"])
                except Exception:
                    pass
            return out
        except Exception:
            return []
