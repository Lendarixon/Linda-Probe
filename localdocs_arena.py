"""Compact column/name data classes used by localdocs_fastindex (split out to keep that file small)."""
import sys
import types

import numpy as np


class _NameArena(np.lib.mixins.NDArrayOperatorsMixin):
    """Immutable dictionary-coded UTF-16 units with an optional spelling overlay.

    Consumers still see uint16 units and unchanged offsets (including lone
    surrogates). Up to 256 distinct units use one byte per unit; other corpora
    fall back to ordinary UTF-16 arrays. Original spelling shares the lowercase
    arena and stores only differing units. This is lossless alphabet coding,
    not a locale-dependent encoding or a case reconstruction heuristic.
    """
    def __init__(self, codes, alphabet, positions=None, spelling=None):
        self.codes, self.alphabet = codes, alphabet
        self.positions = positions if positions is not None else np.empty(0, dtype=np.uint32)
        self.spelling = spelling if spelling is not None else np.empty(0, dtype=np.uint16)
        for value in self.storage:
            value.flags.writeable = False
        self.shape = codes.shape
        self.dtype = np.dtype(np.uint16)
        self.ndim = 1
        self.size = len(codes)
        self.flags = types.SimpleNamespace(writeable=False)

    @property
    def storage(self):
        return (self.codes, self.alphabet, self.positions, self.spelling)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self.storage)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        out = self.alphabet[self.codes[key]]
        if not len(self.positions):
            return out
        if isinstance(key, slice) and (key.step is None or key.step == 1):
            start, stop, _ = key.indices(len(self))
            # Needles must share the haystack dtype: mixed uint32/int64 makes NumPy re-cast the whole positions array per call.
            a, b = np.searchsorted(self.positions, np.array([start, stop], dtype=self.positions.dtype))
            if a != b:
                out[self.positions[a:b].astype(np.int64) - start] = self.spelling[a:b]
            return out
        if np.isscalar(key):
            pos = int(key) % len(self)
            at = int(np.searchsorted(self.positions, self.positions.dtype.type(pos)))
            return self.spelling[at] if at < len(self.positions) and self.positions[at] == pos else out
        # Only uncommon strided/fancy reads need explicit position selection.
        positions = (np.arange(*key.indices(len(self))) if isinstance(key, slice)
                     else np.asarray(key))
        if positions.dtype.kind == 'b':
            positions = np.flatnonzero(positions)
        positions = np.where(positions < 0, positions + len(self), positions)
        # Negative/out-of-range needles cannot match; cast to the haystack dtype after clipping.
        clipped = np.clip(positions, 0, np.iinfo(self.positions.dtype).max).astype(self.positions.dtype)
        at = np.searchsorted(self.positions, clipped)
        safe = np.minimum(at, len(self.positions) - 1)
        mask = (at < len(self.positions)) & (self.positions[safe] == clipped)
        return np.where(mask, self.spelling[safe], out)

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out.astype(dtype, copy=False) if dtype is not None else out

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return getattr(ufunc, method)(*[np.asarray(v) if isinstance(v, _NameArena) else v
                                       for v in inputs], **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func is np.shares_memory and all(isinstance(v, _NameArena) for v in args[:2]):
            # Different spelling views are not interchangeable name arrays.
            return args[0] is args[1]
        def unpack(value):
            if isinstance(value, _NameArena):
                return np.asarray(value)
            if isinstance(value, (list, tuple)):
                return type(value)(unpack(v) for v in value)
            return value
        return func(*unpack(args), **kwargs)

    def astype(self, dtype, **kwargs):
        return np.asarray(self).astype(dtype, **kwargs)

    def copy(self):
        return np.asarray(self)

    def tobytes(self):
        return np.asarray(self).tobytes()


class _DefaultColumn:
    """Sparse scalar metadata; array reads allocate only the requested selection.

    Full materialization is explicit for metadata operators. Copies own their
    dictionary, so journal publication never aliases writable cached values.
    """
    def __init__(self, n, dtype, default, values=None):
        self.shape = (int(n),)
        self.dtype = np.dtype(dtype)
        self.default = self.dtype.type(default)
        self.values = dict(values or {})

    def __len__(self):
        return self.shape[0]

    @property
    def nbytes(self):
        return (sys.getsizeof(self.values) + sum(sys.getsizeof(k) +
                sys.getsizeof(v) for k, v in self.values.items()))

    def __array__(self, dtype=None, copy=None):
        out = np.full(self.shape, self.default, dtype=dtype or self.dtype)
        if self.values:
            keys = np.fromiter(self.values, dtype=np.int64)
            out[keys] = list(self.values.values())
        return out

    def __getitem__(self, key):
        if np.isscalar(key):
            k = int(key)
            if k < 0:
                k += len(self)
            if not 0 <= k < len(self):
                raise IndexError(k)
            return self.values.get(k, self.default)
        if isinstance(key, slice):
            return self._gather(np.arange(*key.indices(len(self)), dtype=np.int64))
        rows = np.asarray(key)
        if rows.dtype.kind == 'b':
            rows = np.flatnonzero(rows)
        return self._gather(rows.astype(np.int64, copy=False))

    def _sorted_values(self):
        """Sorted key/value arrays of the sparse dictionary (vectorised gathers); rebuilt after any write."""
        cache = getattr(self, '_sorted_cache', None)
        if cache is None or cache[0] != len(self.values):
            keys = np.fromiter(self.values, dtype=np.int64, count=len(self.values))
            vals = np.fromiter(self.values.values(), dtype=self.dtype, count=len(self.values))
            order = np.argsort(keys, kind='stable')
            cache = (len(self.values), keys[order], vals[order])
            self._sorted_cache = cache
        return cache[1], cache[2]

    def _gather(self, rows):
        shape = rows.shape
        flat = rows.reshape(-1).copy()
        flat[flat < 0] += len(self)
        if flat.size and (flat.min() < 0 or flat.max() >= len(self)):
            raise IndexError('index out of range')
        out = np.full(flat.shape, self.default, dtype=self.dtype)
        if self.values and flat.size:
            keys, vals = self._sorted_values()
            pos = np.searchsorted(keys, flat)
            pos[pos >= len(keys)] = 0
            hit = keys[pos] == flat
            out[hit] = vals[pos[hit]]
        return out.reshape(shape)

    def __setitem__(self, key, value):
        if not np.isscalar(key):
            rows = np.arange(len(self))[key]
            for k, v in zip(np.atleast_1d(rows), np.broadcast_to(value, np.shape(rows))):
                self[int(k)] = v
            return
        k = int(key)
        if k < 0:
            k += len(self)
        if not 0 <= k < len(self):
            raise IndexError(k)
        value = self.dtype.type(value)
        if value == self.default:
            self.values.pop(k, None)
        else:
            self.values[k] = value
        self._sorted_cache = None

    def copy(self):
        return _DefaultColumn(len(self), self.dtype, self.default, self.values)

    def astype(self, dtype, **kwargs):
        return np.asarray(self).astype(dtype, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func is np.concatenate and kwargs.get('axis', 0) == 0:
            parts = args[0]
            parts = [(_DefaultColumn(len(p), self.dtype, self.default,
                      {i: v for i, v in enumerate(p) if v != self.default})
                      if isinstance(p, np.ndarray) and p.ndim == 1 and len(p) <= 1024 else p)
                     for p in parts]
            if all(isinstance(p, _DefaultColumn) and p.dtype == self.dtype and
                   p.default == self.default for p in parts):
                values, offset = {}, 0
                for part in parts:
                    values.update((k + offset, v) for k, v in part.values.items())
                    offset += len(part)
                return _DefaultColumn(offset, self.dtype, self.default, values)
        return func(*[np.asarray(a) if isinstance(a, _DefaultColumn) else
                      [np.asarray(p) if isinstance(p, _DefaultColumn) else p for p in a]
                      if isinstance(a, (list, tuple)) else a for a in args], **kwargs)


class _MappedBytes:
    """bytes.find over an NPY member without copying its entire name arena."""
    def __init__(self, array):
        self.array = array
        self.mapping = array._mmap
        import mmap
        self.offset = int(array.offset) % mmap.ALLOCATIONGRANULARITY
        self.length = int(array.nbytes)

    def find(self, needle, start=0, end=None):
        end = self.length if end is None else min(int(end), self.length)
        at = self.mapping.find(needle, self.offset + int(start), self.offset + end)
        return at - self.offset if at >= 0 else -1
