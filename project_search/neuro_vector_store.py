# Adapted from neuro_gm_project/app/embeddings.py. The original is unchanged.
# This copy uses a fixed local CPU embedder, independent of the game settings.
"""
Локальный семантический поиск: sentence-transformers (эмбеддинги считаются
на машине пользователя, без обращений в облако) + FAISS (уже был в проекте).

Заменяет lore_engine.py (который использовал GoogleGenerativeAIEmbeddings).
pymorphy3-лемматизация вынесена отдельно в app/text_normalize.py и
применяется к ключам лорбука (точное срабатывание по ключевым словам),
а не к самим эмбеддингам — модель эмбеддингов сама справляется с формами слов.

VectorStore используется в двух ролях в рамках одного мира:
  - индекс лорбука (полностью перестраивается при сохранении лорбука),
  - индекс памяти (LTM персонажей + эпизоды NPC), пополняется инкрементально.
"""

import json
import os
import operator
import sqlite3
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Exact Flat search is <1 ms at 18k chunks and ~23 ms at 880k; approximate HNSW
# only pays off above HNSW_MIN vectors (see bench/quality_report.md).
# Both thresholds are overridable via env (ints). 21.10: default stays 1.5M: on a real 990k-vector e5 index exact Flat answers in ~17 ms/query while HNSW (ef64..1024) reached only recall@10 0.48-0.65.
HNSW_MIN = int(os.environ.get("LOCALDOCS_HNSW_MIN", "1500000"))  # exact Flat below this
EF_SEARCH = int(os.environ.get("LOCALDOCS_HNSW_EF", "128"))  # legacy floor, see search()
# IVF-PQ measured recall@10 vs exact = 0.31 (nprobe=8, bench/test_ann_vs_exact.py), HNSW ef=16 0.68,
# ef=128 0.94: lossy indexes only where exact Flat no longer fits in RAM (~1.5 GB per 1M x 384 floats).
PQ_MIN = int(os.environ.get("LOCALDOCS_PQ_MIN", "5000000"))
INDEX_PREF = os.environ.get("LOCALDOCS_INDEX", "auto")  # auto|flat|hnsw|ivfpq
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
PQ_NLIST = int(os.environ.get("LOCALDOCS_PQ_NLIST", "100"))
PQ_M = int(os.environ.get("LOCALDOCS_PQ_M", "32"))
PQ_NPROBE = int(os.environ.get("LOCALDOCS_PQ_NPROBE", "8"))
# Night B2: opt-in binary-quantization candidate stage (default off).
BQ_MIN = int(os.environ.get("LOCALDOCS_BQ_MIN", "20000"))


def _bq_enabled():
    """Live env override: LOCALDOCS_BQ=1 enables the binary candidate stage."""
    return os.environ.get("LOCALDOCS_BQ", "0") == "1"


def _bq_min():
    """Live env override so tests can set LOCALDOCS_BQ_MIN after import."""
    try:
        return int(os.environ.get("LOCALDOCS_BQ_MIN", BQ_MIN))
    except (TypeError, ValueError):
        return BQ_MIN


def _bq_pack(vectors):
    """Sign-bit packing: dim 384 -> 48 bytes/vector via np.packbits on (v > 0).

    Accepts a single 1-D vector (treated as one row) or a 2-D (n, dim) batch.
    """
    arr = np.asarray(vectors)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    bits = np.asarray(arr > 0, dtype=np.uint8)
    packed = np.packbits(bits, axis=1)
    return np.ascontiguousarray(packed, dtype=np.uint8)


def _bq_cosine_rescore(query_vec, vectors, indices):
    """Exact cosine rescoring over candidate rows.

    Stored vectors are expected to be unit-norm (embed_texts normalizes), but
    both sides are normalised here defensively so the scores stay true cosines
    even for hand-built or legacy non-normalised vectors; zero rows score 0.0.
    Returns (score, index) sorted by descending score (stable on ties).
    """
    q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
    qn = float(np.linalg.norm(q))
    if qn > 0:
        q = q / qn
    # Vectorized: single fancy-index + matvec + single norm pass
    # (was per-candidate asarray/norm/dot in Python).
    try:
        cand = [int(i) for i in indices]
    except TypeError:
        return []
    if not cand:
        return []
    try:
        V = vectors if isinstance(vectors, np.ndarray) else np.asarray(vectors, dtype=np.float64)
        if V.dtype != np.float64:
            V = np.asarray(V, dtype=np.float64)
        sub = V[cand]
        if sub.ndim == 1:
            sub = sub[np.newaxis, :]
        dots = sub @ q
        norms = np.linalg.norm(sub, axis=1)
        scores = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
        order = np.argsort(-scores, kind="stable")
        return [(float(scores[o]), int(cand[o])) for o in order]
    except Exception:
        # Exotic container/shape: preserve exact legacy semantics.
        scored = []
        for i in cand:
            v = np.asarray(vectors[i], dtype=np.float64)
            vn = float(np.linalg.norm(v))
            s = float(np.dot(v, q) / vn) if vn > 0 else 0.0
            scored.append((s, i))
        scored.sort(key=lambda t: -t[0])
        return scored


def _hnsw_min():
    """Live env override so tests can set LOCALDOCS_HNSW_MIN after import."""
    try:
        return int(os.environ.get("LOCALDOCS_HNSW_MIN", HNSW_MIN))
    except (TypeError, ValueError):
        return HNSW_MIN


def _pq_min():
    """Live env override so tests can set LOCALDOCS_PQ_MIN after import."""
    try:
        return int(os.environ.get("LOCALDOCS_PQ_MIN", PQ_MIN))
    except (TypeError, ValueError):
        return PQ_MIN


def _make_hnsw(dim):
    """HNSW with M=32, efConstruction=200 (efSearch is set per query)."""
    import faiss
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    try:
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    except AttributeError:
        pass
    return index

LEGACY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # indexes without "model" in meta
_models = {}
_model_lock = threading.Lock()


def get_embedder(model_name=None):
    """Ленивая загрузка модели эмбеддингов (тяжёлая операция, один раз за процесс на модель)."""
    # Probe entitlement: get_embedder
    import probe_licensing as _pl
    _pl.require('semantic')
    if model_name is None:
        from project_search.backend import MODEL_NAME as model_name
    m = _models.get(model_name)
    if m is None:
        with _model_lock:
            m = _models.get(model_name)
            if m is None:
                from sentence_transformers import SentenceTransformer
                local = Path(__file__).resolve().parents[1] / 'models' / model_name.replace('/', '__')
                source = str(local) if (local / 'config.json').is_file() else model_name
                m = SentenceTransformer(source, device='cpu')
                m.max_seq_length = min(int(m.max_seq_length or 256), int(os.environ.get("LOCALDOCS_MAX_SEQ", "256")))
                _models[model_name] = m
    return m


def _prefix(model_name, kind):
    """E5 family needs 'query: ' / 'passage: ' prefixes; other models none."""
    if "e5" in (model_name or "").lower():
        return "query: " if kind == "query" else "passage: "
    return ""


def embed_texts(texts: list[str], kind: str = "passage", model_name=None) -> np.ndarray:
    # Probe entitlement: embed_texts
    import probe_licensing as _pl
    _pl.require('semantic')
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    if model_name is None:
        from project_search.backend import MODEL_NAME as model_name
    pre = _prefix(model_name, kind)
    if pre:
        texts = [pre + t for t in texts]
    try:
        from project_search import onnx_embed
        if onnx_embed.available(model_name, kind):
            return onnx_embed.embed(texts, model_name, kind)
    except Exception:
        pass
    model = get_embedder(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


_META_COLUMNS = ('key', 'path', 'start_line', 'end_line', 'mtime_ns', 'text')


def _metadata_path(path_prefix):
    """The published index uses meta.sqlite; staging uses pending_meta.sqlite."""
    prefix = Path(path_prefix)
    return prefix.with_name('meta.sqlite' if prefix.name == 'index'
                            else prefix.name + '_meta.sqlite')


class MetadataList(list):
    """Mutable metadata, with the same explicit materialization API as SQLite."""

    def to_list(self):
        return list(self)


class SQLiteMetadata(Sequence):
    """Read-only, rowid-addressed metadata with a bounded per-store LRU.

    No connection is kept open between calls (or across iterator yields), so
    Windows can replace the database during journaled publication. Callers must
    materialize with to_list() before mutating metadata.
    """

    def __init__(self, path, count, cache_size=128):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._count = operator.index(count)
        if self._count < 0:
            raise ValueError('Negative metadata count')
        self._cache_size = max(0, cache_size)
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self):
        return self._count

    def _connect(self):
        return sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)

    @staticmethod
    def _decode(row):
        # Explicit NULL values are also in extra; absent fields stay absent.
        result = json.loads(row[6])
        result.update((name, value) for name, value in zip(_META_COLUMNS, row[:6])
                      if value is not None)
        return result

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step == 1:
                return list(self._iter_range(start, max(start, stop)))
            return [self[i] for i in range(start, stop, step)]
        index = operator.index(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError('metadata index out of range')
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
            conn = self._connect()
            try:
                row = conn.execute(
                    'SELECT key,path,start_line,end_line,mtime_ns,text,extra '
                    'FROM chunks WHERE rowid=?', (index + 1,)).fetchone()
            finally:
                conn.close()
            if row is None:
                raise IndexError('Missing metadata rowid %d' % (index + 1))
            value = self._decode(row)
            if self._cache_size:
                self._cache[index] = value
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return value

    def __iter__(self):
        return self._iter_range(0, self._count)

    def _iter_range(self, first, stop):
        # Page the explicit full-scan path without holding a Windows file handle
        # while caller code is running. Search only uses __getitem__ and len().
        for start in range(first, stop, 512):
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT key,path,start_line,end_line,mtime_ns,text,extra '
                    'FROM chunks WHERE rowid>? AND rowid<=? ORDER BY rowid',
                    (start, min(start + 512, stop))).fetchall()
            finally:
                conn.close()
            if len(rows) != min(512, stop - start):
                raise ValueError('Metadata count does not match SQLite rows')
            for row in rows:
                yield self._decode(row)

    def to_list(self):
        return list(self)


def _write_metadata(path, metadata):
    """Stream rows to a complete standalone DB before replacing the staging file."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                     dir=str(path.parent))
    os.close(fd)
    try:
        conn = sqlite3.connect(temporary)
        try:
            conn.execute('CREATE TABLE chunks (key TEXT, path TEXT, start_line INTEGER, '
                         'end_line INTEGER, mtime_ns INTEGER, text TEXT, extra TEXT NOT NULL)')

            def rows():
                for i, m in enumerate(metadata, 1):
                    extra = {k: v for k, v in m.items()
                             if k not in _META_COLUMNS or v is None}
                    yield (i, *(m.get(k) for k in _META_COLUMNS),
                           json.dumps(extra, ensure_ascii=False))

            conn.executemany('INSERT INTO chunks(rowid,key,path,start_line,end_line,'
                             'mtime_ns,text,extra) VALUES(?,?,?,?,?,?,?,?)', rows())
            conn.commit()
        finally:
            conn.close()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class VectorStore:
    """
    Обёртка над faiss.IndexFlatIP (косинусная близость на нормализованных
    векторах) + ленивые метаданные SQLite и небольшой JSON-заголовок.
    """

    def __init__(self, dim: int | None = None, model: str | None = None):
        self._dim = dim
        self.model = model  # embedder id this index was built with (None = decide on first add)
        self._index = None
        self.index_type = "flat"
        self.metadata = []
        self.vectors: np.ndarray | None = None
        # Night B2: derived binary-quantization candidate index (never persisted).
        self._bq_index = None
        self._bq_key = None
        # Night B3: generation counter + lock for the derived BQ cache.
        # _vec_version bumps on every vectors/index mutation; the BQ key is
        # (version, n, dim) so id() reuse after GC can never alias stale bits.
        self._vec_version = 0
        self._bq_lock = threading.Lock()

    def _bq_invalidate(self):
        """Drop the derived binary index and bump the generation counter.

        Every mutation path (add, rebuild, load; extend here if a remove()
        method is ever added) must call this so a stale binary index can
        never match the new (version, n, dim) key.
        """
        with self._bq_lock:
            self._bq_index = None
            self._bq_key = None
            self._vec_version += 1

    def _ensure_bq(self):
        """Lazily build the IndexBinaryFlat of sign bits; None when unavailable.

        Thread-safe: double-checked locking on _bq_lock builds the index once
        even when many threads search a cold store concurrently.
        """
        import faiss
        if self.vectors is None or self._dim is None:
            return None
        try:
            n, dim = self.vectors.shape
        except (AttributeError, ValueError):
            return None
        if n != len(self.metadata) or n == 0:
            return None
        if dim % 8 != 0:
            return None
        key = (self._vec_version, n, dim)
        if self._bq_index is not None and self._bq_key == key:
            return self._bq_index
        with self._bq_lock:
            try:
                n2, dim2 = self.vectors.shape
            except (AttributeError, ValueError):
                return None
            if n2 != len(self.metadata) or n2 == 0 or dim2 % 8 != 0:
                return None
            key = (self._vec_version, n2, dim2)
            if self._bq_index is not None and self._bq_key == key:
                return self._bq_index
            vectors = self.vectors
            try:
                index = faiss.IndexBinaryFlat(dim2)
                codes = _bq_pack(vectors)
                index.add(codes)
            except Exception:
                return None
            if vectors is not self.vectors:
                # Mutated mid-build; discard the stale index so the next
                # search rebuilds from the current generation.
                return None
            self._bq_index = index
            self._bq_key = key
            return self._bq_index

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = (value if isinstance(value, (MetadataList, SQLiteMetadata))
                          else MetadataList(value))

    def _pick_type(self, n_expected):
        want = os.environ.get("LOCALDOCS_INDEX", INDEX_PREF)
        hnsw_min = _hnsw_min()
        pq_min = _pq_min()
        total = max(n_expected, len(self.metadata))
        if want == "auto":
            if total >= pq_min:
                want = "ivfpq"
            elif total >= hnsw_min:
                want = "hnsw"
            else:
                want = "flat"
        if want == "ivfpq" and total < 256:
            want = "hnsw" if total >= hnsw_min else "flat"
        return want

    def _ensure_index(self, dim: int, n_expected: int = 0):
        import faiss
        if self._index is None:
            self._dim = dim
            self.index_type = self._pick_type(n_expected)
            if self.index_type == "hnsw":
                self._index = _make_hnsw(dim)
            elif self.index_type == "ivfpq":
                quant = faiss.IndexFlatIP(dim)
                self._index = faiss.IndexIVFPQ(quant, dim, PQ_NLIST, PQ_M, 8)
            else:
                self._index = faiss.IndexFlatIP(dim)

    def _build_ivfpq(self, vectors):
        import faiss
        self._index.train(vectors)
        self._index.add(vectors)
        try:
            self._index.nprobe = PQ_NPROBE
        except AttributeError:
            pass

    @property
    def size(self) -> int:
        return len(self.metadata)

    def add(self, texts: list[str], metadatas: list[dict]) -> None:
        if not texts:
            return
        if self.model is None:
            from project_search.backend import MODEL_NAME
            self.model = MODEL_NAME
        vectors = embed_texts(texts, "passage", self.model)
        self._ensure_index(vectors.shape[1], len(self.metadata) + len(texts))
        if self.index_type == "ivfpq" and self._index.ntotal == 0:
            self._build_ivfpq(vectors)
        else:
            self._index.add(vectors)
        try:
            self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
            if isinstance(self.metadata, SQLiteMetadata):
                self.metadata = self.metadata.to_list()
            self.metadata.extend(metadatas)
            self._bq_invalidate()
        except Exception:
            self._index = None
            self.vectors = None
            self.metadata = []
            self._bq_invalidate()
            raise

    def rebuild(self, texts: list[str], metadatas: list[dict]) -> None:
        """Полная пересборка индекса с нуля (используется для лорбука при сохранении)."""
        import faiss
        if not texts:
            self.metadata = []
            self.vectors = None
            self._index = None
            self._dim = None
            self._bq_invalidate()
            return
        if self.model is None:
            from project_search.backend import MODEL_NAME
            self.model = MODEL_NAME
        vectors = embed_texts(texts, "passage", self.model)
        self._dim = vectors.shape[1]
        want = self._pick_type(len(texts))
        self.index_type = want
        if want == "hnsw":
            self._index = _make_hnsw(self._dim)
            self._index.add(vectors)
        elif want == "ivfpq":
            quant = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIVFPQ(quant, self._dim, PQ_NLIST, PQ_M, 8)
            self._build_ivfpq(vectors)
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(vectors)
        self.vectors = vectors
        self.metadata = (metadatas.to_list() if isinstance(metadatas, SQLiteMetadata)
                         else list(metadatas))
        self._bq_invalidate()

    def search(self, query: str, k: int = 5, *, return_indices=False) -> list[tuple]:
        """Return (metadata, score), optionally with the FAISS position as a third item."""
        if self._index is None or self.size == 0 or not query.strip():
            return []
        query_vec = embed_texts([query], "query", self.model or LEGACY_MODEL)
        k = min(k, self.size)
        # Night B2: opt-in binary-quantization candidate stage (default off).
        if (_bq_enabled() and self.size >= _bq_min() and self.vectors is not None
                and len(self.vectors) == self.size):
            try:
                bq = self._ensure_bq()
            except Exception:
                bq = None
            if bq is not None:
                try:
                    n_fetch = min(self.size, max(10 * k, 200))
                    q_bin = _bq_pack(query_vec)
                    _, b_indices = bq.search(q_bin, n_fetch)
                    cand = [int(i) for i in b_indices[0] if int(i) >= 0]
                except Exception:
                    cand = []
                if cand:
                    # Cosine rescoring with normalisation (true cosine even for
                    # non-normalised stored vectors); stable sort keeps the
                    # Hamming candidate order on ties.
                    exact = _bq_cosine_rescore(query_vec[0], self.vectors, cand)
                    return [(self.metadata[i], sim, i) if return_indices else (self.metadata[i], sim)
                            for sim, i in exact[:k]]
        import faiss as _faiss
        is_hnsw = isinstance(self._index, _faiss.IndexHNSWFlat)
        if is_hnsw:
            # Over-retrieve candidates, then exactly rescore with the stored
            # normalized vectors (IndexHNSWFlat ranks by L2, equivalent to
            # cosine here, but the candidate set is approximate).
            n_fetch = k
            if (self.vectors is not None and len(self.vectors) == self.size
                    and self.size > k):
                n_fetch = min(self.size, max(k * 10, 100))
            try:
                self._index.hnsw.efSearch = max(64, 4 * n_fetch)
            except AttributeError:
                pass
            scores, indices = self._index.search(query_vec, n_fetch)
            cand = [int(i) for i in indices[0] if int(i) >= 0]
            if n_fetch > k and cand:
                q = query_vec[0].astype("float64")
                exact = [(float(np.dot(self.vectors[i].astype("float64"), q)), i)
                         for i in cand]
                exact.sort(key=lambda t: -t[0])
                return [(self.metadata[i], sim, i) if return_indices else (self.metadata[i], sim)
                        for sim, i in exact[:k]]
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                sim = float(1.0 - float(score) / 2.0) if is_hnsw else float(score)
                hit = (self.metadata[int(idx)], sim)
                results.append((*hit, int(idx)) if return_indices else hit)
            return results[:k]
        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sim = float(1.0 - float(score) / 2.0) if is_hnsw else float(score)
            hit = (self.metadata[idx], sim)
            results.append((*hit, int(idx)) if return_indices else hit)
        return results

    def save(self, path_prefix: str) -> None:
        import faiss
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        meta_path = f"{path_prefix}_meta.json"
        _write_metadata(_metadata_path(path_prefix), self.metadata)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"dim": self._dim, "index_type": self.index_type,
                       "model": self.model or LEGACY_MODEL, "count": len(self.metadata),
                       "format": 2}, f, ensure_ascii=False, indent=2)
        if self.vectors is not None:
            np.save(f"{path_prefix}.npy", self.vectors)
        elif os.path.exists(f"{path_prefix}.npy"):
            os.remove(f"{path_prefix}.npy")
        if self._index is not None:
            faiss.write_index(self._index, f"{path_prefix}.faiss")
        elif os.path.exists(f"{path_prefix}.faiss"):
            os.remove(f"{path_prefix}.faiss")

    @classmethod
    def load(cls, path_prefix: str) -> "VectorStore":
        import faiss
        store = cls()
        meta_path = f"{path_prefix}_meta.json"
        index_path = f"{path_prefix}.faiss"
        vec_path = f"{path_prefix}.npy"
        if not os.path.exists(meta_path):
            return store
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get('format') == 2:
            store.metadata = SQLiteMetadata(_metadata_path(path_prefix), data['count'])
        else:
            store.metadata = data.get("metadata", [])
        store._dim = data.get("dim")
        store.index_type = data.get("index_type", "flat")
        store.model = data.get("model", LEGACY_MODEL)
        if os.path.exists(vec_path):
            try:
                store.vectors = np.load(vec_path)
            except (OSError, ValueError):
                store.vectors = None
        if (store.index_type == "ivfpq" and INDEX_PREF == "auto" and store.vectors is not None
                and len(store.vectors) == len(store.metadata) and len(store.vectors) < _hnsw_min()
                and store._dim):
            store.index_type = "flat"  # legacy lossy IVF-PQ index: serve exactly from the saved vectors
            store._index = faiss.IndexFlatIP(store._dim)
            if len(store.vectors):
                store._index.add(store.vectors)
        elif store.index_type == "hnsw" and os.path.exists(index_path):
            # Keep the exact persisted HNSW graph so save/load round-trips
            # return identical rankings (rebuilding would re-randomize links).
            try:
                store._index = faiss.read_index(index_path)
            except Exception:
                store._index = None
            if store._index is None and store.vectors is not None and store._dim:
                if len(store.metadata) != len(store.vectors):
                    store.vectors = store.vectors[:len(store.metadata)]
                store._index = _make_hnsw(store._dim)
                if len(store.vectors):
                    store._index.add(store.vectors)
        elif store.index_type == "ivfpq" and os.path.exists(index_path):
            store._index = faiss.read_index(index_path)
            try:
                store._index.nprobe = PQ_NPROBE
            except AttributeError:
                pass
        elif store.vectors is not None and store._dim:
            if len(store.metadata) != len(store.vectors):
                store.vectors = store.vectors[:len(store.metadata)]
            if store.index_type == "hnsw":
                store._index = _make_hnsw(store._dim)
            else:
                store._index = faiss.IndexFlatIP(store._dim)
            if len(store.vectors):
                store._index.add(store.vectors)
        elif os.path.exists(index_path):
            store._index = faiss.read_index(index_path)
        # Night B3: loaded vectors are a new generation for the derived BQ cache.
        store._vec_version += 1
        return store
