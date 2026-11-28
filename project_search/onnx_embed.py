"""ONNX-эмбеддер как опция (включается через LOCALDOCS_ONNX).

Модель берётся из каталога models_onnx/<последняя часть model_name>/
(рядом с корнем проекта, на уровень выше папки project_search):
  - model_int8.onnx (или model.onnx, если LOCALDOCS_ONNX=fp32),
  - токенизатор из того же каталога (transformers.AutoTokenizer).

Префиксы query:/passage: НЕ добавляются здесь — их уже добавил
embed_texts ДО вызова (см. project_search/neuro_vector_store.py).

Night G1: опциональный GPU (DirectML) для passage/query через
LOCALDOCS_GPU=1|dml|auto. По умолчанию GPU выключен (поведение CPU без
изменений). Все GPU-инференсы процесса сериализованы одним модульным
_gpu_lock; при любой ошибке GPU — прозрачный фолбэк на CPU до конца процесса.
"""

import os
import sys
import threading
from pathlib import Path

import numpy as np

_DEFAULT_BATCH = 64
_DEFAULT_GPU_BATCH = 32
_ALLOWED = ("on", "1", "int8", "fp32")
_PASSAGE_ALLOWED = ("fp32", "int8")
_GPU_REQUEST_VALUES = ("1", "dml", "auto", "on")
_GPU_KINDS = ("passage", "query")

_sessions = {}
_sessions_lock = threading.Lock()

# --- Night G1: состояние опционального GPU (DirectML) ---
_gpu_sessions = {}
_gpu_lock = threading.Lock()  # ОДИН замок на все GPU-инференсы процесса
_gpu_state_lock = threading.Lock()
_gpu_failed = None  # текст последней ошибки GPU или None
_texts_on_gpu = 0
_texts_on_cpu = 0


def _passage_mode():
    """ONNX for indexing passages. 21.10 measured on 240 real chunks: fp32 ONNX vectors are identical to torch
    (cos 1.00000, top-10 neighbour agreement 1.000) and ~45% faster, so fp32 is the DEFAULT; int8 is opt-in only
    (cos 0.98, neighbour agreement 0.82). LOCALDOCS_ONNX_PASSAGE=off|torch forces torch, =int8 opts into int8."""
    value = os.environ.get("LOCALDOCS_ONNX_PASSAGE", "").strip().lower()
    if value in ("off", "torch", "0"):
        return ""
    if not value:
        return "fp32"
    return value if value in _PASSAGE_ALLOWED else "fp32"


def _batch_size():
    try:
        return max(1, int(os.environ.get("LOCALDOCS_ONNX_BATCH", str(_DEFAULT_BATCH))))
    except (TypeError, ValueError):
        return _DEFAULT_BATCH


def _mode(kind=None):
    """Explicit LOCALDOCS_ONNX wins ('off' disables). By default queries use fp32 ONNX
    (numerically identical to torch, ~1 s cold start instead of ~10 s); indexing stays on torch
    unless LOCALDOCS_ONNX_PASSAGE=fp32|int8 opts passages into ONNX."""
    value = os.environ.get("LOCALDOCS_ONNX", "").strip().lower()
    if value == "off" or value == "0":
        return ""
    if value:
        return value
    if kind == "passage":
        return _passage_mode()
    return "fp32" if kind == "query" else ""


def _max_len():
    try:
        return max(1, int(os.environ.get("LOCALDOCS_MAX_SEQ", "256")))
    except (TypeError, ValueError):
        return 256


def _model_dir(model_name):
    last = (model_name or "").split("/")[-1].strip()
    # каталог models_onnx рядом с корнем проекта (на уровень выше project_search)
    root = Path(__file__).resolve().parent.parent
    return root / "models_onnx" / last, last


def _onnx_path(model_name, kind=None):
    model_dir, _ = _model_dir(model_name)
    fname = "model.onnx" if _mode(kind) == "fp32" else "model_int8.onnx"
    return model_dir / fname


def available(model_name, kind=None):
    """True только если ONNX включён (явно или по умолчанию для запросов) и файлы модели существуют."""
    if _mode(kind) not in _ALLOWED:
        return False
    if not model_name or not model_name.split("/")[-1].strip():
        return False
    try:
        return _onnx_path(model_name, kind).is_file()
    except (OSError, ValueError):
        return False


# --- Night G1: GPU (DirectML) ---

def _gpu_requested():
    """LOCALDOCS_GPU unset/0/off -> CPU (DEFAULT); 1/dml/auto(/on) -> пробовать DirectML."""
    return os.environ.get("LOCALDOCS_GPU", "").strip().lower() in _GPU_REQUEST_VALUES


def _gpu_device_id():
    try:
        return max(0, int(os.environ.get("LOCALDOCS_GPU_DEVICE", "0") or 0))
    except (TypeError, ValueError):
        return 0


def _gpu_batch_size():
    try:
        return max(1, int(os.environ.get("LOCALDOCS_GPU_BATCH", str(_DEFAULT_GPU_BATCH))))
    except (TypeError, ValueError):
        return _DEFAULT_GPU_BATCH


def _ort_providers():
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def _mark_gpu_failed(err):
    """Логирует один раз, запоминает текст ошибки до конца процесса."""
    global _gpu_failed
    text = ("%s" % (err,)).strip() or "unknown GPU error"
    with _gpu_state_lock:
        first = _gpu_failed is None
        _gpu_failed = text
    if first:
        try:
            print("[onnx_embed] GPU failed, falling back to CPU: %s" % text, file=sys.stderr)
        except Exception:
            pass


def _add_counts(gpu=0, cpu=0):
    global _texts_on_gpu, _texts_on_cpu
    with _gpu_state_lock:
        _texts_on_gpu += int(gpu or 0)
        _texts_on_cpu += int(cpu or 0)


def gpu_status():
    """Состояние опционального GPU для сервера/UI."""
    requested = _gpu_requested()
    providers = _ort_providers()
    available_flag = "DmlExecutionProvider" in providers
    with _gpu_state_lock:
        failed = _gpu_failed
        on_gpu = _texts_on_gpu
        on_cpu = _texts_on_cpu
    active = bool(requested and available_flag and failed is None)
    return {
        "requested": requested,
        "available": available_flag,
        "active": active,
        "providers": providers,
        "device_id": _gpu_device_id(),
        "failed": failed,
        "texts_on_gpu": on_gpu,
        "texts_on_cpu": on_cpu,
    }


def _get_gpu_session_and_tokenizer(model_name, kind=None):
    """DML-сессия (кэш: не более одной GPU-сессии на модель+файл). Бросает исключение -> фолбэк на CPU."""
    import onnxruntime as ort

    if _gpu_failed is not None:
        raise RuntimeError("GPU disabled: %s" % _gpu_failed)
    try:
        providers_now = list(ort.get_available_providers())
    except Exception as e:
        raise RuntimeError("cannot list onnxruntime providers: %s" % e)
    if "DmlExecutionProvider" not in providers_now:
        raise RuntimeError("DmlExecutionProvider not available (providers=%s)" % (providers_now,))
    onnx_path = _onnx_path(model_name, kind)
    key = (model_name, str(onnx_path))
    # Создание под замком: никогда не больше одной GPU-сессии на модель.
    with _sessions_lock:
        cached = _gpu_sessions.get(key)
        if cached is not None and cached[2] == str(onnx_path):
            return cached[0], cached[1]
        model_dir, _ = _model_dir(model_name)
        tokenizer = _load_tokenizer(model_dir)
        options = ort.SessionOptions()
        options.enable_mem_pattern = False  # требование DirectML
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # требование DirectML
        device_id = _gpu_device_id()
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=[("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
        )
        _gpu_sessions[key] = (session, tokenizer, str(onnx_path))
        return session, tokenizer


class _FastTokenizer:
    """tokenizers-only wrapper: avoids importing transformers/torch (~8 s) on a cold start."""

    def __init__(self, path, max_length):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(path))
        self._max = max_length
        pad_id = self._tok.token_to_id("<pad>")
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding(pad_id=pad_id if pad_id is not None else 0, pad_token="<pad>")

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors="np"):
        encodings = self._tok.encode_batch(list(texts))
        return {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64)}


def _load_tokenizer(model_dir):
    fast = Path(model_dir) / "tokenizer.json"
    if fast.is_file():
        try:
            return _FastTokenizer(fast, _max_len())
        except Exception:
            pass
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir))


def _get_session_and_tokenizer(model_name, kind=None):
    import onnxruntime as ort

    onnx_path = _onnx_path(model_name, kind)
    key = (model_name, str(onnx_path))
    with _sessions_lock:
        cached = _sessions.get(key)
    model_dir, _ = _model_dir(model_name)
    if cached is not None and cached[2] == str(onnx_path):
        return cached[0], cached[1]
    tokenizer = _load_tokenizer(model_dir)
    options = ort.SessionOptions()
    try:
        threads = max(1, int(os.environ.get("LOCALDOCS_CPU_CORES", "") or 0)) or (os.cpu_count() or 1)
    except ValueError:
        threads = os.cpu_count() or 1
    options.intra_op_num_threads = threads  # follow the CPU limit; oversubscription made ONNX 4x slower
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    with _sessions_lock:
        _sessions[key] = (session, tokenizer, str(onnx_path))
    return session, tokenizer


def _build_feed(session, encoded):
    input_names = {i.name for i in session.get_inputs()}
    feed = {}
    if "input_ids" in input_names:
        feed["input_ids"] = np.asarray(encoded["input_ids"], dtype=np.int64)
    if "attention_mask" in input_names:
        feed["attention_mask"] = np.asarray(encoded["attention_mask"], dtype=np.int64)
    if "token_type_ids" in input_names and "token_type_ids" in encoded:
        feed["token_type_ids"] = np.asarray(
            encoded["token_type_ids"], dtype=np.int64
        )
    return feed


def _pool(last_hidden, mask):
    last_hidden = np.asarray(last_hidden, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    # пулинг: среднее по токенам с attention_mask, затем L2-нормализация
    summed = (last_hidden * mask[:, :, None]).sum(axis=1)
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1e-9)
    mean_pooled = summed / counts
    norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (mean_pooled / norms).astype(np.float32)


def _embed_cpu(texts, model_name, kind):
    """CPU-путь без изменений (батчинг LOCALDOCS_ONNX_BATCH)."""
    max_length = _max_len()
    session, tokenizer = _get_session_and_tokenizer(model_name, kind)
    batch_size = _batch_size()

    # глобальная сортировка по длине для меньшего паддинга, затем возврат в исходный порядок
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    ordered_texts = [texts[i] for i in order]
    out_ordered = [None] * len(texts)
    for start in range(0, len(texts), batch_size):
        chunk = ordered_texts[start:start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feed = _build_feed(session, encoded)
        (last_hidden,) = session.run(None, feed)
        normalized = _pool(last_hidden, encoded["attention_mask"])
        for pos, vec in enumerate(normalized):
            out_ordered[start + pos] = vec
    out = [None] * len(texts)
    for rank, idx in enumerate(order):
        out[idx] = out_ordered[rank]
    res = np.stack(out).astype(np.float32)
    _add_counts(cpu=len(texts))
    return res


def embed(texts, model_name, kind):
    """Возвращает numpy float32 массив нормализованных эмбеддингов."""
    texts = list(texts or [])
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    if kind not in _GPU_KINDS or not _gpu_requested() or _gpu_failed is not None:
        return _embed_cpu(texts, model_name, kind)
    try:
        gpu_session, gpu_tokenizer = _get_gpu_session_and_tokenizer(model_name, kind)
    except Exception as e:
        _mark_gpu_failed(e)
        return _embed_cpu(texts, model_name, kind)
    max_length = _max_len()
    batch_size = _gpu_batch_size()

    # глобальная сортировка по длине для меньшего паддинга, затем возврат в исходный порядок
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    ordered_texts = [texts[i] for i in order]
    out_ordered = [None] * len(texts)
    cpu_session = None
    cpu_only_rest = False
    for start in range(0, len(texts), batch_size):
        chunk = ordered_texts[start:start + batch_size]
        if cpu_only_rest or _gpu_failed is not None:
            if cpu_session is None:
                cpu_session, _ = _get_session_and_tokenizer(model_name, kind)
            encoded = gpu_tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            feed = _build_feed(cpu_session, encoded)
            (last_hidden,) = cpu_session.run(None, feed)
            _add_counts(cpu=len(chunk))
            normalized = _pool(last_hidden, encoded["attention_mask"])
            for pos, vec in enumerate(normalized):
                out_ordered[start + pos] = vec
            continue
        encoded = gpu_tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feed = _build_feed(gpu_session, encoded)
        try:
            with _gpu_lock:  # ВСЕ GPU-инференсы процесса — строго по одному
                (last_hidden,) = gpu_session.run(None, feed)
        except Exception as e:
            _mark_gpu_failed(e)
            if cpu_session is None:
                cpu_session, _ = _get_session_and_tokenizer(model_name, kind)
            feed_cpu = _build_feed(cpu_session, encoded)
            (last_hidden,) = cpu_session.run(None, feed_cpu)
            _add_counts(cpu=len(chunk))
            normalized = _pool(last_hidden, encoded["attention_mask"])
            for pos, vec in enumerate(normalized):
                out_ordered[start + pos] = vec
            cpu_only_rest = True
            continue
        _add_counts(gpu=len(chunk))
        normalized = _pool(last_hidden, encoded["attention_mask"])
        for pos, vec in enumerate(normalized):
            out_ordered[start + pos] = vec
    out = [None] * len(texts)
    for rank, idx in enumerate(order):
        out[idx] = out_ordered[rank]
    return np.stack(out).astype(np.float32)
