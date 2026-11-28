"""LocalDocs O5 settings store: content search + OCR switches.

Thread-safe, persisted as JSON in the LocalDocs data dir (same directory
selection rules as localdocs_server.py: LOCALDOCS_CONFIG dir or module ROOT).
"""

import json
import os
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "content_search": True,
    "ocr_images": True,
    "ocr_pdf_scans": True,
    "ocr_gpu": False,
    "ocr_engine": "ready",
    "ocr_when": "search",
    "ocr_lang": "rus+eng+pol",
    "ocr_max_mb": 25,
    "names_only_default": False,
    "fuzzy_names": True,
    "name_memory": "fast",
    "detailed_metadata": True,
    "image_device": "cpu",
    "watch_new_files": True,
    "ocr_new_files": True,
    "ocr_rotation": 0,
    "ocr_spreads": False,
}

_BOOL_KEYS = {"watch_new_files", "ocr_new_files", "content_search", "ocr_images", "ocr_pdf_scans", "ocr_gpu", "names_only_default", "fuzzy_names", "detailed_metadata", "ocr_spreads"}

_OCR_MAX_MIN = 1
_OCR_MAX_MAX = 500

_lock = threading.RLock()

# Set by the on-demand "OCR everything" job; OCR during indexing is otherwise allowed only when ocr_when == "index".
FORCE_OCR = threading.Event()


def ocr_during_index():
    """True when indexing may run OCR now (setting "index" or an explicit on-demand job)."""
    if FORCE_OCR.is_set():
        return True
    override = os.environ.get("LOCALDOCS_OCR_WHEN")  # tests / CI of the OCR pipeline
    if override:
        return override == "index"
    try:
        return get().get("ocr_when", "search") == "index"
    except Exception:
        return False
_cache = None


def _settings_file():
    override = os.environ.get("LOCALDOCS_SETTINGS")
    if override:
        return override
    cfg = os.environ.get("LOCALDOCS_CONFIG")
    if cfg:
        try:
            base = os.path.dirname(os.path.abspath(cfg))
            if base:
                return os.path.join(base, "localdocs_settings.json")
        except (OSError, ValueError):
            pass
    data_dir = os.environ.get("LOCALDOCS_DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "localdocs_settings.json")
    return os.path.join(ROOT, "localdocs_settings.json")


def _coerce_bool(key, value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError("Invalid value for %s: must be bool" % key)


def _coerce_lang(value):
    if not isinstance(value, str):
        raise ValueError("Invalid ocr_lang: must be a string")
    text = value.strip()
    if not text or len(text) > 64:
        raise ValueError("Invalid ocr_lang: must be 1..64 chars")
    import re as _re
    if not _re.match(r"^[A-Za-z+_\- ]+$", text):
        raise ValueError("Invalid ocr_lang: bad characters")
    # normalise: collapse whitespace, keep case as given
    text = _re.sub(r"\s+", "", text)
    if not text:
        raise ValueError("Invalid ocr_lang: must be 1..64 chars")
    return text


def _coerce_max_mb(value):
    if isinstance(value, bool):
        raise ValueError("Invalid ocr_max_mb: must be a number")
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Invalid ocr_max_mb: must be a number")
        value = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Invalid ocr_max_mb: must be a number")
        try:
            value = int(float(text))
        except ValueError:
            raise ValueError("Invalid ocr_max_mb: must be a number")
    elif not isinstance(value, int):
        raise ValueError("Invalid ocr_max_mb: must be a number")
    if value < _OCR_MAX_MIN:
        return _OCR_MAX_MIN
    if value > _OCR_MAX_MAX:
        return _OCR_MAX_MAX
    return int(value)


def _validate_patch(patch):
    if not isinstance(patch, dict):
        raise ValueError("settings patch must be an object")
    cleaned = {}
    for key, value in patch.items():
        if key not in DEFAULTS:
            raise ValueError("Unknown settings key: %s" % str(key))
        if key in _BOOL_KEYS:
            cleaned[key] = _coerce_bool(key, value)
        elif key == "name_memory":
            if not isinstance(value, str) or value not in ("fast", "compact"):
                raise ValueError("Invalid name_memory: must be fast or compact")
            cleaned[key] = value
        elif key == "ocr_when":
            if not isinstance(value, str) or value not in ("search", "manual", "index"):
                raise ValueError("Invalid ocr_when: must be search, manual or index")
            cleaned[key] = value
        elif key == "ocr_engine":
            if not isinstance(value, str) or value not in ("ready", "own"):
                raise ValueError("Invalid ocr_engine: must be ready or own")
            cleaned[key] = value
        elif key == "image_device":
            if not isinstance(value, str) or value not in ("cpu", "gpu"):
                raise ValueError("Invalid image_device: must be cpu or gpu")
            cleaned[key] = value
        elif key == "ocr_lang":
            cleaned[key] = _coerce_lang(value)
        elif key == "ocr_max_mb":
            cleaned[key] = _coerce_max_mb(value)
        elif key == "ocr_rotation":
            try:
                if isinstance(value, bool):
                    raise ValueError
                value = int(value)
            except (TypeError, ValueError):
                raise ValueError("Invalid ocr_rotation: must be 0, 90, 180 or 270")
            if value % 360 not in (0, 90, 180, 270):
                raise ValueError("Invalid ocr_rotation: must be 0, 90, 180 or 270")
            cleaned[key] = value % 360
        else:  # pragma: no cover - defensive
            raise ValueError("Unknown settings key: %s" % str(key))
    return cleaned


def _load_nolock():
    path = _settings_file()
    data = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return data
    except (OSError, ValueError):
        return data
    if not isinstance(raw, dict):
        return data
    try:
        cleaned = _validate_patch(raw)
    except ValueError:
        # Persisted file has an unknown/invalid key: keep known-valid subset.
        cleaned = {}
        for key, value in raw.items():
            if key in DEFAULTS:
                try:
                    single = _validate_patch({key: value})
                    cleaned.update(single)
                except ValueError:
                    continue
    data.update(cleaned)
    return data


def get():
    """Return a copy of the current settings."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load_nolock()
        data = dict(_cache)
        import probe_licensing as _pl
        if not _pl.allowed('ocr'):
            data.update(ocr_images=False, ocr_pdf_scans=False, ocr_new_files=False, ocr_gpu=False)
        return data


def _save_nolock():
    path = _settings_file()
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        pass


def update(patch):
    """Validate + merge a patch dict, persist, return a copy of new settings."""
    global _cache
    cleaned = _validate_patch(patch)
    with _lock:
        if _cache is None:
            _cache = _load_nolock()
        _cache.update(cleaned)
        _save_nolock()
        return dict(_cache)


def reset():
    """Reset to defaults (used by tests). Persists defaults."""
    global _cache
    with _lock:
        _cache = dict(DEFAULTS)
        _save_nolock()
        return dict(_cache)


def _reset_cache():
    """Forget in-memory cache so next get() re-reads the file (tests)."""
    global _cache
    with _lock:
        _cache = None


def as_env(settings=None):
    """Map settings to env switches consumed by index/search workers."""
    src = dict(settings) if isinstance(settings, dict) else get()
    merged = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in src:
            merged[key] = src[key]
    merged["name_memory"] = _validate_patch({"name_memory": merged["name_memory"]})["name_memory"]
    ocr_on = bool(merged.get("ocr_images")) or bool(merged.get("ocr_pdf_scans"))
    return {
        "LOCALDOCS_NAME_MEMORY": merged["name_memory"],
        "LOCALDOCS_FUZZY_NAMES": "1" if merged.get("fuzzy_names") else "0",
        "LOCALDOCS_CONTENT_SEARCH": "1" if merged.get("content_search") else "0",
        "LOCALDOCS_OCR": "on" if ocr_on else "off",
        "LOCALDOCS_OCR_IMAGES": "1" if merged.get("ocr_images") else "0",
        "LOCALDOCS_OCR_PDF": "1" if merged.get("ocr_pdf_scans") else "0",
        "LOCALDOCS_OCR_GPU": "1" if merged.get("ocr_gpu") else "0",
        "LOCALDOCS_OCR_LANG": str(merged.get("ocr_lang")),
        "LOCALDOCS_OCR_MAX_MB": str(int(merged.get("ocr_max_mb"))),
        "LOCALDOCS_OCR_ROTATION": str(int(merged.get("ocr_rotation") or 0)),
        "LOCALDOCS_OCR_SPREADS": "1" if merged.get("ocr_spreads") else "0",
        "LOCALDOCS_NAMES_ONLY_DEFAULT": "1" if merged.get("names_only_default") else "0",
    }
