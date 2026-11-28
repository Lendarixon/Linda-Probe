"""Local page OCR. No model downloads; CPU unless GPU is explicitly requested.

Boxes are four [x,y] points in original image coordinates; confidence is 0..1.
``auto`` runs Tesseract first and offers uncertain lines to the line recogniser.
The optional batch_size keyword is also used by the CPU benchmark.
"""
import argparse
import os
from collections import OrderedDict, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import warnings

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
OWN_MODEL = ROOT / "ocr_train/run_v1/model.onnx"
OWN_CHARSET = ROOT / "ocr_train/data_v1/charset.txt"
_LOCK = threading.RLock()  # All shared sessions, including DML, have one owner.
_CACHE = OrderedDict()
_CACHE_MAX = 8
_GPU_FAILED = False
_GPU_FAILED_AT = None
_GPU_BACKOFF_SEC = 300.0
LOW_CONFIDENCE = 0.85


def _close_entry(entry):
    """Best-effort close of a cached recogniser (fake sessions have close in tests)."""
    try:
        if isinstance(entry, tuple) and len(entry) == 2:
            sess = entry[0]
            close = getattr(sess, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            return
        # RapidOCR component: inner ORT session lives in cfg/session attrs.
        sess = None
        try:
            cfg = getattr(entry, "cfg", None)
            if isinstance(cfg, dict) and "session" in cfg:
                sess = cfg["session"]
            elif isinstance(cfg, dict):
                pass
            if sess is None:
                for attr in ("session", "_session"):
                    cand = getattr(entry, attr, None)
                    if cand is not None:
                        sess = cand
                        break
            if sess is None and isinstance(cfg, dict):
                # _Config dict stored alongside component in some versions
                pass
        except Exception:
            sess = None
        if sess is not None:
            close = getattr(sess, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    except Exception:
        pass


def _cache_get(key):
    try:
        value = _CACHE[key]
    except KeyError:
        return None
    try:
        _CACHE.move_to_end(key)
    except Exception:
        pass
    return value


def _cache_put(key, value):
    try:
        if key in _CACHE:
            old = _CACHE.pop(key)
            if old is not value:
                _close_entry(old)
        while len(_CACHE) >= _CACHE_MAX:
            _, oldest = _CACHE.popitem(last=False)
            _close_entry(oldest)
        _CACHE[key] = value
        try:
            _CACHE.move_to_end(key)
        except Exception:
            pass
    except Exception:
        # Never break recognition on cache bookkeeping.
        try:
            _CACHE[key] = value
        except Exception:
            pass


def _cache_clear():
    try:
        items = list(_CACHE.values())
    except Exception:
        items = []
    try:
        _CACHE.clear()
    except Exception:
        pass
    for entry in items:
        _close_entry(entry)


def _image(source):
    if isinstance(source, Image.Image):
        im = source.copy()  # Borrowed: the caller keeps ownership.
    else:
        with Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source) as opened:
            transposed = ImageOps.exif_transpose(opened)
            try:
                im = transposed.convert("RGBA")
            finally:
                if transposed is not opened:
                    transposed.close()
    try:
        if im.mode == "RGBA":
            background = Image.new("RGBA", im.size, "white")
            try:
                composite = Image.alpha_composite(background, im)
            finally:
                background.close()
                im.close()
            im = composite
        try:
            return im.convert("RGB")
        finally:
            im.close()
    except BaseException:
        im.close()
        raise


def _session(path, device):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if device == "gpu":
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML is unavailable")
        providers.insert(0, "DmlExecutionProvider")
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if device == "gpu" and "DmlExecutionProvider" not in session.get_providers():
        raise RuntimeError("DirectML session creation fell back to CPU")
    return session


class _Config(dict):
    __getattr__ = dict.__getitem__


def _probe_rapid_directory(task, language):
    import localdocs_models as M
    ident = 'rapid-detector' if task == 'det' else {'EN':'rapid-rec-en','LATIN':'rapid-rec-latin','CYRILLIC':'rapid-rec-cyrillic'}[language]
    try:
        st=M.status(ident)
        if st.get('installed'): return Path(st['path'])
    except (OSError,ValueError): pass
    import rapidocr
    return Path(rapidocr.__file__).resolve().parent / 'models'


def _rapid_model_version(task, language):
    """Fingerprint of the RapidOCR model file (name:size:mtime), no loads."""
    try:
        import rapidocr
    except Exception:
        return "rapid-absent"
    try:
        models = _probe_rapid_directory(task, language)
        if task == "det":
            candidates = sorted(models.glob("*det*.onnx"))
        else:
            prefix = {"CYRILLIC": "cyrillic_", "LATIN": "latin_", "EN": "en_"}[language]
            candidates = sorted(models.glob(prefix + "*rec*.onnx"))
    except Exception:
        return "rapid-unknown"
    if not candidates:
        return "rapid-missing"
    parts = []
    for cand in candidates[:4]:
        try:
            st = cand.stat()
            parts.append("%s:%d:%d" % (cand.name, st.st_size, st.st_mtime_ns))
        except OSError:
            parts.append(cand.name + ":absent")
    return ",".join(parts)


def _rapid_component(task, language, device):
    """Use installed RapidOCR preprocessing/DB/CTC with explicit local sessions."""
    version = _rapid_model_version(task, language)
    try:
        import localdocs_models
        active = str(localdocs_models.get_active("ocr_rec") or "default")
    except Exception:
        active = "models-unknown"
    key = (task, language, device, version, active)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    import rapidocr
    from rapidocr import RapidOCR, LangRec, ModelType, OCRVersion
    models = _probe_rapid_directory(task, language)
    if task == "det":
        candidates = sorted(models.glob("*det*.onnx"))
    else:
        prefix = {"CYRILLIC": "cyrillic_", "LATIN": "latin_", "EN": "en_"}[language]
        candidates = sorted(models.glob(prefix + "*rec*.onnx"))
    if not candidates:
        raise FileNotFoundError("No local RapidOCR " + task + " model for " + language)
    params = {"Global.use_cls": False, "EngineConfig.onnxruntime.use_cuda": False,
              "EngineConfig.onnxruntime.use_dml": (device == "gpu")}
    if task == "rec":
        params.update({"Rec.ocr_version": OCRVersion.PPOCRV5,
                       "Rec.model_type": ModelType.MOBILE,
                       "Rec.lang_type": getattr(LangRec, language)})
    engine = RapidOCR(params=params)  # Lazy: no models loaded or downloaded here.
    section = engine.cfg.Det if task == "det" else engine.cfg.Rec
    cfg = _Config({k: v for k, v in section.items()})
    cfg["session"] = _session(candidates[0], device)
    cfg["model_path"] = str(candidates[0])
    cfg["font_path"] = None
    if task == "det":
        from rapidocr.ch_ppocr_det import TextDetector
        component = TextDetector(cfg)
    else:
        # Embedded alphabet avoids RapidOCR's optional dictionary downloader.
        if "character" not in cfg.session.get_modelmeta().custom_metadata_map:
            raise ValueError("RapidOCR model has no embedded character dictionary")
        from rapidocr.ch_ppocr_rec import TextRecognizer
        component = TextRecognizer(cfg)
    _cache_put(key, component)
    return component


def _detect(image, device):
    bgr = np.asarray(image)[:, :, ::-1].copy()
    result = _rapid_component("det", "EN", device)(bgr)
    if result.boxes is None:
        return []
    return [np.asarray(box, dtype=np.float32).tolist() for box in result.boxes]


def detect_lines(path_or_bytes, device="cpu"):
    """DB text regions in reading order; raises if the local detector is absent."""
    _validate(device, "rapid", 16)
    with _LOCK:
        return _with_device(lambda d: _detect(_image(path_or_bytes), d), device)


def _crop(image, box):
    import cv2
    from rapidocr.utils.process_img import get_rotate_crop_image
    pixels = np.asarray(image)[:, :, ::-1].copy()
    crop = get_rotate_crop_image(pixels, np.asarray(box, dtype=np.float32).copy())
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _prepare(crop):
    # Fixed buckets make padding independent of batch size (also for the BiLSTM).
    width = max(24, min(1024, math.ceil(crop.width * 48 / crop.height)))
    bucket = max(64, ((width + 63) // 64) * 64)
    resized = crop.resize((width, 48), Image.Resampling.BILINEAR)
    try:
        padded = Image.new("RGB", (bucket, 48), "white")
        try:
            padded.paste(resized, (0, 0))
        except BaseException:
            padded.close()
            raise
        return padded, (width + 3) // 4
    finally:
        try:
            resized.close()
        except Exception:
            pass


def _decode(logits, chars, lengths):
    if logits.ndim != 3 or logits.shape[1] != len(lengths) or logits.shape[2] != len(chars) + 1:
        raise ValueError("Own model must emit [T,N,len(charset)+1] logits")
    shifted = logits - logits.max(axis=2, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=2, keepdims=True)
    ids = logits.argmax(axis=2)
    results = []
    for n, length in enumerate(lengths):
        previous, text, scores = 0, [], []
        for t, token in enumerate(ids[:length, n]):
            if token and token != previous:
                text.append(chars[int(token) - 1])
                scores.append(float(probs[t, n, token]))
            previous = token
        results.append(("".join(text), float(np.mean(scores)) if scores else 0.0))
    return results


def _own_paths():
    """The recognizer the user selected in Settings -> Models (downloaded or custom); the bundled one otherwise."""
    try:
        import localdocs_models
        mid = localdocs_models.get_active("ocr_rec")
        if mid and mid != "linda-ocr-own":
            folder = Path(localdocs_models.status(mid)["path"])
            onnx = sorted(folder.rglob("*.onnx"))
            charset = sorted(folder.rglob("*.txt"))
            if onnx and charset:
                return onnx[0], charset[0]
    except Exception:
        pass
    return OWN_MODEL, OWN_CHARSET


def _own(device):
    model, charset = _own_paths()
    if not model.is_file() or not charset.is_file():
        raise FileNotFoundError("Own ONNX model/charset absent")
    try:
        mst = model.stat()
        cst = charset.stat()
        key = ("own", str(model), mst.st_size, mst.st_mtime_ns,
               str(charset), cst.st_size, cst.st_mtime_ns, device)
    except OSError:
        key = ("own", str(model), str(charset), device)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    chars = charset.read_text(encoding="utf-8").splitlines()
    if not chars or any(len(c) != 1 for c in chars) or len(set(chars)) != len(chars):
        raise ValueError("Invalid own charset")
    value = (_session(model, device), chars)
    _cache_put(key, value)
    return value


def _languages(lang):
    tokens = set(lang.lower().split("+"))
    languages = []
    if tokens & {"rus", "ru"}:
        languages.append("CYRILLIC")
    if tokens & {"pol", "pl"}:
        languages.append("LATIN")
    if tokens & {"eng", "en"} or not languages:
        languages.append("EN")
    return languages


def _recognize_backend(crops, backend, lang, device, batch_size):
    # Own CRNN and RapidOCR need different inputs: the 48px/batched/padded
    # preparation is ONLY for the own recogniser. RapidOCR receives the
    # original line crops (RGB->BGR) so Probe no longer corrupts EN lines
    # (measured 27.34% CER after _prepare vs 0% native).
    if backend == "own":
        prepared = []
        try:
            for crop in crops:
                prepared.append(_prepare(crop))
            buckets = defaultdict(list)
            for i, (im, _) in enumerate(prepared):
                buckets[im.width].append(i)
            results = [None] * len(crops)
            own = _own(device)
            for indices in buckets.values():
                for start in range(0, len(indices), batch_size):
                    ids = indices[start:start + batch_size]
                    session, chars = own
                    pixels = np.stack([np.asarray(prepared[i][0].convert("L"), dtype=np.float32)
                                       for i in ids])[:, None] / 255.0
                    logits = session.run(None, {session.get_inputs()[0].name: pixels})[0]
                    decoded = _decode(logits, chars, [prepared[i][1] for i in ids])
                    if len(decoded) != len(ids):
                        raise RuntimeError("Recognizer returned wrong number of lines")
                    for i, (text, conf) in zip(ids, decoded):
                        results[i] = (str(text), float(np.clip(conf, 0, 1)))
            return results
        finally:
            for im, _ in prepared:
                try:
                    im.close()
                except Exception:
                    pass
    # RapidOCR path: no _prepare, no padding buckets.
    results = [None] * len(crops)
    languages = _languages(lang)
    for start in range(0, max(len(crops), 1), batch_size):
        ids = list(range(start, min(start + batch_size, len(crops))))
        if not ids:
            break
        from rapidocr.ch_ppocr_rec import TextRecInput
        images = []
        for i in ids:
            arr = np.asarray(crops[i].convert("RGB"))[:, :, ::-1].copy()
            images.append(arr)
        candidates = []
        for language in languages:
            rec = _rapid_component("rec", language, device)
            rec.rec_batch_num = batch_size
            output = rec(TextRecInput(img=images, return_word_box=False))
            candidates.append(list(zip(output.txts, output.scores)))
        decoded = [max((c[j] for c in candidates), key=lambda x: float(x[1]) if x[0] else -1)
                   for j in range(len(ids))]
        if len(decoded) != len(ids):
            raise RuntimeError("Recognizer returned wrong number of lines")
        for i, (text, conf) in zip(ids, decoded):
            results[i] = (str(text), float(np.clip(conf, 0, 1)))
    return results


def _recognize(crops, engine, lang, device, batch_size):
    if not crops:
        backend = "own" if engine != "rapid" and all(x.is_file() for x in _own_paths()) else "rapid"
        return [], backend
    backends = ["rapid"] if engine == "rapid" else ["own", "rapid"]
    for backend in backends:
        try:
            return _recognize_backend(crops, backend, lang, device, batch_size), backend
        except Exception:
            if device == "gpu":
                raise  # Retry the complete operation on CPU before backend fallback.
    results = []
    for crop in crops:
        result = _tesseract(crop, lang, psm=7)
        confidences = [line["conf"] for line in result["lines"]]
        results.append((result["text"], float(np.mean(confidences)) if confidences else 0.0))
    return results, "tesseract"


def _tesseract(image, lang, psm=3):
    executable = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    command = str(executable) if executable.is_file() else shutil.which("tesseract")
    if not command:
        raise FileNotFoundError("Tesseract executable not found")
    args = [command, "stdin", "stdout", "-l", lang, "--psm", str(psm)]
    tessdata = ROOT / "tessdata"
    if tessdata.is_dir():
        args += ["--tessdata-dir", str(tessdata)]
    args.append("tsv")
    data = io.BytesIO()
    image.save(data, format="PNG")
    # The app runs under pythonw: without CREATE_NO_WINDOW every tesseract call flashes a console window.
    process = subprocess.run(args, input=data.getvalue(), capture_output=True, timeout=120,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) | (0x4000 if os.name == 'nt' else 0))
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", "replace").strip())
    groups = {}
    dropped = 0
    for row in csv.DictReader(io.StringIO(process.stdout.decode("utf-8-sig")), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = row.get("text", "").strip()
        if row.get("level") != "5" or not text:
            continue
        try:
            word_conf = float(row.get("conf", -1))
        except ValueError:
            word_conf = -1.0
        if word_conf < 30.0:  # skewed photos otherwise yield pages of low-confidence noise
            dropped += 1
            continue
        key = tuple(row[k] for k in ("page_num", "block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append(row)
    lines = []
    for words in groups.values():
        x = min(int(w["left"]) for w in words)
        y = min(int(w["top"]) for w in words)
        right = max(int(w["left"]) + int(w["width"]) for w in words)
        bottom = max(int(w["top"]) + int(w["height"]) for w in words)
        weights = [max(1, len(w["text"])) for w in words]
        conf = sum(max(0, float(w["conf"])) * n for w, n in zip(words, weights)) / sum(weights) / 100
        lines.append({"box": [[x, y], [right, y], [right, bottom], [x, bottom]],
                      "text": " ".join(w["text"] for w in words), "conf": min(1.0, conf)})
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines, "engine": "tesseract",
            "dropped": dropped}


def _close_crops(crops):
    for crop in crops:
        try:
            crop.close()
        except Exception:
            pass


def _pipeline(image, lang, device, engine, batch_size):
    if engine == "tesseract":
        return _tesseract(image, lang)
    if engine == "auto":
        # Auto-pick per page: run rapid and tesseract, keep the better result
        # by (status rank, confidence, text length). Flat print goes to
        # tesseract (measured best), hard cases keep rapid/own. Old comment
        # about ocr_lab/pages CER is superseded by frozen-42 (P2).
        cands = []
        try:
            r = _pipeline(image, lang, device, "rapid", batch_size)
            if str(r.get("text") or "").strip() or (r.get("lines") or []):
                cands.append(r)
        except Exception:
            if device == "gpu":
                raise
        try:
            t = _tesseract(image, lang)
            if str(t.get("text") or "").strip() or (t.get("lines") or []):
                cands.append(t)
        except Exception:
            pass
        if cands:
            def _key(r):
                try:
                    st, cf, _ = _classify(r.get("text", ""), r.get("lines") or [],
                                          int(r.get("dropped", 0) or 0))
                except Exception:
                    st, cf = "empty", 0.0
                return (_STATUS_RANK.get(st, 0), float(cf or 0.0), len(str(r.get("text") or "")))
            # Flat print: tesseract is measured best (frozen-42 CER 0.07% vs
            # rapid 0.70%); confidence alone prefers rapid (0.989 vs 0.963)
            # despite worse CER. Trust a confident tesseract outright.
            for r in cands:
                if str(r.get("engine") or "") == "tesseract":
                    try:
                        st, cf, _ = _classify(r.get("text", ""), r.get("lines") or [],
                                              int(r.get("dropped", 0) or 0))
                    except Exception:
                        st, cf = "empty", 0.0
                    if st == "ok" and float(cf or 0.0) >= 0.9:
                        return r
            cands.sort(key=_key, reverse=True)
            return cands[0]
        return _tesseract(image, lang)
    if engine == "hybrid":
        try:
            first = _tesseract(image, lang)
        except Exception:
            return _pipeline(image, lang, device, "own", batch_size)
        low = [i for i, line in enumerate(first["lines"]) if line["conf"] < LOW_CONFIDENCE]
        if not first["lines"]:
            return _pipeline(image, lang, device, "own", batch_size)
        if low:
            crops = [_crop(image, first["lines"][i]["box"]) for i in low]
            try:
                alternatives, backend = _recognize(crops, "own", lang, device, batch_size)
                for i, (text, conf) in zip(low, alternatives):
                    if text.strip() and conf > first["lines"][i]["conf"]:
                        first["lines"][i].update(text=text, conf=conf)
                first["engine"] = "hybrid:tesseract+" + backend
            except Exception:
                if device == "gpu":
                    raise
                # Keep successful Tesseract output if optional second opinion fails.
            finally:
                _close_crops(crops)
        first["text"] = "\n".join(line["text"] for line in first["lines"])
        return first
    try:
        boxes = _detect(image, device)
        crops = [_crop(image, b) for b in boxes]
        try:
            decoded, backend = _recognize(crops, engine, lang, device, batch_size)
        finally:
            _close_crops(crops)
        lines = [{"box": box, "text": text, "conf": conf} for box, (text, conf) in zip(boxes, decoded)]
        return {"text": "\n".join(line["text"] for line in lines), "lines": lines, "engine": backend}
    except Exception:
        if device == "gpu":
            raise
        return _tesseract(image, lang)


def _with_device(operation, device):
    global _GPU_FAILED_AT, _GPU_FAILED
    if device == "gpu":
        now = time.monotonic()
        failed_at = _GPU_FAILED_AT
        if failed_at is None or (now - failed_at) >= _GPU_BACKOFF_SEC:
            try:
                result = operation("gpu")
                _GPU_FAILED = False
                return result
            except Exception as exc:
                _GPU_FAILED_AT = time.monotonic()
                _GPU_FAILED = True
                _cache_clear()
                warnings.warn("DirectML OCR failed; using CPU: " + str(exc), RuntimeWarning)
    return operation("cpu")


def _validate(device, engine, batch_size):
    if device not in ("cpu", "gpu") or engine not in ("auto", "hybrid", "tesseract", "rapid", "own"):
        raise ValueError("Invalid device or engine")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


# --- versioned structured OCR result cache -----------------------------------
# Text callers keep using result['text'] and the legacy .txt cache file. The
# JSON sidecar preserves line boxes, page identity and the geometry applied,
# so a newer caller can reuse coordinates without re-running recognition.
OCR_CACHE_VERSION = 2
OCR_CACHE_SIDECAR_SUFFIX = ".v1.json"


def cache_tag(engine, rotation=None, spreads=False):
    """Short cache discriminator: model identity + geometry options + schema.

    Uses the same model fingerprint as the journal. Callers with a stub module
    (tests) fall back to a stable class name, so mock compatibility is kept.
    """
    try:
        identity = _model_identity(engine)
    except Exception:
        identity = "unknown"
    try:
        fingerprint = hashlib.sha1(str(identity).encode("utf-8")).hexdigest()[:12]
    except Exception:
        fingerprint = "unknown"
    angle = 0
    try:
        if rotation not in (None, "auto"):
            angle = int(rotation) % 360
    except (TypeError, ValueError):
        angle = 0
    return "v%d_m%s_r%d_s%d" % (OCR_CACHE_VERSION, fingerprint, angle, 1 if spreads else 0)


def _cache_paths(digest, engine, lang):
    """(legacy text path, versioned JSON path) for one image/engine/language."""
    safe_lang = re.sub(r'[^A-Za-z0-9]+', '_', str(lang or ''))[:64] or 'lang'
    safe_eng = re.sub(r'[^A-Za-z0-9]+', '_', str(engine or 'auto'))[:16] or 'auto'
    directory = os.path.join(tempfile.gettempdir(), 'ld_ocr_cache')
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        pass
    base = '%s_%s_%s' % (digest, safe_eng, safe_lang)
    return (os.path.join(directory, base + '.txt'),
            os.path.join(directory, base + OCR_CACHE_SIDECAR_SUFFIX))


def _clean_box(box):
    """Four finite [x, y] points, or None. Coordinates are copied, not reinterpreted."""
    try:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        points = []
        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            points.append([x, y])
        return points
    except (TypeError, ValueError):
        return None


def normalise_lines(lines):
    """Keep boxed lines only; unknown keys are preserved for forward compatibility."""
    out = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        box = _clean_box(line.get("box"))
        if box is None:
            continue
        item = dict(line)
        item["box"] = box
        try:
            item["conf"] = float(item.get("conf", 0.0) or 0.0)
        except (TypeError, ValueError):
            item["conf"] = 0.0
        out.append(item)
    return out


def result_payload(result, page=None):
    """Versioned, JSON-serialisable subset of an OCR result (text stays untouched)."""
    result = result if isinstance(result, dict) else {"text": str(result or "")}
    payload = {
        "version": OCR_CACHE_VERSION,
        "page": page,
        "engine": result.get("engine"),
        "device": result.get("device"),
        "text": str(result.get("text") or ""),
        "lines": normalise_lines(result.get("lines")),
        "confidence": result.get("confidence"),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "rotated": result.get("rotated", 0),
        "spread": result.get("spread"),
        "pages": result.get("pages"),
    }
    return payload


def parse_payload(raw):
    """Parsed payload or None for old/corrupt/foreign cache versions."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != OCR_CACHE_VERSION:
        return None
    text = data.get("text")
    if not isinstance(text, str):
        return None
    lines = data.get("lines")
    if not isinstance(lines, list):
        return None
    data["lines"] = normalise_lines(lines)
    return data


def write_result_cache(digest, engine, lang, result, page=None):
    """Write JSON sidecar atomically and keep the legacy text file for old callers."""
    text_path, json_path = _cache_paths(digest, engine, lang)
    payload = result_payload(result, page=page)
    try:
        temporary = json_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, json_path)
    except Exception:
        try:
            if os.path.exists(json_path + ".tmp"):
                os.unlink(json_path + ".tmp")
        except Exception:
            pass
    try:
        with open(text_path, "w", encoding="utf-8") as stream:
            stream.write(payload["text"])
    except Exception:
        pass
    return text_path, json_path


def read_result_cache(digest, engine, lang):
    """Valid structured payload, or None (caller may fall back to the text file)."""
    _, json_path = _cache_paths(digest, engine, lang)
    try:
        with open(json_path, "r", encoding="utf-8") as stream:
            return parse_payload(stream.read())
    except Exception:
        return None


# --- geometry preparation hooks (no recognition, no implicit cropping) --------

_CJK = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF), (0xAC00, 0xD7AF))


def _is_cjk_text(text):
    text = str(text or "")
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    hits = sum(1 for c in chars if any(lo <= ord(c) <= hi for lo, hi in _CJK))
    return hits * 10 >= len(chars) * 3  # >=30% CJK: likely vertical writing, not rotation


def detect_orientation(lines, width=None, height=None):
    """Suggest a page rotation from box geometry only.

    Returns {angle, confidence, lines, vertical_script, reason}. ``angle`` is
    the clockwise correction (0/90/180/-90). CJK-dominant lines are treated as
    genuine vertical writing and never rotated. This is a preparation hint;
    callers still decide whether to re-run OCR on a rotated bitmap.
    """
    candidates = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        box = _clean_box(line.get("box"))
        # Short tokens (1-3 chars) have tall/narrow boxes even when upright;
        # they are not evidence of sideways text.
        if box is None or len(str(line.get("text") or "").strip()) < 4:
            continue
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        top = math.hypot(dx, dy)
        ux = box[3][0] - box[0][0]
        uy = box[3][1] - box[0][1]
        side = math.hypot(ux, uy)
        if top <= 0 or side <= 0:
            continue
        candidates.append((line, math.degrees(math.atan2(dy, dx)) % 360, top, side))
    base = {"angle": 0, "confidence": 0.0, "lines": len(candidates),
            "vertical_script": False, "reason": "no boxed text lines"}
    if len(candidates) < 3:
        return base
    cjk = sum(1 for line, _, _, _ in candidates if _is_cjk_text(line.get("text")))
    if cjk * 2 >= len(candidates):
        base.update(vertical_script=True, reason="CJK-dominant text: keep upright (vertical writing)")
        return base
    # Top edges of upright text are close to 0/180 degrees; sideways text is ~90/-90.
    sideways = 0
    for _, angle, top, side in candidates:
        # For sideways text the baseline (top edge) runs along the image's
        # vertical axis and is still longer than the glyph height.
        if top > side * 1.2 and (75 <= angle <= 105 or 255 <= angle <= 285):
            sideways += 1
    if sideways * 2 < len(candidates):
        base["reason"] = "line boxes are horizontal"
        return base
    signed = [((angle + 180) % 360) - 180 for _, angle, _, _ in candidates
              if 75 <= angle <= 105 or 255 <= angle <= 285]
    median = sorted(signed)[len(signed) // 2]
    angle = -90 if median > 0 else 90  # correction, clockwise
    base.update(angle=angle, confidence=sideways / len(candidates),
                reason="sideways line boxes (%d/%d)" % (sideways, len(candidates)))
    return base


def _ink_columns(image):
    """Per-column count of dark pixels in the central band, plus band height."""
    grey = image.convert("L")
    try:
        width, height = grey.size
        top, bottom = int(height * 0.05), max(int(height * 0.95), int(height * 0.05) + 1)
        band = grey.crop((0, top, width, bottom))
        try:
            data = band.tobytes()
            rows = band.size[1]
        finally:
            band.close()
        counts = [0] * width
        for offset in range(0, len(data), width):
            row = data[offset:offset + width]
            for x, value in enumerate(row):
                if value < 200:
                    counts[x] += 1
        return counts, rows
    finally:
        grey.close()


def spread_pages(image, *, min_gutter_frac=0.01):
    """Locate a two-page spread gutter without cutting the image.

    Requires a landscape page, a blank vertical run in the central 40-60%
    band and ink on both sides. Returns
    {gutter, left: {page, box}, right: {page, box}} in source coordinates,
    or None. Callers crop explicitly and map boxes by the box offset.
    """
    try:
        width, height = image.size
    except Exception:
        return None
    if width < height * 1.2:
        return None
    try:
        counts, rows = _ink_columns(image)
    except Exception:
        return None
    if not counts or rows <= 0:
        return None
    blank = max(1, int(rows * 0.005))
    runs, start = [], None
    for x, count in enumerate(counts):
        if count <= blank:
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, width))
    central = [(a, b) for a, b in runs
               if a < width * 0.6 and b > width * 0.4 and b - a >= width * min_gutter_frac]
    if not central:
        return None
    left_end, right_start = max(central, key=lambda run: run[1] - run[0])
    if left_end <= 0 or right_start >= width:
        return None
    left_ink = sum(counts[:left_end])
    right_ink = sum(counts[right_start:])
    if left_ink <= rows or right_ink <= rows:  # both halves must carry real content
        return None
    mid = (left_end + right_start) // 2
    return {
        "gutter": [left_end, right_start],
        "left": {"page": "L", "box": [[0.0, 0.0], [float(mid), 0.0],
                                      [float(mid), float(height)], [0.0, float(height)]]},
        "right": {"page": "R", "box": [[float(mid), 0.0], [float(width), 0.0],
                                       [float(width), float(height)], [float(mid), float(height)]]},
    }


def crop_view(image, box):
    """Crop and return (image, to_source) where to_source maps view -> source coords."""
    x0 = min(point[0] for point in box)
    y0 = min(point[1] for point in box)
    x1 = max(point[0] for point in box)
    y1 = max(point[1] for point in box)
    crop = image.crop((int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))))
    ox, oy = int(x0), int(y0)

    def to_source(box_in_view):
        mapped = _clean_box(box_in_view)
        if mapped is None:
            raise ValueError("Box must contain four finite [x, y] points")
        return [[point[0] + ox, point[1] + oy] for point in mapped]
    return crop, to_source


def map_lines(lines, transform, *, target="box"):
    """Copy line dicts with ``target`` boxes mapped; original boxes stay untouched."""
    out = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        mapped = _clean_box(transform(line.get("box")))
        if mapped is None:
            continue
        item = dict(line)
        item[target] = mapped
        out.append(item)
    return out


def rotate_page(image, degrees):
    """Rotate clockwise by a multiple of 90; returns (image, to_source).

    to_source maps boxes from the rotated bitmap back to original bitmap
    coordinates. Non-right angles are rejected instead of guessed.
    """
    angle = int(degrees) % 360
    if angle not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0/90/180/270 degrees")
    width, height = image.size
    if angle == 0:
        return image.copy(), lambda box: _clean_box(box)
    if angle == 90:
        view = image.transpose(Image.Transpose.ROTATE_270)  # 270 CCW = 90 CW
        return view, lambda box: [[p[1], height - p[0]] for p in _clean_box(box)]
    if angle == 180:
        view = image.transpose(Image.Transpose.ROTATE_180)
        return view, lambda box: [[width - p[0], height - p[1]] for p in _clean_box(box)]
    view = image.transpose(Image.Transpose.ROTATE_90)  # 90 CCW = 270 CW
    return view, lambda box: [[width - p[1], p[0]] for p in _clean_box(box)]


    view = image.transpose(Image.Transpose.ROTATE_90)  # 90 CCW = 270 CW
    return view, lambda box: [[width - p[1], p[0]] for p in _clean_box(box)]


_STATUS_RANK = {"ok": 2, "uncertain": 1, "empty": 0}


def _osd_signal(image):
    """Tesseract OSD angle/confidence. Returns dict or None (missing binary)."""
    try:
        import io as _io
        buf = _io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
    except Exception:
        return None
    try:
        import subprocess as _sp
        from localdocs_ocr import _run_ocr as _ro  # local alias, same module
    except Exception:
        _ro = None
    try:
        import subprocess as _sp2
        exe = shutil.which("tesseract")
        if not exe:
            return None
        proc = _sp2.run([exe, "stdin", "stdout", "--psm", "0", "-l", "osd"],
                        input=data, capture_output=True, timeout=60,
                        creationflags=getattr(_sp2, 'CREATE_NO_WINDOW', 0))
        out = (proc.stdout or b"").decode("utf-8", "replace")
    except Exception:
        return None
    import re as _re2
    m = _re2.search(r"Orientation in degrees:\s*(\d+)", out)
    c = _re2.search(r"Orientation confidence:\s*([\d.]+)", out)
    if not m:
        return None
    try:
        osd = int(m.group(1)) % 360
        conf = float(c.group(1)) / 15.0 if c else 0.0
    except (ValueError, AttributeError):
        return None
    angle_cw = (360 - osd) % 360  # tesseract reports CCW detect; we correct CW
    return {"osd_ccw": osd, "angle_cw": angle_cw, "confidence": max(0.0, min(1.0, conf)), "raw": out[:2000]}


def orientation_signal_4way(image, lang="eng"):
    """4-way orientation signal: OSD (primary) + PSM6 rec-vote + box-vote.

    180 is decided by OSD, never by box tilt. Returns
    {angle_cw, confidence, needs_manual, signals}. No recognition side effects
    on the caller's image; thumbnails only.
    """
    box_hint = detect_orientation([])
    try:
        box_hint = detect_orientation(None) or box_hint
    except Exception:
        pass
    osd = _osd_signal(image)
    votes = {}
    if osd is not None:
        votes[osd["angle_cw"]] = votes.get(osd["angle_cw"], 0.0) + float(osd.get("confidence", 0.0)) * 2.0
    # PSM6 rec-vote on a small thumbnail (independent cross-check; PSM3
    # self-rotates and is unsuitable as an orientation signal).
    rec_confs = {}
    try:
        thumb = image.copy()
        thumb.thumbnail((800, 800))
        for cand in (0, 90, 180, 270):
            try:
                view = thumb if cand == 0 else thumb.transpose(
                    {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                     270: Image.Transpose.ROTATE_90}[cand])
                r = _tesseract(view, lang, psm=6)
                n = len(r.get("lines") or [])
                conf = sum(float(l.get("conf", 0.0)) for l in r.get("lines") or []) / max(1, n)
                rec_confs[cand] = (conf, n, str(r.get("text") or ""))
            except Exception:
                continue
            finally:
                if cand != 0:
                    try:
                        view.close()
                    except Exception:
                        pass
        try:
            thumb.close()
        except Exception:
            pass
        if rec_confs:
            best = max(rec_confs, key=lambda c: (rec_confs[c][0], rec_confs[c][1]))
            margin = 0.0
            try:
                top2 = sorted((v[0] for v in rec_confs.values()), reverse=True)[:2]
                margin = top2[0] - top2[1] if len(top2) == 2 else top2[0]
            except Exception:
                margin = 0.0
            votes[best] = votes.get(best, 0.0) + margin
    except Exception:
        rec_confs = {}
    angle = 0
    total = sum(votes.values())
    if votes:
        angle = max(votes, key=lambda k: votes[k])
    conf = 0.0
    if total > 0:
        conf = votes.get(angle, 0.0) / total
    if osd is not None:
        # OSD is the verifiable primary: trust a confident OSD outright.
        if float(osd.get("confidence", 0.0)) >= 0.6:
            angle, conf = int(osd["angle_cw"]) % 360, float(osd.get("confidence", 0.0))
    needs_manual = (not votes) or (conf < 0.6)
    return {"angle_cw": int(angle) % 360, "confidence": float(conf),
            "needs_manual": bool(needs_manual),
            "signals": {"osd": osd, "rec_vote": {str(k): v[0] for k, v in (rec_confs or {}).items()},
                        "box_hint": box_hint}}


def deskew_search(image, ocr_fn, angles=(-7, -5, -3, 3, 5, 7), threshold=0.85):
    """Try small rotations for poor pages only. Returns (result, applied_angle).

    ocr_fn(view) -> result dict with text/lines/confidence/status. Candidates
    are ranked by (status_rank, confidence, text_len) - never by raw strings.
    Boxes are mapped back to the original bitmap; the input image is untouched.
    """
    best, best_key, applied = None, None, 0
    for a in (0,) + tuple(angles):
        try:
            if a == 0:
                view, back = image, None
            else:
                import math as _math
                w, h = image.size
                view = image.rotate(a, expand=True, fillcolor="white")
                # Inverse affine of PIL rotate(a, expand) about centers.
                rad = _math.radians(-a)
                ca, sa = _math.cos(rad), _math.sin(rad)
                nw, nh = view.size
                cx, cy, ncx, ncy = w / 2.0, h / 2.0, nw / 2.0, nh / 2.0
                def back(box, _ca=ca, _sa=sa):
                    pts = _clean_box(box)
                    if pts is None:
                        raise ValueError("bad box")
                    return [[(p[0] - ncx) * _ca - (p[1] - ncy) * _sa + cx,
                             (p[0] - ncx) * _sa + (p[1] - ncy) * _ca + cy] for p in pts]
            res = ocr_fn(view)
            st = str(res.get("status") or ("ok" if str(res.get("text") or "").strip() else "empty"))
            cf = float(res.get("confidence") or 0.0)
            ln = len(str(res.get("text") or ""))
            key = (_STATUS_RANK.get(st, 0), cf, ln)
            if best is None or key > best_key:
                if back is not None:
                    try:
                        res["lines"] = map_lines(res.get("lines"), back)
                    except Exception:
                        pass
                    res["deskew"] = a
                best, best_key, applied = res, key, a
            if a != 0:
                try:
                    view.close()
                except Exception:
                    pass
        except Exception:
            continue
    return best, applied


def layout_classify(image):
    """spread|columns2|table|single without cutting. Cheap geometry only.

    spread: landscape + verified central gutter (reuses spread_pages).
    columns2: portrait with a central vertical gutter and ink on both sides.
    table: grid of >=3 long horizontal and >=3 long vertical lines (cv2 when
      available, else 'single' - never guessed).
    """
    try:
        w, h = image.size
    except Exception:
        return {"layout": "single", "detail": "no size"}
    found = None
    try:
        found = spread_pages(image)
    except Exception:
        found = None
    if found:
        return {"layout": "spread", "detail": "verified gutter", "gutter": found.get("gutter")}
    # Portrait central gutter -> two text columns (do NOT cut like a spread).
    if h >= w:
        try:
            counts, rows = _ink_columns(image)
            blank = max(1, int(rows * 0.005))
            runs, start = [], None
            for x, c in enumerate(counts):
                if c <= blank:
                    if start is None:
                        start = x
                elif start is not None:
                    runs.append((start, x))
                    start = None
            if start is not None:
                runs.append((start, w))
            central = [(a, b) for a, b in runs if a < w * 0.55 and b > w * 0.45 and b - a >= w * 0.008]
            if central:
                g0, g1 = max(central, key=lambda r: r[1] - r[0])
                if sum(counts[:g0]) > rows and sum(counts[g1:]) > rows:
                    return {"layout": "columns2", "detail": "central gutter portrait",
                            "gutter": [g0, g1]}
        except Exception:
            pass
    # Table grid (Hough when cv2 present).
    try:
        import cv2 as _cv2
        import numpy as _np
        g = image.convert("L")
        gw, gh = g.size
        scale = 800.0 / max(1, max(gw, gh))
        small = g.resize((max(1, int(gw * scale)), max(1, int(gh * scale))))
        arr = _np.asarray(small)
        edges = _cv2.Canny(arr, 50, 150)
        lines = _cv2.HoughLinesP(edges, 1, _np.pi / 180, 120,
                                 minLineLength=int(small.size[0] * 0.4), maxLineGap=10)
        if lines is not None:
            hh = vv = 0
            for x1, y1, x2, y2 in lines[:, 0, :]:
                if abs(x2 - x1) > abs(y2 - y1) * 3:
                    hh += 1
                elif abs(y2 - y1) > abs(x2 - x1) * 3:
                    vv += 1
            if hh >= 3 and vv >= 3:
                return {"layout": "table", "detail": "grid h=%d v=%d" % (hh, vv)}
    except Exception:
        pass
    return {"layout": "single", "detail": "no spread/columns/table evidence"}


def order_columns2(lines, width):
    """Reading order for a two-column page: left column top-down, then right."""
    items = []
    for line in lines or []:
        box = _clean_box(line.get("box") if isinstance(line, dict) else None)
        if box is None:
            continue
        cx = sum(p[0] for p in box) / 4.0
        cy = sum(p[1] for p in box) / 4.0
        items.append((0 if cx < width / 2.0 else 1, cy, line))
    items.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in items]


def _model_identity(engine):
    """Cheap opaque fingerprint of the recogniser behind engine (no loads or inference).

    Returns 'unknown' when the model files/registry cannot be inspected; the
    journal then falls back to legacy matching for that call.
    """
    try:
        parts = [engine]
        if engine in ("own", "hybrid", "auto"):
            model, charset = _own_paths()
            for candidate in (model, charset):
                try:
                    stat = candidate.stat()
                    parts.append("%s:%d:%d" % (candidate.name, stat.st_size, stat.st_mtime_ns))
                except OSError:
                    parts.append(candidate.name + ":absent")
        if engine in ("rapid", "hybrid", "auto"):
            parts.append("rapid:" + _rapid_model_version("det", "EN"))
            for _lang in ("CYRILLIC", "LATIN", "EN"):
                parts.append(_lang + ":" + _rapid_model_version("rec", _lang))
            try:
                import localdocs_models
                parts.append(str(localdocs_models.get_active("ocr_rec") or "default"))
            except Exception:
                parts.append("models-unknown")
        return "|".join(parts)
    except Exception:
        return "unknown"


def _classify(text, lines, dropped):
    """Split confident ok / true empty / low-confidence uncertain outcomes.

    Thresholds (LOW_CONFIDENCE, tesseract word 30/100) are pre-existing code
    constants, kept unchanged; see stage/OCR_REPORT.md for the contract gap.
    """
    confs = [float(line.get("conf", 0.0)) for line in lines
             if isinstance(line, dict)]
    confidence = sum(confs) / len(confs) if confs else 0.0
    if text.strip():
        if confs and confidence < LOW_CONFIDENCE:
            return "uncertain", confidence, (
                "mean line confidence %.2f below %.2f; retry with another "
                "engine, language or higher DPI" % (confidence, LOW_CONFIDENCE))
        return "ok", confidence, ""
    if dropped or lines:
        return "uncertain", 0.0, (
            "words detected but none confident enough "
            "(tesseract word filter 30/100); retry with another engine, "
            "language or higher DPI")
    return "empty", 0.0, "no text detected in image"


def ocr_image(path_or_bytes, lang="rus+eng+pol", device="cpu", engine="auto", *, batch_size=16,
              skip_done=None, rotation=None, spreads=False, deskew=False):
    """OCR a path or in-memory image.

    Path calls skip known empty/three-failure inputs by default, never ok text.
    Explicit rereads use skip_done=False (or LOCALDOCS_REEXTRACT=1).
    In-memory PDF pages are journaled by the file-level batch caller.

    Optional geometry (all default off, old callers unchanged):
    rotation: 0/90/180/270 clockwise correction, 'auto' for a conservative
              sideways-line hint (never guesses 180 from boxes), or 'auto4'
              for the 4-way signal (OSD primary + PSM6 rec-vote + box-vote;
              180 decided by OSD, conf<0.6 leaves image unrotated with
              needs_manual=True for the existing rotation= UI).
    spreads:  explicit opt-in two-page spread split; parts are labelled L/R and
              every box is mapped back to the original bitmap. A wide page
              without a verified gutter is OCRed whole.
    deskew:   opt-in small-angle search (0,±3,±5,±7) for poor pages only
              (status uncertain/empty or confidence below 0.85); boxes are
              mapped back, the input image is never mutated.
    layout:   always classified (spread|columns2|table|single) and recorded in
              result['layout'] with the coordinate chain in geom_chain.
    """
    # Probe entitlement: ocr_image
    import probe_licensing as _pl
    _pl.require('ocr')
    _validate(device, engine, batch_size)
    started = time.perf_counter()
    import localdocs_ocr_journal as journal
    file_key = None
    identity = _model_identity(engine)
    model = None if identity == "unknown" else identity
    if isinstance(path_or_bytes, (str, os.PathLike)):
        try:
            file_key = journal.signature(path_or_bytes)
            skip = skip_done if skip_done is not None else os.environ.get('LOCALDOCS_REEXTRACT') != '1'
            if skip and journal.should_skip(path_or_bytes, engine, model=model,
                                           lang=lang, device=device):
                return dict(text='', lines=[], engine=engine, seconds=0.0, skipped=True,
                            confidence=0.0, reason='previously recorded outcome; use '
                            'skip_done=False or journal.retry() for an explicit retry')
        except Exception as exc:
            warnings.warn('OCR journal unavailable: ' + str(exc), RuntimeWarning)
    if rotation is not None and rotation not in ("auto", "auto4"):
        try:
            if int(rotation) % 360 not in (0, 90, 180, 270):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("rotation must be 0/90/180/270, 'auto' or 'auto4'")
    if not isinstance(deskew, bool):
        raise ValueError("deskew must be a bool")
    actual_device = device
    def run_view(view, inverse=None, part=None):
        nonlocal actual_device
        def operation(d):
            nonlocal actual_device
            actual_device = d
            return _pipeline(view, lang, d, engine, batch_size)
        with _LOCK:
            recognized = _with_device(operation, device)
        if inverse is not None:
            mapped = []
            for line in recognized.get("lines") or []:
                if not isinstance(line, dict):
                    continue
                try:
                    box = inverse(line.get("box"))
                except Exception:
                    continue
                item = dict(line)
                item["box"] = box
                if part:
                    item["part"] = part
                mapped.append(item)
            recognized["lines"] = mapped
        return recognized
    try:
        image = _image(path_or_bytes)
        work, rotate_back = image, None
        geom_chain, layout_info, orient_signal = [], {"layout": "single", "detail": "pending"}, None
        try:
            if spreads and rotation in ("auto", "auto4"):
                raise ValueError("auto rotation with spreads is not supported; choose 0/90/180/270")
            if rotation == "auto4":
                try:
                    orient_signal = orientation_signal_4way(image, lang)
                except Exception as exc:
                    orient_signal = {"angle_cw": 0, "confidence": 0.0, "needs_manual": True,
                                     "signals": {}, "error": str(exc)[:200]}
                ang, ocf = int(orient_signal.get("angle_cw", 0)) % 360, float(orient_signal.get("confidence", 0.0) or 0.0)
                geom_chain.append({"op": "orientation_signal_4way", "angle_cw": ang,
                                   "confidence": ocf, "needs_manual": bool(orient_signal.get("needs_manual"))})
                if not orient_signal.get("needs_manual") and ang:
                    work, rotate_back = rotate_page(image, ang)
                    geom_chain.append({"op": "rotate", "angle_cw": ang, "coord": "source"})
                    result_rotated = ang
                else:
                    result_rotated = 0
            elif rotation not in (None, "auto") and int(rotation) % 360:
                work, rotate_back = rotate_page(image, int(rotation))
                geom_chain.append({"op": "rotate", "angle_cw": int(rotation) % 360, "coord": "source"})
            if spreads:
                found = spread_pages(work)
                if found:
                    parts = []
                    for label, key in (("L", "left"), ("R", "right")):
                        crop, crop_to = crop_view(work, found[key]["box"])
                        def to_source(box, _crop=crop_to, _back=rotate_back):
                            mapped = _crop(box)
                            return _back(mapped) if _back is not None else mapped
                        try:
                            part_result = run_view(crop, to_source, part=label)
                        finally:
                            crop.close()
                        part_result.update(page=1, part=label)
                        parts.append((part_result, key))
                    def source_box(box, _back=rotate_back):
                        return _back(_clean_box(box)) if _back is not None else _clean_box(box)
                    gutter_box = None
                    try:
                        g0, g1 = found["gutter"]
                        height = work.size[1]
                        gutter_box = source_box([[g0, 0], [g0, height], [g1, height], [g1, 0]])
                    except Exception:
                        gutter_box = None
                    result = dict(
                        text="\n\n".join(str(p.get("text") or "") for p, _ in parts),
                        lines=[dict(line, page=1) for p, _ in parts for line in p.get("lines") or []],
                        engine="+".join(sorted({str(p.get("engine") or "") for p, _ in parts})),
                        pages=[dict(page=1, part=p["part"], text=p.get("text") or "",
                                    lines=p.get("lines") or [], engine=p.get("engine"),
                                    box=source_box(found[key]["box"])) for p, key in parts],
                        spread={"gutter": found["gutter"], "gutter_box": gutter_box,
                                "parts": ["L", "R"], "rotation": int(rotation or 0) % 360,
                                "coord": "source"},
                    )
                else:
                    result = run_view(work, rotate_back)
            elif rotation == "auto":
                result = run_view(image)
                hint = detect_orientation(result.get("lines"))
                base_status, base_conf, _ = _classify(
                    result.get("text", ""), result.get("lines") or [],
                    int(result.get("dropped", 0) or 0))
                if (abs(int(hint.get("angle", 0))) == 90 and
                        float(hint.get("confidence", 0.0)) >= 0.8 and
                        base_status in ("uncertain", "empty")):
                    corrected, to_source = rotate_page(image, hint["angle"])
                    try:
                        retry = run_view(corrected, to_source)
                    finally:
                        corrected.close()
                    retry["rotated"] = hint["angle"]
                    retry_status, retry_conf, _ = _classify(
                        retry.get("text", ""), retry.get("lines") or [],
                        int(retry.get("dropped", 0) or 0))
                    if (retry_status != "empty" and
                            (retry_status, retry_conf, len(str(retry.get("text") or ""))) >=
                            (base_status, base_conf, len(str(result.get("text") or "")))):
                        result = retry
            else:
                result = run_view(work, rotate_back)
            if rotate_back is not None and rotation != "auto4":
                result["rotated"] = int(rotation) % 360
            if rotation == "auto4":
                result["orientation_signal"] = orient_signal
                if 'result_rotated' in dir() and result_rotated:
                    result["rotated"] = result_rotated
            # Layout classification (cheap geometry, never cuts implicitly).
            try:
                if spreads and result.get("spread"):
                    layout_info = {"layout": "spread", "detail": "verified gutter",
                                   "gutter": result["spread"].get("gutter")}
                else:
                    layout_info = layout_classify(work)
                    if layout_info.get("layout") == "columns2":
                        try:
                            result["lines"] = order_columns2(result.get("lines"), work.size[0])
                        except Exception:
                            pass
                geom_chain.append({"op": "layout", **{k: v for k, v in layout_info.items() if k != "detail"},
                                   "detail": layout_info.get("detail")})
            except Exception as exc:
                layout_info = {"layout": "single", "detail": "classify-failed"}
                geom_chain.append({"op": "layout", "layout": "single", "error": str(exc)[:120]})
            result["layout"] = layout_info.get("layout", "single")
            result["geom_chain"] = geom_chain
            # Opt-in deskew for poor pages only (status/conf gate, rank compare).
            try:
                st0, cf0 = _classify(result.get("text", ""), result.get("lines") or [],
                                     int(result.get("dropped", 0) or 0))[:2]
            except Exception:
                st0, cf0 = "uncertain", 0.0
            if deskew and not spreads and st0 in ("uncertain", "empty") or (deskew and not spreads and cf0 < 0.85):
                def _fn(view):
                    def _op(d):
                        return _pipeline(view, lang, d, engine, batch_size)
                    with _LOCK:
                        return _with_device(_op, device)
                try:
                    best, applied = deskew_search(work, _fn)
                    if best is not None:
                        try:
                            bst = _classify(best.get("text", ""), best.get("lines") or [],
                                            int(best.get("dropped", 0) or 0))
                        except Exception:
                            bst = ("empty", 0.0, "")
                        cur = (_STATUS_RANK.get(st0, 0), cf0, len(str(result.get("text") or "")))
                        new = (_STATUS_RANK.get(bst[0], 0), float(bst[1] or 0.0), len(str(best.get("text") or "")))
                        if new > cur:
                            if rotate_back is not None:
                                try:
                                    best["lines"] = map_lines(best.get("lines"), rotate_back)
                                except Exception:
                                    pass
                            best["geom_chain"] = geom_chain + [{"op": "deskew", "angle": applied, "coord": "source"}]
                            best["layout"] = result.get("layout", "single")
                            if orient_signal is not None:
                                best["orientation_signal"] = orient_signal
                            result = best
                except Exception:
                    pass
        finally:
            if work is not image:
                work.close()
            image.close()
    except Exception as exc:
        if file_key is not None:
            _journal_record(journal, path_or_bytes, 'error', engine, actual_device, lang,
                            seconds=time.perf_counter() - started, error=str(exc), file_signature=file_key,
                            model=model)
        raise
    result["seconds"] = time.perf_counter() - started
    result["device"] = actual_device
    status, confidence, reason = _classify(result.get('text', ''),
                                           result.get('lines', []) or [],
                                           int(result.get('dropped', 0) or 0))
    result["confidence"] = confidence
    result["reason"] = reason
    result["status"] = status
    if file_key is not None:
        _journal_record(journal, path_or_bytes, status, engine,
                        actual_device, lang, chars=len(result.get('text', '')),
                        seconds=result['seconds'], error=reason,
                        file_signature=file_key, model=model)
    return result


def _journal_record(journal, *args, **kwargs):
    # A locked/unwritable journal must not discard successfully recognized text.
    try:
        journal.record(*args, **kwargs)
    except Exception as exc:
        warnings.warn('OCR journal write failed: ' + str(exc), RuntimeWarning)


def _norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def _distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def benchmark(directory):
    directory = Path(directory)
    truth = json.loads((directory / "truth.json").read_text(encoding="utf-8"))
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    if not files:
        raise ValueError("No benchmark images")
    print("engine\tbatch\tpages\tsec/page\tCER\tchar_errors/chars\tactual_engine\tstatus")
    for label, engine in (("tesseract", "tesseract"), ("rapid", "rapid"), ("hybrid", "hybrid"), ("auto", "auto")):
        for batch in (1, 16, 64):
            with _LOCK:
                _cache_clear()
            seconds = errors = characters = completed = 0
            actual, failures = set(), []
            # Cold model loads are included; no hidden warmup or GPU work.
            for path in files:
                try:
                    language = path.stem.split("_")[0]
                    reference = truth.get(path.name, truth.get(language))
                    if reference is None:
                        raise ValueError("Missing ground truth for " + path.name)
                    reference = _norm(" ".join(reference) if isinstance(reference, list) else reference)
                    result = ocr_image(path, lang={"ru": "rus", "pl": "pol", "en": "eng"}.get(language, "rus+eng+pol"),
                                       device="cpu", engine=engine, batch_size=batch, skip_done=False)
                    seconds += result["seconds"]
                    errors += _distance(_norm(result["text"]), reference)
                    characters += len(reference)
                    completed += 1
                    actual.add(result["engine"])
                except Exception as exc:
                    failures.append(path.name + ": " + str(exc).replace("\n", " "))
            # Never present a partial run as complete quality or throughput.
            valid = completed == len(files) and characters > 0
            speed = f"{seconds / completed:.4f}" if valid else "UNKNOWN"
            cer = f"{errors / characters:.6f}" if valid else "UNKNOWN"
            status = "OK" if valid else "INCOMPLETE " + "; ".join(failures)
            print(f"{label}\t{batch}\t{completed}/{len(files)}\t{speed}\t{cer}\t{errors}/{characters}\t{','.join(sorted(actual))}\t{status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--lang", default="rus+eng+pol")
    parser.add_argument("--engine", choices=("auto", "hybrid", "tesseract", "rapid", "own"), default="auto")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.bench:
        benchmark(args.bench)
    elif args.image:
        print(json.dumps(ocr_image(args.image, args.lang, args.device, args.engine, batch_size=args.batch_size, skip_done=False), ensure_ascii=True))
    else:
        parser.error("image or --bench is required")
