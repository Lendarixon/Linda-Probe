"""LocalDocs extractor: txt/docx/pdf -> text with page marks. Additive, backend.py untouched."""
import codecs as _codecs
from pathlib import Path


def _decode_txt_bytes(data):
    """BOM-aware decode: UTF-8/16/32, then utf-8, cp1251, cp1250.

    Returns (text, encoding, warnings). Never raises; binary NUL content
    that is not valid UTF-16/32 text yields ("", "binary", [...]) instead of
    masked garbage. No silent errors="ignore".
    """
    warnings = []
    if not data:
        return "", "empty", warnings
    if data.startswith(_codecs.BOM_UTF8):
        try:
            return data[len(_codecs.BOM_UTF8):].decode("utf-8"), "utf-8-sig", warnings
        except Exception:
            pass
    if data.startswith((_codecs.BOM_UTF16_LE, _codecs.BOM_UTF16_BE)):
        try:
            return data.decode("utf-16"), "utf-16", warnings
        except Exception:
            pass
    if data.startswith((_codecs.BOM_UTF32_LE, _codecs.BOM_UTF32_BE)):
        try:
            return data.decode("utf-32"), "utf-32", warnings
        except Exception:
            pass
    # Raw NUL bytes mean UTF-16/32 text or binary. Try UTF-16 LE/BE first
    # when the pattern looks like text (alternating NULs); otherwise report
    # binary instead of decoding garbage.
    if b"\x00" in data:
        for enc in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
            try:
                cand = data.decode(enc)
            except Exception:
                continue
            if "\x00" in cand:
                continue
            stripped = cand.strip().lstrip("\ufeff")
            if not stripped:
                continue
            letters = sum(ch.isalpha() or ch.isspace() for ch in stripped[:2000])
            if letters / max(1, len(stripped[:2000])) > 0.5:
                warnings.append("no-bom-guessed-" + enc)
                return cand.lstrip("\ufeff"), enc, warnings
        return "", "binary", ["binary-nul-bytes"]
    for enc in ("utf-8", "cp1251", "cp1250"):
        try:
            return data.decode(enc), enc, warnings
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    warnings.append("fallback-replace")
    return data.decode("utf-8", errors="replace"), "utf-8-replace", warnings


def _read_txt_file(path):
    raw = Path(path).read_bytes()
    # Cap single TXT reads at 32 MiB; larger files are truncated with a warning
    # instead of loading unbounded blobs into the index.
    warnings_extra = []
    if len(raw) > 32 * 1024 * 1024:
        warnings_extra.append("truncated-32MiB")
        raw = raw[:32 * 1024 * 1024]
    text, enc, warnings = _decode_txt_bytes(raw)
    return text, enc, warnings + warnings_extra


def _docx_paragraph_text(p_el, qn):
    parts = []
    for e in p_el.iter():
        tag = e.tag
        if tag == qn("w:t") and e.text:
            parts.append(e.text)
        elif tag in (qn("w:br"), qn("w:cr")):
            parts.append("\n")
        elif tag == qn("w:tab"):
            parts.append("\t")
    return "".join(parts)


def _docx_cell_text(tc_el, qn):
    blocks = []
    for child in tc_el.iterchildren():
        if child.tag == qn("w:p"):
            t = _docx_paragraph_text(child, qn)
            if t.strip():
                blocks.append(t.strip())
        elif child.tag == qn("w:tbl"):
            nested = _docx_table_text(child, qn)
            if nested.strip():
                blocks.append(nested)
    return " ".join(blocks)


def _docx_table_text(tbl_el, qn):
    rows = []
    for tr in tbl_el.findall(qn("w:tr")):
        cells = []
        for tc in tr.findall(qn("w:tc")):
            cells.append(_docx_cell_text(tc, qn))
        # Keep empty cells as structure ("a || b"); drop fully empty rows.
        if any(c.strip() for c in cells):
            rows.append(" | ".join(cells).strip())
    return "\n".join(rows)


def _extract_docx_body(path):
    """Body paragraphs + tables in document order; nested tables recursed.

    Returns (text, warnings). Headers/footers/footnotes/comments are NOT
    extracted by design; callers report this policy instead of inventing
    Word page numbers (page identity stays 1).
    """
    import docx
    from docx.oxml.ns import qn
    doc = docx.Document(str(path))
    body = doc.element.body
    blocks = []
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            t = _docx_paragraph_text(child, qn)
            if t.strip():
                blocks.append(t.strip())
        elif child.tag == qn("w:tbl"):
            t = _docx_table_text(child, qn)
            if t.strip():
                blocks.append(t)
    warnings = ["body-only:headers-footers-footnotes-not-extracted"]
    # Fallback for documents where python-docx body iteration misses content
    # (e.g. structured document tags): raw w:p scan in document order.
    if not any(b.strip() for b in blocks):
        import xml.etree.ElementTree as _ET
        import zipfile as _zf
        try:
            with _zf.ZipFile(str(path)) as z:
                root = _ET.fromstring(z.read("word/document.xml"))
            paras = []
            for p in root.iter():
                if p.tag.rsplit("}", 1)[-1] == "p":
                    s = "".join((e.text or "") for e in p.iter()
                                if e.tag.rsplit("}", 1)[-1] == "t")
                    if s.strip():
                        paras.append(s.strip())
            if paras:
                blocks = paras
                warnings.append("fallback-raw-w-p-scan")
        except Exception as exc:
            warnings.append("fallback-failed:" + str(exc)[:80])
    return "\n\n".join(blocks), warnings
def extract_text(path):
    """Text-only compatibility contract: [(page_number, text), ...]."""
    return [(page["page"], page["text"]) for page in extract_pages(path)]
def extract_pages(path):
    """Structured page records: page, text, source, and OCR lines when available.

    Page identity is 1-based and survives both the PDF text layer and OCR.
    OCR line boxes are rendered-bitmap pixel coordinates for this page; they
    are not PDF user-space coordinates (localdocs_pdf_ocr provides that mapping
    for the invisible layer).
    """
    from pathlib import Path as _P
    p = _P(path); s = p.suffix.lower()
    if s == ".txt":
        _text, _enc, _warn = _read_txt_file(p)
        return [{"page": 1, "text": _text, "source": "text",
                 "lines": None, "encoding": _enc, "warnings": _warn}]
    if s == ".docx":
        _text, _warn = _extract_docx_body(p)
        return [{"page": 1, "text": _text,
                 "source": "docx", "lines": None, "warnings": _warn}]
    if s == ".pdf":
        from pypdf import PdfReader
        import os as _os
        budget = int(_os.environ.get("LOCALDOCS_OCR_MAXPAGES", "10"))
        r = PdfReader(str(p))
        extracted = [(i, pg.extract_text() or "") for i, pg in enumerate(r.pages)]
        ocr_pages = [i for i, text in extracted if not text.strip()][:max(0, budget)]
        if ocr_pages:
            from concurrent.futures import ThreadPoolExecutor
            workers = int(_os.environ.get(
                "LOCALDOCS_OCR_WORKERS", str(min(4, max(1, (_os.cpu_count() or 1) // 2)))))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda i: ocr_page_result(p, i), ocr_pages))
            ocr_result = dict(zip(ocr_pages, results))
        else:
            ocr_result = {}
        out = []
        for i, text in extracted:
            raw = ocr_result.get(i)
            if not text.strip() and isinstance(raw, dict):
                out.append({"page": i + 1, "text": raw.get("text") or "",
                            "source": "ocr:" + str(raw.get("source") or "unknown"),
                            "lines": raw.get("lines"), "orientation": raw.get("orientation"),
                            "engine": raw.get("engine")})
            else:
                out.append({"page": i + 1, "text": text, "source": "pdf", "lines": None})
        return out
    raise ValueError(s)
def chunk_pages(pages, path, with_geometry=False):
    """Chunks with stable page identity; geometry is opt-in to keep index size.

    Accepts either (page, text) tuples (legacy) or extract_pages() dicts. When
    with_geometry is true and a page has OCR lines, the first chunk of that page
    carries an ``ocr_geometry`` snapshot (page, boxes) for coordinate lookups.
    """
    out = []
    for record in pages:
        if isinstance(record, dict):
            page = record.get("page")
            text = record.get("text") or ""
            lines = record.get("lines")
            source = record.get("source")
        else:
            page, text = record
            lines, source = None, None
        chunks = []
        for j in range(0, max(len(text), 1), 800):
            part = text[j:j + 800]
            if part.strip():
                chunk = {"path": path, "page": page, "text": part}
                if with_geometry:
                    chunk["page_source"] = source
                    chunk["char_start"] = j
                    chunk["char_end"] = j + len(part)
                chunks.append(chunk)
        if with_geometry and chunks and lines:
            chunks[0]["ocr_geometry"] = {"page": page, "source": source,
                                         "lines": lines}
        out.extend(chunks)
    return out

import os as _os
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile

def _ensure_tesseract_path():
    """Find tesseract.exe without relying on PATH (service/scheduler launches)."""
    if _shutil.which("tesseract"):
        return
    for d in (_os.environ.get("LOCALDOCS_TESSERACT_DIR", ""),
              r"C:\Program Files\Tesseract-OCR", r"C:\Program Files (x86)\Tesseract-OCR"):
        if d and _os.path.exists(_os.path.join(d, "tesseract.exe")):
            _os.environ["PATH"] = d + _os.pathsep + _os.environ.get("PATH", "")
            return


def ocr_backend():
    # Probe entitlement: ocr_backend
    import probe_licensing as _pl
    if not _pl.allowed('ocr'):
        return None
    if _os.environ.get("LOCALDOCS_OCR", "").lower() in ("off", "0", "false"):
        return None  # explicit no-OCR mode (benchmarks, locked-down hosts): scans become needs_ocr
    _ensure_tesseract_path()
    """How to OCR image-only pages: custom cmd > tesseract > none."""
    if _os.environ.get("LOCALDOCS_OCR_CMD"):
        return "custom"
    if _shutil.which("tesseract"):
        return "tesseract"
    return None


def _run_ocr(args, png, env, timeout):
    """File-backed tesseract run: avoids pipe-reader hang on timeout kill."""
    with _tempfile.TemporaryFile() as fin, _tempfile.TemporaryFile() as fout, \
            _tempfile.TemporaryFile() as ferr:
        fin.write(png)
        fin.seek(0)
        completed = _subprocess.run(
            args, stdin=fin, stdout=fout, stderr=ferr,
            timeout=timeout, env=env, creationflags=getattr(_subprocess, 'CREATE_NO_WINDOW', 0) | (0x4000 if _os.name == 'nt' else 0))
        fout.seek(0)
        ferr.seek(0)
        out = fout.read()
        err = ferr.read()
    return _subprocess.CompletedProcess(completed.args, completed.returncode, out, err)


def _tess_conf(png, env, langs="rus+eng", psm=None):
    """OCR one image, return (text, mean word confidence). -1 conf on failure."""
    args = ["tesseract", "stdin", "stdout", "-l", langs, "tsv"]
    if psm:
        args += ["--psm", psm]
    try:
        r = _run_ocr(args, png, env, 120)
    except Exception:
        return None, -1.0
    if r.returncode != 0:
        return None, -1.0
    try:
        tsv = r.stdout.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return None, -1.0
    if len(tsv) < 2:
        return None, -1.0
    words, confs = [], []
    for line in tsv[1:]:
        cols = line.split("\t")
        if len(cols) >= 12 and cols[0] == "5" and cols[11].strip():
            try:
                confs.append(float(cols[10]))
                words.append(cols[11])
            except ValueError:
                pass
    if not words:
        return None, -1.0
    return " ".join(words), sum(confs) / len(confs)


def _tess_plain(png, env, langs="rus+eng", psm=None):
    args = ["tesseract", "stdin", "stdout", "-l", langs]
    if psm:
        args += ["--psm", psm]
    try:
        r = _run_ocr(args, png, env, 120)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        out = r.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None
    return out if out.strip() else None


def _tess(png, env):
    """Best OCR text across lang/PSM variants. Returns (text, mean_conf)."""
    import itertools as _it
    langs_order = [s for s in _os.environ.get(
        "LOCALDOCS_OCR_LANGS", "rus+eng,eng").split(",") if s]
    best, best_conf = None, -1.0
    for langs, psm in _it.product(langs_order, (None, "6")):
        text = _tess_plain(png, env, langs, psm)
        if not text:
            continue
        _, conf = _tess_conf(png, env, langs, psm)
        # Tie on conf keeps the first (multilingual) model: eng-only can never
        # read Cyrillic correctly, while rus+eng reads both scripts.
        if conf > best_conf or (conf == best_conf and best is None):
            best, best_conf = text, conf
            if conf >= 75.0:
                return best, best_conf
    return best, best_conf


def _binarize(png):
    import io as _io
    from PIL import Image as _Image
    img = _Image.open(_io.BytesIO(png)).convert("L")
    w, h = img.size
    img = img.resize((w * 2, h * 2), _Image.LANCZOS).point(lambda x: 0 if x < 160 else 255, "1")
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_page(pdf_path, page_num, dpi=300, max_mp=25):
    """Rasterize one PDF page to PNG bytes (pypdfium2), capped at max_mp megapixels."""
    import pypdfium2 as _pdfium
    doc = _pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_num]
        scale = dpi / 72
        w, h = page.get_size()
        mp = (w * scale) * (h * scale) / 1e6
        if mp > max_mp:
            scale *= (max_mp / mp) ** 0.5
        img = page.render(scale=scale).to_pil()
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()

def _deskew_search(png, env, best_len=0):
    """Try small rotations, keep the longest OCR output. Only for poor pages."""
    import io as _io
    from PIL import Image as _Image
    try:
        img = _Image.open(_io.BytesIO(png)).convert("L")
    except Exception:
        return None
    best = None
    best_conf = -1.0
    # Opt-in only: confidence-based stop costs recall (en_rot5 0.91 -> 0.64 even at conf>=90)
    early_exit = _os.environ.get("LOCALDOCS_OCR_EARLY_EXIT", "0").lower() in (
        "on", "1", "true", "yes")
    for a in (-7, -5, -3, 3, 5, 7):
        try:
            buf = _io.BytesIO()
            img.rotate(a, expand=True, fillcolor="white").save(buf, format="PNG")
            got, conf = _tess(buf.getvalue(), env)
        except Exception:
            continue
        if got and (conf, len(got)) > (best_conf, len(best or "")):
            best, best_conf = got, conf
            # 75 (the quality gate) exited too early: en_rot5 recall 0.91 -> 0.64; only near-perfect stops the search
            if early_exit and conf >= float(_os.environ.get("LOCALDOCS_OCR_EARLY_CONF", "90")):
                return best
    return best


def _tess_deskewed(png, env):
    """OSD orientation detect -> rotate -> OCR. Last local resort before cloud."""
    import io as _io
    import re as _re
    r = _run_ocr(["tesseract", "stdin", "stdout", "--psm", "0", "-l", "osd"],
                   png, env, 60)
    try:
        info = r.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None
    m = _re.search(r"Orientation in degrees:\s*(\d+)", info)
    conf = _re.search(r"Orientation confidence:\s*([\d.]+)", info)
    if not m:
        return None
    angle = int(m.group(1)) % 360
    try:
        if float(conf.group(1)) < 2.0:
            return None
    except (AttributeError, ValueError):
        pass
    if angle == 0:
        return None
    from PIL import Image as _Image
    img = _Image.open(_io.BytesIO(png))
    buf = _io.BytesIO()
    img.rotate(360 - angle, expand=True, fillcolor="white").save(buf, format="PNG")
    text, _ = _tess(buf.getvalue(), env)
    return text


def _ld_settings_dict():
    """localdocs_settings.get() dict or None. Lazy, never raises."""
    try:
        import sys as _sys
        if 'localdocs_settings' in _sys.modules:
            _m = _sys.modules['localdocs_settings']
            if _m is None:
                return None
            try:
                _get = getattr(_m, 'get', None)
                if _get is not None:
                    _d = _get()
                    return _d if isinstance(_d, dict) else None
            except Exception:
                return None
            return None
        try:
            import localdocs_settings as _ls
            _d = _ls.get()
            return _d if isinstance(_d, dict) else None
        except Exception:
            pass
        try:
            import importlib.util as _ilu
            from pathlib import Path as _P
            _cand = _P(__file__).resolve().parents[1] / 'localdocs_settings.py'
            if _cand.is_file():
                _spec = _ilu.spec_from_file_location('localdocs_settings', str(_cand))
                if _spec is not None and getattr(_spec, 'loader', None) is not None:
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    _d = _mod.get()
                    return _d if isinstance(_d, dict) else None
        except Exception:
            pass
        return None
    except Exception:
        return None


def _ld_ocr_module():
    """localdocs_ocr module or None. Lazy, never raises; respects sys.modules blocks."""
    try:
        import sys as _sys
        if 'localdocs_ocr' in _sys.modules:
            _m = _sys.modules['localdocs_ocr']
            return None if _m is None else _m
        try:
            import localdocs_ocr as _ocr
            return _ocr
        except Exception:
            pass
        try:
            import importlib.util as _ilu
            from pathlib import Path as _P
            _cand = _P(__file__).resolve().parents[1] / 'localdocs_ocr.py'
            if _cand.is_file():
                _spec = _ilu.spec_from_file_location('localdocs_ocr', str(_cand))
                if _spec is not None and getattr(_spec, 'loader', None) is not None:
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    return _mod
        except Exception:
            pass
        return None
    except Exception:
        return None


_LD_OCR_CACHE_VERSION = 2


def _ld_cache_paths(digest, engine, lang):
    """(legacy text cache, versioned structured cache) for one raster image."""
    import os as _o
    import re as _re
    import tempfile as _tf
    safe_lang = _re.sub(r'[^A-Za-z0-9]+', '_', str(lang or ''))[:64] or 'lang'
    cdir = _o.path.join(_tf.gettempdir(), 'ld_ocr_cache')
    try:
        _o.makedirs(cdir, exist_ok=True)
    except Exception:
        pass
    base = '%s_%s_%s' % (digest, engine, safe_lang)
    return _o.path.join(cdir, base + '.txt'), _o.path.join(cdir, base + '.v1.json')


def _ld_clean_lines(lines):
    out = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        box = line.get('box')
        try:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            clean = []
            for point in box:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError
                clean.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
        item = dict(line)
        item['box'] = clean
        out.append(item)
    return out


def _ld_read_cached_result(digest, engine, lang):
    """Structured cache payload, or None when absent/corrupt/old-version."""
    try:
        import json as _js
        path = _ld_cache_paths(digest, engine, lang)[1]
        with open(path, 'r', encoding='utf-8') as stream:
            data = _js.load(stream)
        if not isinstance(data, dict) or data.get('version') != _LD_OCR_CACHE_VERSION:
            return None
        if not isinstance(data.get('text'), str) or not isinstance(data.get('lines'), list):
            return None
        data['lines'] = _ld_clean_lines(data['lines'])
        return data
    except Exception:
        return None


def _ld_write_cached_result(digest, engine, lang, result, page=None):
    import json as _js
    import os as _o
    import pathlib as _pl
    text_path, json_path = _ld_cache_paths(digest, engine, lang)
    payload = {
        'version': _LD_OCR_CACHE_VERSION,
        'page': page,
        'engine': (result or {}).get('engine'),
        'device': (result or {}).get('device'),
        'text': str((result or {}).get('text') or ''),
        'lines': _ld_clean_lines((result or {}).get('lines')),
        'confidence': (result or {}).get('confidence'),
        'status': (result or {}).get('status'),
        'rotated': (result or {}).get('rotated', 0),
        'source': (result or {}).get('source'),
        'pages': (result or {}).get('pages'),
        'layout': (result or {}).get('layout'),
        'geom_chain': (result or {}).get('geom_chain'),
        'orientation': (result or {}).get('orientation'),
        'orientation_signal': (result or {}).get('orientation_signal'),
    }
    try:
        temporary = json_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            _js.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            _o.fsync(stream.fileno())
        _o.replace(temporary, json_path)
    except Exception:
        try:
            _o.unlink(json_path + '.tmp')
        except Exception:
            pass
    try:
        _pl.Path(text_path).write_text(payload['text'], encoding='utf-8')
    except Exception:
        pass
    return payload


def _ld_hybrid_result(png, lang, device, rotation=None, spreads=False):
    """localdocs_ocr.ocr_image with a versioned cache. None on any failure.

    Returns the full result dict (text, lines, engine, page when known), so
    coordinates survive for callers that need them; _ld_hybrid_text stays the
    text-only compatibility wrapper.
    """
    try:
        import hashlib as _hl
        _cfg = _ld_settings_dict() or {}
        engine = 'own' if isinstance(_cfg, dict) and _cfg.get('ocr_engine') == 'own' else 'auto'
        digest = _hl.sha1(bytes(png)).hexdigest()
        _mod0 = _ld_ocr_module()
        _rot = 0
        try:
            if rotation not in (None, 'auto'):
                _rot = int(rotation) % 360
        except (TypeError, ValueError):
            _rot = 0
        try:
            _tag = _mod0.cache_tag(engine, rotation=_rot or None, spreads=spreads)
        except Exception:
            try:
                _ident = _mod0._model_identity(engine)
            except Exception:
                _ident = type(_mod0).__name__ if _mod0 is not None else 'absent'
            _tag = 'v%d_%s_r%d_s%d' % (_LD_OCR_CACHE_VERSION,
                                        _hl.sha1(str(_ident).encode('utf-8')).hexdigest()[:12],
                                        _rot, 1 if spreads else 0)
        digest = digest + '_' + _tag
        cached = _ld_read_cached_result(digest, engine, lang)
        if cached is not None:
            return cached if str(cached.get('text') or '').strip() else None
        import pathlib as _pl
        text_path, json_path = _ld_cache_paths(digest, engine, lang)
        sidecar_exists = _pl.Path(json_path).is_file()
        try:
            # An invalid sidecar forces geometric refresh; legacy text is only a
            # migration path when no structured cache was ever written.
            if not sidecar_exists and _pl.Path(text_path).is_file():
                legacy = _pl.Path(text_path).read_text(encoding='utf-8')
                if not legacy.strip():
                    return None
                return {'version': _LD_OCR_CACHE_VERSION, 'text': legacy, 'lines': [],
                        'engine': engine, 'cached': True, 'geometry': 'legacy-text-cache'}
        except Exception:
            pass
        _mod = _ld_ocr_module()
        if _mod is None:
            return None
        try:
            _extra = {}
            if _rot:
                _extra['rotation'] = _rot
            if spreads:
                _extra['spreads'] = True
            _res = _mod.ocr_image(bytes(png), lang=lang, device=device, engine=engine, **_extra)
        except Exception:
            return None
        if isinstance(_res, str):
            _res = {'text': _res}
        if not isinstance(_res, dict):
            return None
        if not str((_res or {}).get('text') or '').strip():
            return None  # never persist an empty result over a later fixed model/setting
        payload = _ld_write_cached_result(digest, engine, lang, _res)
        return payload if str(payload.get('text') or '').strip() else None
    except Exception:
        return None


def _ld_hybrid_text(png, lang, device):
    """Text-only compatibility wrapper (legacy return contract)."""
    result = _ld_hybrid_result(png, lang, device)
    return result.get('text') if result else None


def ocr_page(pdf_path, page_num):
    """Text of an image-only page via OCR backend, or None (needs_ocr)."""
    result = ocr_page_result(pdf_path, page_num)
    return result.get('text') if result else None


def ocr_page_result(pdf_path, page_num):
    """OCR result dict for an image-only page, or None (needs_ocr).

    Keys: page (1-based), text, lines (rendered-pixel boxes or None), engine,
    source ('tesseract' / 'hybrid' / 'custom'), orientation (geometry hint).
    Coordinates are relative to the rendered page bitmap (see render_page),
    not the original PDF user space.
    """
    # Probe entitlement: ocr_page_result
    import probe_licensing as _pl
    if not _pl.allowed('ocr'):
        return None
    _s = None
    try:
        _s = _ld_settings_dict()
        if isinstance(_s, dict):
            if not _s.get('ocr_pdf_scans') or not _s.get('content_search'):
                return None
            try:
                import localdocs_settings as _ls
                if not _ls.ocr_during_index():
                    return None  # on demand only (see localdocs_settings.FORCE_OCR)
            except ImportError:
                pass
    except Exception:
        pass
    be = ocr_backend()
    if _os.environ.get('LOCALDOCS_OCR', '').lower() in ('off', '0', 'false'):
        return None
    cfg = _s if isinstance(_s, dict) else {}
    rotation = cfg.get('ocr_rotation') or 0
    spreads = bool(cfg.get('ocr_spreads'))
    # Explicitly chosen Linda OCR and geometry corrections must reach the
    # recognizer even if the legacy Tesseract pass looks confident or is absent.
    if be != 'custom' and (cfg.get('ocr_engine') == 'own' or rotation or spreads):
        try:
            png = render_page(pdf_path, page_num, int(_os.environ.get('LOCALDOCS_OCR_DPI', '300')))
            selected = _ld_hybrid_result(png, cfg.get('ocr_lang') or 'rus+eng+pol',
                                         'gpu' if cfg.get('ocr_gpu') else 'cpu',
                                         rotation=rotation or None, spreads=spreads)
            if selected and str(selected.get('text') or '').strip():
                return dict(selected, page=page_num + 1, source='selected-ocr')
        except Exception:
            pass  # Keep the established fallback when an optional engine fails.
    if be is None:
        return None
    if be == "tesseract":
        import pathlib as _pl
        _td = str(_pl.Path(__file__).resolve().parents[1] / "tessdata")
        _env = dict(_os.environ)
        if _os.path.isdir(_td):
            _env["TESSDATA_PREFIX"] = _td
        dpi = int(_env.get("LOCALDOCS_OCR_DPI", "300"))
        try:
            png = render_page(pdf_path, page_num, dpi)
        except Exception:
            return None
        cands = []  # (text, lines, engine, source)
        confs = []
        got, conf = _tess(png, _env)
        if got:
            cands.append((got, None, 'tesseract', 'tesseract'))
            confs.append(conf)
        if not cands:
            try:
                got, conf = _tess(_binarize(png), _env)
                if got:
                    cands.append((got, None, 'tesseract', 'tesseract'))
                    confs.append(conf)
            except Exception:
                pass
        best_conf = max(confs, default=-1.0)
        if best_conf < 75.0:
            try:
                got = _tess_deskewed(png, _env)
                if got:
                    cands.append((got, None, 'tesseract', 'tesseract-deskew'))
            except Exception:
                pass
            try:
                got = _deskew_search(png, _env)
                if got:
                    cands.append((got, None, 'tesseract', 'tesseract-deskew-search'))
            except Exception:
                pass
            try:
                _s2 = _s if isinstance(_s, dict) else _ld_settings_dict()
                _lang2 = str((_s2 or {}).get('ocr_lang') or 'rus+eng+pol')
                _dev2 = 'gpu' if ((_s2 or {}).get('ocr_gpu')) else 'cpu'
                import importlib.util as _ilu
                # A second opinion only helps with a real line recogniser; a second Tesseract pass just adds longer noise.
                _rot2 = int((_s2 or {}).get('ocr_rotation') or 0) % 360
                if _rot2 not in (0, 90, 180, 270):
                    _rot2 = 0
                _spread2 = bool((_s2 or {}).get('ocr_spreads'))
                _extra = (_ld_hybrid_result(png, _lang2, _dev2, rotation=_rot2 or None,
                                            spreads=_spread2)
                          if _ilu.find_spec('rapidocr') else None)
                if _extra and _extra.get('text'):
                    cands.append((str(_extra.get('text')), _extra.get('lines'),
                                  _extra.get('engine') or 'hybrid', 'hybrid'))
            except Exception:
                pass
        if not cands:
            return None
        text, lines, engine, source = max(cands, key=lambda item: len(item[0]))
        result = dict(page=page_num + 1, text=text, lines=lines, engine=engine,
                      source=source, device='cpu')
        try:
            _mod = _ld_ocr_module()
            if _mod is not None and lines:
                size = None
                try:
                    from PIL import Image as _Image
                    import io as _io
                    with _Image.open(_io.BytesIO(png)) as _im:
                        size = _im.size
                except Exception:
                    size = None
                result['orientation'] = _mod.detect_orientation(
                    lines, *(size or (None, None)))
        except Exception:
            pass
        return result
    try:
        png = render_page(pdf_path, page_num)
    except Exception:
        return None
    cmd = _os.environ["LOCALDOCS_OCR_CMD"]  # {img} and {txt} placeholders
    with _tempfile.TemporaryDirectory() as d:
        img = _os.path.join(d, "p.png")
        out = _os.path.join(d, "p.txt")
        open(img, "wb").write(png)
        with _tempfile.TemporaryFile() as _fout, _tempfile.TemporaryFile() as _ferr:
            r = _subprocess.run(cmd.format(img=img, txt=out).split(),
                                stdin=_subprocess.DEVNULL, stdout=_fout,
                                stderr=_ferr, timeout=180,
                                creationflags=getattr(_subprocess, 'CREATE_NO_WINDOW', 0) | (0x4000 if _os.name == 'nt' else 0))
            _fout.seek(0)
            _stdout = _fout.read().decode("utf-8", errors="replace")
        if r.returncode != 0:
            return None
        try:
            text = open(out, encoding="utf-8").read()
        except OSError:
            text = _stdout or ''
        if not text:
            return None
        return dict(page=page_num + 1, text=text, lines=None, engine='custom',
                    source='custom', device='cpu')
def snapshot(folder):
    from pathlib import Path as _P
    out = {}
    for p in _P(folder).rglob("*"):
        if p.is_file():
            try: out[p.relative_to(folder).as_posix()] = p.stat().st_mtime_ns
            except OSError: pass
    return out
def watch_once(folder, prev):
    cur = snapshot(folder)
    changed = [k for k in set(cur) | set(prev) if cur.get(k) != prev.get(k)]
    if changed:
        from project_search.backend import index_project
        rep = index_project(folder)
        return cur, rep, changed
    return cur, None, []
