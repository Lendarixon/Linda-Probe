"""Stdlib-only text extractors for office/mail/web/archive formats.

extract(path) -> str | None: plain text, or None when unsupported, empty or
broken. Never raises on parsing problems: any error means None so the
caller (backend.py) simply skips the file and indexing continues.
"""
import hashlib
import os
import re
import sys
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path


def _local(tag):
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


class _TextParser(HTMLParser):
    """Visible text collector, skipping script/style content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ('script', 'style'):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in ('script', 'style') and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _strip_tags(html):
    p = _TextParser()
    try:
        p.feed(html)
    except Exception:
        pass
    return '\n'.join(p.parts)


def _read_text_bytes(data):
    import codecs as _c
    if data.startswith((_c.BOM_UTF16_LE, _c.BOM_UTF16_BE)):
        try:
            return data.decode('utf-16')
        except Exception:
            pass
    if data.startswith((_c.BOM_UTF32_LE, _c.BOM_UTF32_BE)):
        try:
            return data.decode('utf-32')
        except Exception:
            pass
    if b'\x00' in data:
        # Possibly BOM-less UTF-16 text; binary otherwise. Never mask with
        # errors="ignore": only accept a plausible text decode.
        for _enc in ('utf-16-le', 'utf-16-be'):
            try:
                _cand = data.decode(_enc)
            except Exception:
                continue
            if '\x00' in _cand:
                continue
            _s = _cand.strip().lstrip('\ufeff')
            if _s and sum(ch.isalpha() or ch.isspace() for ch in _s[:2000]) / max(1, len(_s[:2000])) > 0.5:
                return _cand.lstrip('\ufeff')
        # Fall through to strict decoders below; undecodable NUL content
        # stays an error for the caller to skip honestly.
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:
            return data.decode('cp1251')
        except UnicodeDecodeError:
            try:
                return data.decode('cp1250')
            except UnicodeDecodeError:
                return data.decode('utf-8-sig', errors='replace')


def _xlsx(path):
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = []
        if 'xl/sharedStrings.xml' in names:
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.iter():
                if _local(si.tag) == 'si':
                    shared.append(''.join((e.text or '') for e in si.iter()
                                          if _local(e.tag) == 't'))
        out = []
        sheets = sorted(n for n in names
                        if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
        for sn in sheets:
            root = ET.fromstring(z.read(sn))
            for c in root.iter():
                if _local(c.tag) != 'c':
                    continue
                t = c.get('t')
                if t == 's':
                    v = ''.join((e.text or '') for e in c.iter()
                                if _local(e.tag) == 'v').strip()
                    try:
                        out.append(shared[int(v)])
                    except (ValueError, IndexError):
                        pass
                elif t == 'inlineStr':
                    out.append(''.join((e.text or '') for e in c.iter()
                                       if _local(e.tag) == 't'))
                else:
                    for e in c.iter():
                        if _local(e.tag) == 'v' and (e.text or '').strip():
                            out.append(e.text.strip())
        if not any(s.strip() for s in out):
            out = list(shared)
        text = '\n'.join(s for s in out if s and s.strip())
        return text if text.strip() else None


def _pptx(path):
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as z:
        slides = sorted(n for n in z.namelist()
                        if re.match(r'ppt/slides/slide\d+\.xml$', n))
        if not slides:
            return None
        parts = []
        for sn in slides:
            root = ET.fromstring(z.read(sn))
            texts = [(e.text or '').strip() for e in root.iter()
                     if _local(e.tag) == 't' and (e.text or '').strip()]
            if texts:
                parts.append(' '.join(texts))
        text = '\n'.join(parts)
        return text if text.strip() else None


def _eml(path):
    from email import policy
    from email.parser import BytesParser
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    head = [str(msg['subject'] or ''), str(msg['from'] or ''),
            str(msg['to'] or '')]

    def _body(part):
        try:
            content = part.get_content()
        except Exception:
            content = None
        if isinstance(content, str) and content.strip():
            return content
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            return None
        if not payload:
            return None
        return _read_text_bytes(bytes(payload))

    plain, html = None, None
    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            if part.get_content_disposition() == 'attachment':
                continue
            ctype = part.get_content_type()
            body = _body(part)
            if not body:
                continue
            if ctype == 'text/plain' and plain is None:
                plain = body
            elif ctype == 'text/html' and html is None:
                html = body
    else:
        body = _body(msg)
        if body:
            if msg.get_content_type() == 'text/html':
                html = body
            else:
                plain = body
    main = plain if plain and plain.strip() else (_strip_tags(html) if html else '')
    text = '\n'.join(s for s in head + [main] if s and s.strip())
    return text if text.strip() else None


def _html_file(path):
    text = _strip_tags(_read_text_bytes(Path(path).read_bytes()))
    return text if text.strip() else None


def _rtf(path):
    """Bounded stdlib RTF -> text. Never raises (None on error/empty)."""
    import re as _re
    try:
        with Path(path).open('rb') as source:
            data = source.read(2 * 1024 * 1024)
    except Exception:
        return None
    if not data:
        return None
    try:
        # --- codepage from \ansicpgN (default 1252) ---
        enc = 'cp1252'
        try:
            m = _re.search(rb'\\ansicpg(\d+)', data[:8192])
            if m:
                cand = 'cp' + m.group(1).decode('ascii', errors='ignore')
                b'a'.decode(cand)  # validate codec exists
                enc = cand
        except Exception:
            enc = 'cp1252'
        _IGNORE = frozenset({
            'fonttbl', 'colortbl', 'stylesheet', 'info',
            'pict', 'shppict', 'nonshppict',
            'object', 'objdata', 'objclass', 'fldinst',
        })
        _NEWLINE = frozenset({'par', 'line', 'page', 'sect', 'column', 'row', 'lbr'})
        _TAB = frozenset({'tab', 'cell'})
        _SPECIAL = {
            'emdash': '\u2014', 'endash': '\u2013',
            'bullet': '\u2022', 'lquote': '\u2018',
            'rquote': '\u2019', 'ldblquote': '\u201c',
            'rdblquote': '\u201d',
        }
        out = []
        out_len = 0
        CAP = 1000000  # bound output chars
        ignore_stack = [False]
        expect_stack = [False]  # True right after '{': next word may be destination
        state_stack = []
        uc = 1
        skip = 0  # pending \u fallback chars to skip
        n = len(data)
        i = 0

        def _emit(s):
            nonlocal out_len
            if out_len + len(s) > CAP:
                return False  # stop
            out.append(s)
            out_len += len(s)
            return True

        def _clear_expect():
            if expect_stack[-1]:
                expect_stack[-1] = False

        while i < n:
            # --- consume \u fallback chars first ---
            if skip > 0:
                b = data[i]
                if b == 92 and i + 1 < n:  # backslash
                    nb = data[i + 1]
                    if nb == 39:  # \'xx fallback (one logical char)
                        if i + 3 < n:
                            i += 4
                        else:
                            i = n
                        skip -= 1
                        continue
                    elif nb in (92, 123, 125):  # \\ \{ \} fallback
                        i += 2
                        skip -= 1
                        continue
                    elif nb == 126 or nb in (95, 45):
                        # ~, _, \- symbols are never valid \u fallbacks
                        # per spec; treat as real content below
                        skip = 0
                    elif (65 <= nb <= 90) or (97 <= nb <= 122):
                        # control word, not a fallback char: stop skipping
                        skip = 0
                    else:
                        i += 2  # other control symbol counts as fallback
                        skip -= 1
                        continue
                elif b == 123 or b == 125:
                    skip = 0  # structure is not fallback content
                else:
                    i += 1  # single (possibly 8-bit) fallback byte
                    skip -= 1
                    continue
            b = data[i]
            if b == 123:  # '{'
                if len(ignore_stack) > 512:
                    return None
                state_stack.append((uc, enc))
                ignore_stack.append(ignore_stack[-1])
                expect_stack.append(True)
                i += 1
                continue
            elif b == 125:  # '}'
                if len(ignore_stack) > 1:
                    ignore_stack.pop()
                    expect_stack.pop()
                    uc, enc = state_stack.pop()
                i += 1
                continue
            elif b == 92:  # backslash
                if i + 1 >= n:
                    i += 1
                    continue
                nb = data[i + 1]
                if nb == 39:  # \'xx hex byte escape
                    hx = data[i + 2:i + 4]
                    if len(hx) == 2 and all(
                            (48 <= c <= 57) or (65 <= c <= 70) or (97 <= c <= 102)
                            for c in hx):
                        if not ignore_stack[-1]:
                            try:
                                if not _emit(bytes((int(hx, 16),)).decode(enc, errors='replace')):
                                    break
                            except Exception:
                                pass
                        _clear_expect()
                        i += 4
                    else:
                        i += 2
                    continue
                elif nb in (92, 123, 125):  # \\ \{ \}
                    if not ignore_stack[-1]:
                        if not _emit(chr(nb)):
                            break
                    _clear_expect()
                    i += 2
                    continue
                elif nb == 42:  # \* : star destination marker
                    if expect_stack[-1]:
                        ignore_stack[-1] = True
                        # keep expect True: next word names the destination
                    i += 2
                    continue
                elif nb == 126:  # \~ nbsp
                    if not ignore_stack[-1]:
                        if not _emit(' '):
                            break
                    _clear_expect()
                    i += 2
                    continue
                elif nb == 95:  # \_ non-breaking hyphen
                    if not ignore_stack[-1]:
                        if not _emit('-'):
                            break
                    _clear_expect()
                    i += 2
                    continue
                elif nb == 45:  # \- optional hyphen -> nothing
                    _clear_expect()
                    i += 2
                    continue
                elif nb in (10, 13):  # \<newline> ignored
                    i += 2
                    continue
                elif (65 <= nb <= 90) or (97 <= nb <= 122):
                    j = i + 1
                    while j < n and ((65 <= data[j] <= 90) or (97 <= data[j] <= 122)):
                        j += 1
                    try:
                        word = data[i + 1:j].decode('ascii')
                    except Exception:
                        i = j
                        continue
                    k = j
                    neg = False
                    if k < n and data[k] == 45:
                        # '-' starts a numeric param only if a digit follows;
                        # (\- symbol was already handled, so bare '-' here
                        # means negative param)
                        if k + 1 < n and 48 <= data[k + 1] <= 57:
                            neg = True
                            k += 1
                        else:
                            # stray '-', treat as delimiter-less word end
                            pass
                    ds = k
                    while k < n and 48 <= data[k] <= 57:
                        k += 1
                    param = None
                    if k > ds:
                        try:
                            param = int(data[ds:k].decode('ascii'))
                            if neg:
                                param = -param
                        except Exception:
                            param = None
                    has_delim = k < n and data[k] == 32
                    if has_delim:
                        k += 1
                    lw = word.lower()
                    # destination detection at group start
                    if expect_stack[-1] and not ignore_stack[-1]:
                        if lw in _IGNORE:
                            ignore_stack[-1] = True
                        expect_stack[-1] = False
                    if ignore_stack[-1]:
                        # still honour \ucN/\ansicpgN inside ignored groups?
                        # No: fully ignored except \bin binary skip.
                        if lw == 'bin' and param:
                            try:
                                k += max(0, int(param))
                            except Exception:
                                pass
                            i = min(k, n)
                        else:
                            i = k
                        continue
                    # --- live content controls ---
                    if lw == 'uc' and param is not None:
                        try:
                            uc = max(0, min(20, int(param)))
                        except Exception:
                            pass
                    elif lw == 'u' and param is not None:
                        try:
                            v = int(param)
                            if v < 0:
                                v += 65536
                            if not _emit(chr(v)):
                                break
                        except Exception:
                            pass
                        skip = uc
                    elif lw == 'ansicpg' and param is not None:
                        try:
                            cand = 'cp' + str(int(param))
                            b'a'.decode(cand)
                            enc = cand
                        except Exception:
                            pass
                    elif lw in _NEWLINE:
                        if not _emit('\n'):
                            break
                    elif lw in _TAB:
                        if not _emit('\t'):
                            break
                    elif lw in _SPECIAL:
                        if not _emit(_SPECIAL[lw]):
                            break
                    elif lw == 'bin' and param:
                        try:
                            k += max(0, int(param))
                        except Exception:
                            pass
                    else:
                        pass  # control-word delimiter spaces are structural, not visible text
                    i = min(k, n)
                    continue
                else:  # other control symbol: skip
                    _clear_expect()
                    i += 2
                    continue
            else:
                if ignore_stack[-1]:
                    i += 1
                    continue
                if b in (10, 13, 0):
                    # raw CR/LF in RTF source is layout, not content
                    i += 1
                    continue
                if b < 128:
                    if not _emit(chr(b)):
                        break
                else:
                    try:
                        if not _emit(bytes((b,)).decode(enc, errors='replace')):
                            break
                    except Exception:
                        pass
                _clear_expect()
                i += 1
                continue
        text = ''.join(out)
        text = text.encode('utf-16-le', errors='surrogatepass').decode('utf-16-le', errors='replace')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [_re.sub(r'[ \t\x0b\x0c]+', ' ', ln).strip() for ln in text.split('\n')]
        text = '\n'.join(ln for ln in lines if ln).strip()
        return text if text else None
    except Exception:
        return None


_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'})
_JUNK_PARTS = frozenset({'node_modules', '.git', 'site-packages', '__pycache__',
                          'icons', 'thumbnails', '.cache'})
_MIN_IMAGE_BYTES = 5 * 1024


def _get_settings():
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
                    if isinstance(_d, dict):
                        return _d
                    return None
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
            _cand = Path(__file__).resolve().parents[1] / 'localdocs_settings.py'
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


def _get_ocr_module():
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
            _cand = Path(__file__).resolve().parents[1] / 'localdocs_ocr.py'
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


def _is_junk_image_path(path):
    try:
        posix = Path(path).as_posix().lower()
        if 'assets/icons' in posix:
            return True
        for part in Path(path).parts:
            if str(part).lower() in _JUNK_PARTS:
                return True
        return False
    except Exception:
        return False


_OCR_CACHE_VERSION = 2


def _ocr_cache_path(digest, engine, lang):
    return _ocr_cache_paths(digest, engine, lang)[0]


def _ocr_cache_paths(digest, engine, lang):
    safe_lang = re.sub(r'[^A-Za-z0-9]+', '_', str(lang or ''))[:64] or 'lang'
    safe_eng = re.sub(r'[^A-Za-z0-9]+', '_', str(engine or 'auto'))[:16] or 'auto'
    d = os.path.join(tempfile.gettempdir(), 'ld_ocr_cache')
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    base = '%s_%s_%s' % (digest, safe_eng, safe_lang)
    return os.path.join(d, base + '.txt'), os.path.join(d, base + '.v1.json')


def _clean_box(box):
    try:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        import math as _math
        out = []
        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            x, y = float(point[0]), float(point[1])
            if not _math.isfinite(x) or not _math.isfinite(y):
                return None
            out.append([x, y])
        return out
    except (TypeError, ValueError):
        return None


def _normalise_lines(lines):
    out = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        box = _clean_box(line.get('box'))
        if box is None:
            continue
        item = dict(line)
        item['box'] = box
        try:
            item['conf'] = float(item.get('conf', 0.0) or 0.0)
        except (TypeError, ValueError):
            item['conf'] = 0.0
        out.append(item)
    return out


def _read_ocr_sidecar(digest, engine, lang):
    """Versioned structured result or None. Corrupt/older sidecars are a miss."""
    try:
        import json as _json
        path = _ocr_cache_paths(digest, engine, lang)[1]
        with open(path, 'r', encoding='utf-8') as stream:
            data = _json.load(stream)
        if not isinstance(data, dict) or data.get('version') != _OCR_CACHE_VERSION:
            return None
        if not isinstance(data.get('text'), str) or not isinstance(data.get('lines'), list):
            return None
        data['lines'] = _normalise_lines(data['lines'])
        return data
    except Exception:
        return None


def _write_ocr_sidecar(digest, engine, lang, result, page=1):
    """Write JSON sidecar atomically; returns the legacy text path."""
    text_path, json_path = _ocr_cache_paths(digest, engine, lang)
    try:
        import json as _json
        payload = {
            'version': _OCR_CACHE_VERSION,
            'page': page,
            'engine': (result or {}).get('engine'),
            'device': (result or {}).get('device'),
            'text': str((result or {}).get('text') or ''),
            'lines': _normalise_lines((result or {}).get('lines')),
            'confidence': (result or {}).get('confidence'),
            'status': (result or {}).get('status'),
            'rotated': (result or {}).get('rotated', 0),
            'pages': (result or {}).get('pages'),
            'layout': (result or {}).get('layout'),
            'geom_chain': (result or {}).get('geom_chain'),
            'orientation': (result or {}).get('orientation'),
            'orientation_signal': (result or {}).get('orientation_signal'),
        }
        temporary = json_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            _json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, json_path)
    except Exception:
        try:
            if os.path.exists(json_path + '.tmp'):
                os.unlink(json_path + '.tmp')
        except Exception:
            pass
    return text_path


def _image_ocr(path):
    """OCR-gated image branch. Never raises; text only (backward compatible)."""
    result = _image_ocr_result(path)
    return result.get('text') if result else None


def _image_ocr_result(path):
    """OCR-gated image branch. Never raises; dict with text, lines, page=1 or None.

    The versioned JSON sidecar preserves boxes; image OCR is a single page, so
    page identity is always 1 (multi-page PDFs keep their own page numbers in
    project_search.localdocs).
    """
    try:
        settings = _get_settings()
        if not isinstance(settings, dict):
            return None
        if not settings.get('ocr_images'):
            return None
        if not settings.get('content_search'):
            return None
        try:
            import localdocs_settings as _ls
            if not _ls.ocr_during_index():
                return None  # OCR runs on demand (first search / button) and is cached, not while indexing
        except ImportError:
            pass
        if _is_junk_image_path(path):
            return None
        try:
            size = Path(path).stat().st_size
        except Exception:
            return None
        if size < _MIN_IMAGE_BYTES:
            return None
        try:
            max_mb = int(settings.get('ocr_max_mb', 25))
        except Exception:
            max_mb = 25
        try:
            if size > max_mb * 1024 * 1024:
                return None
        except Exception:
            return None
        lang = settings.get('ocr_lang') or 'rus+eng+pol'
        device = 'gpu' if settings.get('ocr_gpu') else 'cpu'
        engine = 'own' if settings.get('ocr_engine') == 'own' else 'auto'  # 'ready' = RapidOCR/Tesseract, 'own' = Linda model
        spreads = bool(settings.get('ocr_spreads'))
        try:
            rotation = int(settings.get('ocr_rotation') or 0) % 360
        except (TypeError, ValueError):
            rotation = 0
        if rotation not in (0, 90, 180, 270):
            rotation = 0
        h = hashlib.sha1()
        try:
            with open(path, 'rb') as _f:
                for _chunk in iter(lambda: _f.read(65536), b''):
                    h.update(_chunk)
        except Exception:
            return None
        digest = h.hexdigest()
        ocr_mod = _get_ocr_module()
        try:
            tag = ocr_mod.cache_tag(engine, rotation=rotation or None, spreads=spreads)
        except Exception:
            try:
                identity = ocr_mod._model_identity(engine)
            except Exception:
                identity = type(ocr_mod).__name__ if ocr_mod is not None else 'absent'
            tag = 'v%d_%s_r%d_s%d' % (_OCR_CACHE_VERSION,
                                       hashlib.sha1(str(identity).encode('utf-8')).hexdigest()[:12],
                                       rotation, 1 if spreads else 0)
        cache_digest = digest + '_' + tag
        cpath = _ocr_cache_path(cache_digest, engine, lang)
        cached = _read_ocr_sidecar(cache_digest, engine, lang)
        if cached is not None:
            cached['cached'] = True
            cached.setdefault('page', 1)
            return cached if str(cached.get('text') or '').strip() else None
        sidecar_exists = os.path.isfile(_ocr_cache_paths(cache_digest, engine, lang)[1])
        try:
            # A present-but-invalid sidecar (old version/corrupt) invalidates the
            # text file too, otherwise geometry-aware callers could never refresh.
            if not sidecar_exists and os.path.isfile(cpath):
                try:
                    _cached = Path(cpath).read_text(encoding='utf-8')
                except Exception:
                    _cached = None
                if _cached is not None:
                    if not _cached.strip():
                        return None
                    # Legacy text-only hit: text callers stay fast, geometry unknown.
                    return {'version': _OCR_CACHE_VERSION, 'page': 1, 'text': _cached,
                            'lines': [], 'engine': engine, 'cached': True,
                            'geometry': 'legacy-text-cache'}
        except Exception:
            pass
        if ocr_mod is None:
            return None
        geometry = {}
        if rotation:
            geometry['rotation'] = rotation
        if spreads:
            geometry['spreads'] = True
        try:
            res = ocr_mod.ocr_image(str(path), lang=lang, device=device, engine=engine,
                                    **geometry)
        except Exception:
            return None
        if isinstance(res, str):
            res = {'text': res}
        result = dict(res) if isinstance(res, dict) else {'text': str(res or '')}
        result.setdefault('page', 1)
        text = str(result.get('text') or '')
        if text.strip():
            try:
                Path(_write_ocr_sidecar(cache_digest, engine, lang, result)).write_text(text, encoding='utf-8')
            except Exception:
                pass
            result['cached'] = False
            return result
        # Empty/error outcomes are not persisted: a later model/setting fix
        # must be able to produce text instead of hitting an empty placeholder.
        return None
    except Exception:
        return None


def _csv(path):
    text = _read_text_bytes(Path(path).read_bytes()[:2 * 1024 * 1024])
    return text if text.strip() else None


def _epub(path):
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist()
                       if n.lower().endswith(('.xhtml', '.xhtm', '.html', '.htm')))
        if not names:
            return None
        parts = []
        for n in names:
            t = _strip_tags(_read_text_bytes(z.read(n)))
            if t.strip():
                parts.append(t)
        text = '\n'.join(parts)
        return text if text.strip() else None


def extract(path):
    """Plain text of a supported file, else None (never raises)."""
    try:
        s = Path(path).suffix.lower()
        if s == '.xlsx':
            return _xlsx(path)
        if s == '.pptx':
            return _pptx(path)
        if s == '.eml':
            return _eml(path)
        if s in ('.html', '.htm'):
            return _html_file(path)
        if s == '.rtf':
            return _rtf(path)
        if s in ('.csv', '.tsv'):
            return _csv(path)
        if s == '.epub':
            return _epub(path)
        if s in _IMAGE_EXTS:
            return _image_ocr(path)
        from project_search.extractors_more import MORE_EXT, extract_more
        if s in MORE_EXT or s in ('.gz', '.bz2'):
            return extract_more(path)
        from project_search.extractors_ole import OLE_EXT, extract_ole
        if s in OLE_EXT:
            return extract_ole(path)
    except Exception:
        return None
    return None
