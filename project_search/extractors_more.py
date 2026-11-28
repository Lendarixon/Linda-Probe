"""Additional bounded, stdlib-only extractors used alongside extractors.py."""
import codecs
import io
import json
import mailbox
import os
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

from project_search import extractors as _base

MORE_EXT = frozenset({
    '.odt', '.ods', '.odp', '.fb2', '.mbox', '.ipynb', '.srt', '.vtt',
    '.ics', '.vcf', '.log', '.tex', '.rst', '.adoc', '.org',
    '.docm', '.dotx', '.xlsm', '.xltx', '.pptm', '.potx', '.ppsx',
    '.zip', '.tar', '.tgz', '.gz', '.bz2', '.7z',
    # X3: plain-text / source-code families (decoded as text, capped)
    '.bat', '.cmd', '.php', '.rb', '.kt', '.kts', '.swift', '.lua',
    '.pl', '.pm', '.r', '.scala', '.dart', '.m', '.mm', '.vb', '.vbs',
    '.asm', '.s', '.clj', '.cljs', '.ex', '.exs', '.erl', '.hs', '.jl',
    '.sol', '.proto', '.graphql', '.gql', '.gradle', '.mk', '.cmake',
    '.properties', '.conf', '.scss', '.sass', '.less', '.jsonl', '.ndjson',
    '.csproj', '.sln', '.vcxproj', '.xaml', '.resx', '.plist', '.rss',
    '.atom', '.opml', '.tf', '.hcl', '.bib', '.markdown', '.mdx',
    # X3: zipped-XML documents
    '.xps', '.oxps', '.vsdx',
    # X3: MIME HTML archives
    '.mht', '.mhtml',
})

_MAX_MEMBERS = 2000
_MAX_MEMBER = 50 * 1024 * 1024
_MAX_TOTAL = 500 * 1024 * 1024
_MAX_RATIO = 200
_TEXT_EXT = frozenset({'.txt', '.srt', '.vtt', '.ics', '.vcf', '.log', '.tex', '.rst', '.adoc', '.org',
    '.bat', '.cmd', '.php', '.rb', '.kt', '.kts', '.swift', '.lua',
    '.pl', '.pm', '.r', '.scala', '.dart', '.m', '.mm', '.vb', '.vbs',
    '.asm', '.s', '.clj', '.cljs', '.ex', '.exs', '.erl', '.hs', '.jl',
    '.sol', '.proto', '.graphql', '.gql', '.gradle', '.mk', '.cmake',
    '.properties', '.conf', '.scss', '.sass', '.less', '.jsonl', '.ndjson',
    '.csproj', '.sln', '.vcxproj', '.xaml', '.resx', '.plist', '.rss',
    '.atom', '.opml', '.tf', '.hcl', '.bib', '.markdown', '.mdx'})
_OOXML = frozenset({'.docm', '.dotx', '.xlsm', '.xltx', '.pptm', '.potx', '.ppsx'})

# X3: bounds for plain-text extraction (safety: never return huge blobs).
_MAX_TEXT_BYTES = 2 * 1024 * 1024
_MAX_TEXT_CHARS = 1000000
_MAX_MHT_BYTES = 10 * 1024 * 1024
_BIN_CTRL = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17,
                       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 127})


def _local_tag(tag):
    return tag.rsplit('}', 1)[-1].rsplit(':', 1)[-1]


def _looks_binary(data):
    """Heuristic: True when bytes are clearly not text. Never raises."""
    try:
        if not data:
            return False
        if data.startswith(codecs.BOM_UTF8):
            data = data[len(codecs.BOM_UTF8):]
            if not data:
                return False
        if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
            return False
        if b'\x00' in data:
            return True
        sample = data[:8192]
        if not sample:
            return False
        bad = sum(1 for b in sample if b in _BIN_CTRL)
        if bad == 0:
            return False
        return (bad / len(sample)) > 0.02
    except Exception:
        return False


def _decode_text(data):
    """Decode text bytes: utf-8-sig, utf-16 BOM, cp1251, cp1250 fallbacks."""
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        try:
            return data.decode('utf-16')
        except Exception:
            pass
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        pass
    except Exception:
        pass
    try:
        return data.decode('cp1251')
    except UnicodeDecodeError:
        pass
    except Exception:
        pass
    try:
        return data.decode('cp1250')
    except UnicodeDecodeError:
        pass
    except Exception:
        pass
    try:
        return _base._read_text_bytes(data)
    except Exception:
        return data.decode('utf-8-sig', errors='replace')


def _text_from_bytes(raw):
    """Bounded bytes -> text or None. Never raises, never returns binary garbage."""
    try:
        if raw is None or not raw:
            return None
        data = raw[:_MAX_TEXT_BYTES] if len(raw) > _MAX_TEXT_BYTES else raw
        if _looks_binary(data):
            return None
        text = _decode_text(data)
        if not text or not text.strip():
            return None
        if '\x00' in text:
            return None
        head = text[:8192]
        if head:
            bad = sum(1 for ch in head if ord(ch) < 32 and ch not in ('\t', '\n', '\r'))
            if len(head) >= 256 and (bad / len(head)) > 0.02:
                return None
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS]
        return text if text.strip() else None
    except Exception:
        return None


def _text_from_path(path):
    """Bounded file -> text or None. Never raises."""
    try:
        with open(path, 'rb') as f:
            data = f.read(_MAX_TEXT_BYTES + 1)
        if len(data) > _MAX_TEXT_BYTES:
            data = data[:_MAX_TEXT_BYTES]
        return _text_from_bytes(data)
    except Exception:
        return None


def _xml_text(root, breaks):
    out = []
    def visit(node):
        name = node.tag.rsplit('}', 1)[-1].lower()
        if name in breaks and out and out[-1] != '\n':
            out.append('\n')
        if node.text:
            out.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                out.append(child.tail)
        if name in breaks and (not out or out[-1] != '\n'):
            out.append('\n')
    visit(root)
    return ''.join(out).strip()


def _odf(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('content.xml'))
    return _xml_text(root, {'p', 'h', 'table-row', 'row', 'page'}) or None


def _fb2(path):
    root = ET.fromstring(Path(path).read_bytes())
    return _xml_text(root, {'p', 'subtitle', 'title', 'section', 'v', 'text-author'}) or None


def _message_text(msg):
    def _decoded(value):
        try:
            from email.header import decode_header, make_header
            return str(make_header(decode_header(str(value or ''))))
        except Exception:
            return str(value or '')
    headers = [_decoded(msg.get(k)) for k in ('subject', 'from', 'to')]
    plain = html = None
    parts = msg.walk() if msg.is_multipart() else (msg,)
    for part in parts:
        if part.is_multipart() or part.get_content_disposition() == 'attachment':
            continue
        typ = part.get_content_type()
        try:
            body = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True)
            body = _base._read_text_bytes(payload) if payload else ''
        if isinstance(body, str) and body.strip():
            if typ == 'text/plain' and plain is None:
                plain = body
            elif typ == 'text/html' and html is None:
                html = body
    main = plain or (_base._strip_tags(html) if html else '')
    result = '\n'.join(x for x in headers + [main] if x and x.strip())
    return result or None


def _mbox(path):
    box = mailbox.mbox(str(path), create=False)
    try:
        parts = []
        for i, msg in enumerate(box):
            if i >= 5000:
                break
            text = _message_text(msg)
            if text:
                parts.append(text)
        return '\n\n'.join(parts) or None
    finally:
        box.close()


def _ipynb(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    parts = []
    for cell in doc.get('cells', []):
        if cell.get('cell_type') not in ('markdown', 'code'):
            continue
        source = cell.get('source', '')
        parts.append(''.join(source) if isinstance(source, list) else str(source))
    text = '\n\n'.join(x for x in parts if x.strip())
    return text or None


def _docx(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
    paras = []
    for p in root.iter():
        if p.tag.rsplit('}', 1)[-1] == 'p':
            s = ''.join((e.text or '') for e in p.iter() if e.tag.rsplit('}', 1)[-1] == 't')
            if s.strip():
                paras.append(s)
    return '\n'.join(paras) or None


def _xps_from_zip(z):
    """Collect UnicodeString of Glyphs elements from *.fpage in page order."""
    try:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.fpage'))
    except Exception:
        return None
    if not names:
        return None
    parts = []
    total = 0
    for n in names[:500]:
        try:
            info = z.getinfo(n)
            if info.file_size > 10 * 1024 * 1024:
                continue
            data = z.read(n)
        except Exception:
            continue
        if not data or len(data) > 10 * 1024 * 1024:
            continue
        try:
            root = ET.fromstring(data)
        except Exception:
            continue
        for el in root.iter():
            try:
                if _local_tag(el.tag).lower() != 'glyphs':
                    continue
                u = el.get('UnicodeString')
            except Exception:
                continue
            if u and u.strip():
                parts.append(u.strip())
                total += len(u)
                if total > _MAX_TEXT_CHARS:
                    break
        if total > _MAX_TEXT_CHARS:
            break
    text = '\n'.join(parts)
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS]
    return text if text.strip() else None


def _xps(path):
    try:
        with zipfile.ZipFile(path) as z:
            return _xps_from_zip(z)
    except Exception:
        return None


def _xps_bytes(raw):
    try:
        if not raw or len(raw) > 100 * 1024 * 1024:
            return None
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return _xps_from_zip(z)
    except Exception:
        return None


def _vsdx_from_zip(z):
    """Collect <Text> element text from visio/pages/*.xml in sorted order."""
    try:
        names = sorted(n for n in z.namelist()
                       if n.lower().startswith('visio/pages/') and n.lower().endswith('.xml'))
    except Exception:
        return None
    if not names:
        return None
    parts = []
    total = 0
    for n in names[:500]:
        try:
            info = z.getinfo(n)
            if info.file_size > 10 * 1024 * 1024:
                continue
            data = z.read(n)
        except Exception:
            continue
        if not data or len(data) > 10 * 1024 * 1024:
            continue
        try:
            root = ET.fromstring(data)
        except Exception:
            continue
        for el in root.iter():
            try:
                if _local_tag(el.tag).lower() != 'text':
                    continue
                s = ''.join(el.itertext()).strip()
            except Exception:
                continue
            if s:
                parts.append(s)
                total += len(s)
                if total > _MAX_TEXT_CHARS:
                    break
        if total > _MAX_TEXT_CHARS:
            break
    text = '\n'.join(parts)
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS]
    return text if text.strip() else None


def _vsdx(path):
    try:
        with zipfile.ZipFile(path) as z:
            return _vsdx_from_zip(z)
    except Exception:
        return None


def _vsdx_bytes(raw):
    try:
        if not raw or len(raw) > 100 * 1024 * 1024:
            return None
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return _vsdx_from_zip(z)
    except Exception:
        return None


def _mht(path):
    """Parse MIME HTML archive via email parser; plain + stripped html."""
    try:
        from email import policy
        from email.parser import BytesParser
        with open(path, 'rb') as f:
            raw = f.read(_MAX_MHT_BYTES + 1)
        if len(raw) > _MAX_MHT_BYTES:
            raw = raw[:_MAX_MHT_BYTES]
        if not raw:
            return None
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        return _message_text(msg)
    except Exception:
        return None


def _mht_bytes(raw):
    try:
        from email import policy
        from email.parser import BytesParser
        if not raw:
            return None
        data = raw[:_MAX_MHT_BYTES] if len(raw) > _MAX_MHT_BYTES else raw
        msg = BytesParser(policy=policy.default).parsebytes(data)
        return _message_text(msg)
    except Exception:
        return None


def _safe_name(name):
    if not name or '\x00' in name or name.startswith(('\\\\', '//')):
        return False
    norm = name.replace('\\', '/')
    p = PurePosixPath(norm)
    return not p.is_absolute() and all(part not in ('..', '') for part in norm.split('/'))


def _archive_entries(data, suffix):
    """Yield safe (name, bytes) entries, enforcing aggregate declared-size limits."""
    total = count = 0
    if suffix == '.7z':
        try:
            import py7zr
        except ImportError:
            return
        with py7zr.SevenZipFile(io.BytesIO(data), mode='r') as z:
            for name in z.getnames()[:_MAX_MEMBERS]:
                if not _safe_name(name):
                    continue
                # py7zr extraction APIs vary; readall is supported by current releases.
                for n, stream in z.readall().items():
                    if n == name:
                        raw = stream.read(_MAX_MEMBER + 1)
                        if len(raw) <= _MAX_MEMBER:
                            yield name, raw
                        break
        return
    if suffix in ('.tar', '.tgz', '.gz', '.bz2'):
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as tf:
            for member in tf:
                count += 1
                if count > _MAX_MEMBERS:
                    break
                if not member.isfile() or not _safe_name(member.name):
                    continue
                size = member.size
                if size < 0 or size > _MAX_MEMBER:
                    continue
                total += size
                if total > _MAX_TOTAL:
                    break
                if size > max(len(data), 1) * _MAX_RATIO:
                    continue
                f = tf.extractfile(member)
                raw = f.read(_MAX_MEMBER + 1) if f else b''
                if len(raw) == size and len(raw) <= _MAX_MEMBER:
                    yield member.name, raw
        return
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            count += 1
            if count > _MAX_MEMBERS:
                break
            if info.is_dir() or not _safe_name(info.filename) or info.flag_bits & 1:
                continue
            if info.file_size > _MAX_MEMBER:
                continue
            total += info.file_size
            if total > _MAX_TOTAL:
                break
            if info.file_size > max(info.compress_size, 1) * _MAX_RATIO:
                continue
            try:
                raw = z.read(info)
            except Exception:
                continue
            yield info.filename, raw


def _archive(path, data=None, depth=0):
    if depth > 2:
        return None
    data = Path(path).read_bytes() if data is None else data
    suffix = Path(path).suffix.lower()
    if str(path).lower().endswith('.tar.gz'):
        suffix = '.gz'
    parts = []
    for name, raw in _archive_entries(data, suffix):
        inner = Path(name)
        try:
            text = _extract_bytes(inner.suffix.lower(), raw, inner.name, depth + 1)
        except Exception:
            continue
        if text:
            parts.append('=== ' + name + ' ===\n' + text)
    return '\n\n'.join(parts) or None


def _extract_bytes(suffix, raw, name, depth):
    try:
        if suffix in ('.zip', '.tar', '.tgz', '.gz', '.bz2', '.7z'):
            return _archive(name, raw, depth) if depth <= 2 else None
        if suffix in _TEXT_EXT:
            return _text_from_bytes(raw)
        if suffix in ('.csv', '.tsv'):
            text = _base._read_text_bytes(raw[:2 * 1024 * 1024])
            return text if text.strip() else None
        if suffix in ('.html', '.htm'):
            text = _base._strip_tags(_base._read_text_bytes(raw))
            return text if text.strip() else None
        if suffix == '.eml':
            from email import policy
            from email.parser import BytesParser
            return _message_text(BytesParser(policy=policy.default).parsebytes(raw))
        if suffix in ('.mht', '.mhtml'):
            return _mht_bytes(raw)
        if suffix in ('.xps', '.oxps'):
            return _xps_bytes(raw)
        if suffix == '.vsdx':
            return _vsdx_bytes(raw)
        if suffix in ('.odt', '.ods', '.odp'):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                root = ET.fromstring(z.read('content.xml'))
            return _xml_text(root, {'p', 'h', 'table-row', 'row', 'page'}) or None
        if suffix == '.fb2':
            return _xml_text(ET.fromstring(raw), {'p', 'subtitle', 'title', 'section', 'v'}) or None
        if suffix == '.ipynb':
            doc = json.loads(raw.decode('utf-8-sig'))
            return '\n\n'.join(''.join(c.get('source', [])) if isinstance(c.get('source', []), list) else str(c.get('source', '')) for c in doc.get('cells', []) if c.get('cell_type') in ('markdown', 'code')) or None
        if suffix in ('.xlsx', '.xlsm', '.xltx'):
            return _base._xlsx(io.BytesIO(raw))
        if suffix in ('.pptx', '.pptm', '.potx', '.ppsx'):
            return _base._pptx(io.BytesIO(raw))
        if suffix in ('.docx', '.docm', '.dotx'):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                root = ET.fromstring(z.read('word/document.xml'))
            return '\n'.join(''.join(e.text or '' for e in p.iter() if e.tag.rsplit('}', 1)[-1] == 't') for p in root.iter() if p.tag.rsplit('}', 1)[-1] == 'p') or None
        return None
    except Exception:
        return None


def extract_more(path):
    """Extract supported additional formats; returns None on empty or any error."""
    try:
        suffix = Path(path).suffix.lower()
        low = str(path).lower()
        if suffix in _TEXT_EXT:
            return _text_from_path(path)
        if suffix in ('.odt', '.ods', '.odp'):
            return _odf(path)
        if suffix == '.fb2':
            return _fb2(path)
        if suffix == '.mbox':
            return _mbox(path)
        if suffix == '.ipynb':
            return _ipynb(path)
        if suffix in _OOXML:
            if suffix in ('.xlsm', '.xltx'):
                return _base._xlsx(path)
            if suffix in ('.pptm', '.potx', '.ppsx'):
                return _base._pptx(path)
            return _docx(path)
        if suffix in ('.xps', '.oxps'):
            return _xps(path)
        if suffix == '.vsdx':
            return _vsdx(path)
        if suffix in ('.mht', '.mhtml'):
            return _mht(path)
        if suffix in ('.zip', '.tar', '.tgz', '.gz', '.bz2', '.7z') or low.endswith('.tar.gz'):
            return _archive(path)
    except Exception:
        return None
    return None
