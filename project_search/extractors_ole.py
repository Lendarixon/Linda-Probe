"""Bounded, stdlib-only text extraction for legacy OLE Office files."""
import struct
from pathlib import Path

from project_search._ext.olefile import olefile

OLE_EXT = {'.doc', '.xls', '.ppt', '.msg', '.dot'}
MAX_FILE = 100 * 1024 * 1024
MAX_OUTPUT = 20_000_000


def _u16(data, offset):
    return struct.unpack_from('<H', data, offset)[0]


def _u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def _clip(text):
    return text[:MAX_OUTPUT]


def _doc_bytes(data):
    """Extract Word text from a WordDocument stream; heuristic fallback."""
    try:
        if len(data) < 32:
            return None
        fc_min, fc_mac = _u32(data, 24), _u32(data, 28)
        if fc_min <= fc_mac <= len(data):
            # Best-effort fallback over the main text range, used if CLX is absent.
            fallback = data[fc_min:fc_mac].decode('utf-16le', errors='ignore')
        else:
            fallback = ''
        return fallback.strip() or None
    except Exception:
        return None


def _doc_text(word, table):
    """Parse CLX piece table from WordDocument and selected table stream."""
    try:
        if len(word) < 32 or len(table) < 4:
            return None
        # FibBase occupies 32 bytes. Walk the variable FIB arrays to fc/lcb pairs.
        csw = _u16(word, 32)
        offset = 34 + csw * 2
        if offset + 2 > len(word):
            return _doc_bytes(word)
        cslw = _u16(word, offset)
        pair_base = offset + 2 + cslw * 4
        ccf = _u16(word, pair_base)
        pair_base += 2
        # FibRgFcLcb97 pair index 33 is fcClx/lcbClx.
        if ccf <= 33:
            return _doc_bytes(word)
        pair = pair_base + 33 * 8
        if pair + 8 > len(word):
            return _doc_bytes(word)
        fc, size = _u32(word, pair), _u32(word, pair + 4)
        if size < 1 or fc + size > len(table):
            return _doc_bytes(word)
        clx = table[fc:fc + size]
        pos = 0
        while pos < len(clx) and clx[pos] == 1:
            if pos + 3 > len(clx):
                return _doc_bytes(word)
            pos += 3 + _u16(clx, pos + 1)
        if pos >= len(clx) or clx[pos] != 2 or pos + 5 > len(clx):
            return _doc_bytes(word)
        length = _u32(clx, pos + 1)
        plc = clx[pos + 5:pos + 5 + length]
        if len(plc) != length or length < 16 or (length - 4) % 12:
            return _doc_bytes(word)
        n = (length - 4) // 12
        cps = [_u32(plc, i * 4) for i in range(n + 1)]
        pieces = []
        for i in range(n):
            pcd = 4 * (n + 1) + i * 8
            raw_fc = _u32(plc, pcd + 2)
            compressed = bool(raw_fc & 0x40000000)
            fc = raw_fc & 0x3fffffff
            chars = max(0, cps[i + 1] - cps[i])
            if compressed:
                start, end = fc // 2, fc // 2 + chars
                if end > len(word):
                    continue
                pieces.append(word[start:end].decode('cp1252', errors='replace'))
            else:
                end = fc + chars * 2
                if end > len(word):
                    continue
                pieces.append(word[fc:end].decode('utf-16le', errors='replace'))
        result = ''.join(pieces).strip()
        return _clip(result) if result else _doc_bytes(word)
    except Exception:
        return _doc_bytes(word)


def _records(data):
    """Yield bounded BIFF (type, payload) records, including CONTINUE as records."""
    pos = 0
    while pos + 4 <= len(data):
        kind, size = struct.unpack_from('<HH', data, pos)
        pos += 4
        if size > len(data) - pos:
            break
        yield kind, data[pos:pos + size]
        pos += size


def _biff_string(data, pos):
    """Decode BIFF8 XLUnicodeRichExtendedString at pos; return (text, newpos)."""
    cch = _u16(data, pos)
    flags = data[pos + 2]
    pos += 3
    runs = _u16(data, pos) if flags & 0x08 else 0
    pos += 2 if flags & 0x08 else 0
    ext = _u32(data, pos) if flags & 0x04 else 0
    pos += 4 if flags & 0x04 else 0
    width = 2 if flags & 1 else 1
    nbytes = cch * width
    raw = data[pos:pos + nbytes]
    if len(raw) != nbytes:
        raise ValueError('short BIFF string')
    text = raw.decode('utf-16le' if width == 2 else 'cp1252', errors='replace')
    pos += nbytes + runs * 4 + ext
    return text, pos


def _biff_text(data):
    """Extract basic BIFF8 cell values, SST strings, and worksheet names."""
    try:
        recs = list(_records(data))
        shared = []
        # SST may span CONTINUE records. Concatenation supports records split
        # between strings; character-boundary option-byte handling is best effort.
        for i, (kind, payload) in enumerate(recs):
            if kind != 0x00FC or len(payload) < 8:
                continue
            _, unique = struct.unpack_from('<II', payload)
            buf = bytearray(payload[8:])
            j = i + 1
            while j < len(recs) and recs[j][0] == 0x003C:
                buf.extend(recs[j][1])
                j += 1
            p = 0
            for _ in range(min(unique, 1_000_000)):
                # If a string crosses CONTINUE, the next record starts with a
                # compression flag. Keep parsing bounded and stop on malformed data.
                value, p = _biff_string(buf, p)
                shared.append(value)
            break
        out = []
        for kind, p in recs:
            try:
                if kind == 0x0085 and len(p) >= 8:  # BOUNDSHEET
                    name_len, flags = p[6], p[7]
                    size = name_len * (2 if flags & 1 else 1)
                    out.append(p[8:8 + size].decode('utf-16le' if flags & 1 else 'cp1252', errors='replace'))
                elif kind == 0x00FD and len(p) >= 10:  # LABELSST
                    idx = _u32(p, 6)
                    if idx < len(shared):
                        out.append(shared[idx])
                elif kind == 0x0204 and len(p) >= 8:  # LABEL
                    text, _ = _biff_string(p, 6)
                    out.append(text)
                elif kind == 0x0203 and len(p) >= 14:  # NUMBER
                    val = struct.unpack_from('<d', p, 6)[0]
                    out.append(str(val))
                elif kind == 0x027E and len(p) >= 10:  # RK
                    rk = _u32(p, 6)
                    if rk & 2:
                        value = struct.unpack('<i', struct.pack('<I', rk))[0] >> 2
                        number = float(value)
                    else:
                        number = struct.unpack('<d', struct.pack('<II', 0, rk & 0xfffffffc))[0]
                    if rk & 1:
                        number /= 100
                    out.append(str(number))
            except Exception:
                continue
        result = '\n'.join(x for x in out if x and x.strip())
        return _clip(result) if result else None
    except Exception:
        return None


def _ppt_text(data):
    """Extract TextCharsAtom/TextBytesAtom payloads from PowerPoint records."""
    try:
        out, pos = [], 0
        while pos + 8 <= len(data):
            options, kind, size = struct.unpack_from('<HHI', data, pos)
            pos += 8
            if options & 0x0F == 0x0F:
                continue  # container record: descend into its child records
            if size > len(data) - pos:
                break
            payload = data[pos:pos + size]
            if kind == 0x0FA0 and len(payload) % 2 == 0:
                out.append(payload.decode('utf-16le', errors='replace'))
            elif kind == 0x0FA8:
                out.append(payload.decode('cp1252', errors='replace'))
            pos += size
        seen, uniq = set(), []
        for item in out:
            item = item.strip()
            if len(item) > 1 and item not in seen:  # masters repeat the same placeholders
                seen.add(item)
                uniq.append(item)
        result = '\n'.join(uniq)
        return _clip(result) if result else None
    except Exception:
        return None


def _msg_text(ole):
    """Read common Outlook MAPI subject/body/sender/recipient properties."""
    try:
        streams = {tuple(x)[-1]: x for x in ole.listdir(streams=True, storages=False)}
        ids = [('0037', 'Subject'), ('1000', 'Body'), ('0C1A', 'From'), ('0E04', 'To')]
        out = []
        for prop, label in ids:
            for typ in ('001F', '001E'):
                name = '__substg1.0_' + prop + typ
                entry = streams.get(name)
                if entry is None:
                    continue
                raw = ole.openstream(entry).read(MAX_OUTPUT * 2)
                value = raw.decode('utf-16le' if typ == '001F' else 'cp1252', errors='replace').rstrip('\x00').strip()
                if value:
                    out.append(label + ': ' + value)
                    break
        result = '\n'.join(out)
        return _clip(result) if result else None
    except Exception:
        return None


def extract_ole(path):
    """Return text from legacy Office OLE files, or None on any error."""
    try:
        p = Path(path)
        if p.suffix.lower() not in OLE_EXT or p.stat().st_size > MAX_FILE or p.stat().st_size <= 0:
            return None
        ole = olefile.OleFileIO(str(p))
        try:
            names = {tuple(x)[-1]: x for x in ole.listdir(streams=True, storages=False)}
            suffix = p.suffix.lower()
            if suffix in ('.doc', '.dot'):
                word = ole.openstream(names.get('WordDocument', ['WordDocument'])).read(MAX_FILE)
                table_name = '1Table' if len(word) > 12 and _u16(word, 10) & 0x0200 else '0Table'
                table = ole.openstream(names.get(table_name, [table_name])).read(MAX_FILE)
                return _doc_text(word, table)
            if suffix == '.xls':
                stream_name = names.get('Workbook', names.get('Book', ['Workbook']))
                return _biff_text(ole.openstream(stream_name).read(MAX_FILE))
            if suffix == '.ppt':
                return _ppt_text(ole.openstream(names.get('PowerPoint Document', ['PowerPoint Document'])).read(MAX_FILE))
            if suffix == '.msg':
                return _msg_text(ole)
        finally:
            ole.close()
    except Exception:
        return None
    return None
