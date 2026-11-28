"""Query operators for file-name search: ext:pdf  size:>10mb  size:<1kb  date:2025  date:>2025-03-01  date:<2024-06."""
import calendar
import json
import ntpath
import re
import subprocess
import sys
import time

_UNITS = {'': 1, 'b': 1, 'kb': 1024, 'k': 1024, 'mb': 1 << 20, 'm': 1 << 20, 'gb': 1 << 30, 'g': 1 << 30}
_TOKEN = re.compile(r'(?<!\S)(ext|size|date|regex|path|folder|sort|limit):(\S+)', re.I)
_REGEX_CANDIDATES = 200000
_REGEX_SECONDS = 0.5


class FilterResults(list):
    """List-compatible results; truncated means the candidate scan was incomplete."""
    def __init__(self, hits=(), truncated=False, invalid_regex=False):
        super().__init__(hits)
        self.truncated = truncated
        self.invalid_regex = invalid_regex


def _wildcard(token, name):
    # Wildcards cover the whole nonempty name; plain terms are substrings.
    token, name = token.lower(), name.lower()
    if not name:
        return False
    if '*' not in token and '?' not in token:
        return token in name
    i = j = 0
    star = -1
    resume = 0
    while j < len(name):
        if i < len(token) and (token[i] == '?' or token[i] == name[j]):
            i += 1
            j += 1
        elif i < len(token) and token[i] == '*':
            star, resume = i, j
            i += 1
        elif star >= 0:
            resume += 1
            i, j = star + 1, resume
        else:
            return False
    return all(c == '*' for c in token[i:])


# Batch regex uses an isolated child even for patterns unsafe for single-hit
# evaluation. -c avoids multiprocessing's __main__ bootstrap contract.
_REGEX_WORKER = '''import json, re, sys
patterns, names = json.load(sys.stdin)
compiled = [re.compile(p, re.I) for p in patterns]
json.dump([i for i, name in enumerate(names)
           if all(p.search(name) for p in compiled)], sys.stdout)
'''


def _regex_indices(patterns, names, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return [], True
    child = None
    try:
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
        child = subprocess.Popen([sys.executable, '-I', '-c', _REGEX_WORKER],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, encoding='utf-8', **options)
        payload = json.dumps([patterns, names], ensure_ascii=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return [], True
        output, _ = child.communicate(payload, timeout=remaining)
        if child.returncode:
            return [], True
        return json.loads(output), False
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return [], True
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.communicate()


def _size(v):
    m = re.fullmatch(r'(\d+(?:[.,]\d+)?)\s*([a-z]*)', v.lower())
    if not m or m.group(2) not in _UNITS:
        return None
    return int(float(m.group(1).replace(',', '.')) * _UNITS[m.group(2)])


def _day(v, end=False):
    """'2025' / '2025-03' / '2025-03-01' -> epoch (start of period, or start of the next one when end=True)."""
    m = re.fullmatch(r'(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?', v)
    if not m:
        return None
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    try:
        if d:
            lo = calendar.timegm((y, int(mo), int(d), 0, 0, 0))
            return lo + 86400 if end else lo
        if mo:
            lo = calendar.timegm((y, int(mo), 1, 0, 0, 0))
            hi = calendar.timegm((y + (int(mo) == 12), int(mo) % 12 + 1, 1, 0, 0, 0))
        else:
            lo, hi = calendar.timegm((y, 1, 1, 0, 0, 0)), calendar.timegm((y + 1, 1, 1, 0, 0, 0))
    except (ValueError, OverflowError):
        return None
    return hi if end else lo


def _safe_single_regex(pattern):
    """Conservative subset: no groups, branches, backrefs or stacked repeats.

    At most one quantifier keeps unanchored search bounded on short names.
    Unsupported but valid patterns remain available to the isolated batch path.
    """
    repeats = 0
    in_class = False
    escaped = False
    for char in pattern:
        if escaped:
            if char.isdigit() or char in 'gk':
                return False
            escaped = False
        elif char == chr(92):
            escaped = True
        elif in_class:
            if char == ']':
                in_class = False
        elif char == '[':
            in_class = True
        elif char in '()|{}':
            return False
        elif char in '*+?':
            repeats += 1
            if repeats > 1:
                return False
    return not escaped and not in_class


def parse(query):
    """-> (clean_query, spec). spec is None when the query has no operators."""
    spec = {'ext': set(), 'min_size': None, 'max_size': None, 'min_t': None, 'max_t': None,
            'wildcards': [], 'regex': [], 'path': [], 'folder': [],
            'sort': None, 'limit': None, 'invalid_regex': False, 'truncated': False,
            'min_size_exclusive': False, 'max_size_exclusive': False}
    found = [False]

    def take(m):
        key, val = m.group(1).lower(), m.group(2)
        if key in ('regex', 'path', 'folder', 'sort', 'limit'):
            if key == 'regex':
                try:
                    if len(val) > 200:
                        raise ValueError('regex longer than 200 characters')
                    re.compile(val, re.I)
                except (re.error, ValueError, OverflowError, RecursionError):
                    spec['invalid_regex'] = True
                else:
                    spec['regex'].append(val)
            elif key in ('path', 'folder'):
                spec[key].append(val.lower())
            elif key == 'sort':
                if val.lower().lstrip('-') not in ('name', 'size', 'date', 'ext') or val.startswith('--'):
                    return m.group(0)
                spec[key] = val.lower()
            else:
                if not val.isascii() or not val.isdecimal() or len(val) > 4 or not 0 <= int(val) <= 1000:
                    return m.group(0)
                spec[key] = int(val) or None
            found[0] = True
            return ''
        if key == 'ext':
            spec['ext'] |= {e.lstrip('.').lower() for e in val.split(',') if e.strip('.')}
            found[0] = True
            return ''
        op = val[0] if val[0] in '<>' else ''
        body = val[1:] if op else val
        if key == 'size':
            n = _size(body)
            if n is None:
                return m.group(0)
            if op == '>':
                spec['min_size'] = n
                spec['min_size_exclusive'] = True
            elif op == '<':
                spec['max_size'] = n
                spec['max_size_exclusive'] = True
            else:
                spec['min_size'], spec['max_size'] = n, n
                spec['min_size_exclusive'] = spec['max_size_exclusive'] = False
        else:
            lo, hi = _day(body), _day(body, end=True)
            if lo is None:
                return m.group(0)
            if op == '>':
                spec['min_t'] = hi
            elif op == '<':
                spec['max_t'] = lo
            else:
                spec['min_t'], spec['max_t'] = lo, hi
        found[0] = True
        return ''

    words = _TOKEN.sub(take, query).split()
    plain = []
    for word in words:
        if '*' in word or '?' in word:
            spec['wildcards'].append(word.lower())
            found[0] = True
        else:
            plain.append(word)
    clean = ' '.join(plain)
    patterns = tuple(spec['regex'])
    spec['_regex_validated'] = patterns
    spec['_regex_safe'] = all(_safe_single_regex(p) for p in patterns)
    spec['_regex_compiled'] = tuple(re.compile(p, re.I) for p in patterns) if spec['_regex_safe'] else ()
    return clean, (spec if found[0] else None)


def matches(spec, hit):
    if spec is None:
        return True
    path = str(hit.get('path') or '')
    raw_name = str(hit.get('name') if hit.get('name') is not None else path)
    if not path and ntpath.isabs(raw_name):
        path = raw_name
    name = ntpath.basename(raw_name)
    if not all(_wildcard(token, name) for token in spec.get('wildcards', ())):
        return False
    if not all(text in path.lower() for text in spec.get('path', ())):
        return False
    folder = ntpath.basename(ntpath.dirname(path))
    if not all(text in folder.lower() for text in spec.get('folder', ())):
        return False
    if spec['ext']:
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if ext not in spec['ext']:
            return False
    size, mt = hit.get('size'), hit.get('mtime')
    if spec['min_size'] is not None or spec['max_size'] is not None:
        if size is None or size < 0:
            return False
        if spec['min_size'] is not None and size < spec['min_size']:
            return False
        if spec['max_size'] is not None and size > spec['max_size']:
            return False
        if spec.get('min_size_exclusive') and size == spec['min_size']:
            return False
        if spec.get('max_size_exclusive') and size == spec['max_size']:
            return False
    if spec['min_t'] is not None or spec['max_t'] is not None:
        if mt is None:
            return False
        if spec['min_t'] is not None and mt < spec['min_t']:
            return False
        if spec['max_t'] is not None and mt >= spec['max_t']:
            return False
    if spec.get('regex'):
        # Only parse-validated, cached patterns can enter the in-process path.
        # Mutated/foreign specs must use apply_spec for isolated evaluation.
        if (not spec.get('_regex_safe') or
                tuple(spec['regex']) != spec.get('_regex_validated') or len(name) > 1024):
            spec['truncated'] = True
            return False
        return all(pattern.search(name) for pattern in spec['_regex_compiled'])
    return True


def apply_spec(spec, hits, limit=None, deadline=None):
    """Filter all candidates, sort globally, then slice. Pass an unsliced iterable.

    Result attributes and spec['truncated'] report budget exhaustion. A timed-out
    regex batch is discarded, so every returned hit is a confirmed match.
    Server callers that pre-parse queries must call this before their top slice.
    """
    if spec is None:
        return FilterResults(list(hits) if limit is None or limit <= 0 else list(hits)[:limit])
    regex = spec.get('regex', ())
    if deadline is None and regex:
        deadline = time.monotonic() + _REGEX_SECONDS
    cheap = dict(spec, regex=[])
    selected = []
    truncated = False
    for examined, hit in enumerate(hits):
        if regex and (examined >= _REGEX_CANDIDATES or time.monotonic() >= deadline):
            truncated = True
            break
        if matches(cheap, hit):
            selected.append(hit)
    if regex and selected:
        indices, timed_out = _regex_indices(regex, [ntpath.basename(str(
            h.get('name') or h.get('path') or '')) for h in selected], deadline)
        selected = [selected[i] for i in indices]
        truncated = truncated or timed_out
    order = spec.get('sort')
    if order:
        field = order.lstrip('-')
        def sort_key(hit):
            name = ntpath.basename(str(hit.get('name') or hit.get('path') or '')).lower()
            if field in ('size', 'date'):
                value = hit.get('size' if field == 'size' else 'mtime')
                if field == 'size':
                    return (value is not None and value >= 0, value if value is not None else -1)
                unknown = value is None or value == -1
                return 0 if unknown else value
            if field == 'ext':
                return ntpath.splitext(name)[1], name
            return name
        selected.sort(key=sort_key, reverse=order.startswith('-'))
        if field == 'date':
            # Stable partition keeps unknowns last in both directions.
            selected.sort(key=lambda h: h.get('mtime') is None or h.get('mtime') == -1)
    requested = spec.get('limit')
    if requested is None or requested <= 0:
        requested = limit
    if requested is not None and requested > 0:
        selected = selected[:requested]
    spec['truncated'] = truncated
    return FilterResults(selected, truncated, spec.get('invalid_regex', False))
