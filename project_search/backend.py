from pathlib import Path
from collections import Counter
from contextlib import contextmanager
import os,json,hashlib,re,sqlite3,time,subprocess,shutil,threading
from project_search.resources import configure as _configure_resources, thread_count
_RESOURCE_PROFILE = _configure_resources()

# Embedder profiles. An index remembers the model it was built with ("model" in its meta);
# LOCALDOCS_MODEL=fast|quality|<hf id> forces a model (re-embeds on mismatch), otherwise an
# existing index keeps its own model and only NEW indexes get the default below.
# Measured on the 100-PDF math corpus (bench/model_results.jsonl), hybrid R@1 / R@10 / ru R@1,
# indexing time for 18k chunks on 8 CPU threads: fast 0.818/0.945/0.58 ~10 min,
# balanced 0.842/0.959/0.64 ~8 min, quality 0.877/0.973/0.72 ~28 min.
MODEL_PROFILES={'fast':'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'balanced':'intfloat/multilingual-e5-small',
                'quality':'intfloat/multilingual-e5-base'}
DEFAULT_PROFILE='balanced'
_ENV_MODEL=os.environ.get('LOCALDOCS_MODEL')
MODEL_NAME=MODEL_PROFILES.get(_ENV_MODEL,_ENV_MODEL) if _ENV_MODEL else MODEL_PROFILES[DEFAULT_PROFILE]


def _pin_model(store):
    """Model to use when (re)indexing: explicit env wins, else the index's own model, else default."""
    if _ENV_MODEL:
        return MODEL_NAME
    return store.model if (store.metadata and store.model) else MODEL_NAME
WRITE_DIR=Path(os.environ.get('LOCALDOCS_WRITE_DIR') or Path(__file__).resolve().parents[1])
BASE=WRITE_DIR/'indexes'
SKIP={'.git','.hg','.svn','.venv','venv','env','node_modules','__pycache__','.pytest_cache','.mypy_cache','dist','build','target','.secrets','world_backups'}
EXT={'.py','.js','.ts','.tsx','.jsx','.rs','.go','.java','.c','.h','.cpp','.hpp','.cs','.md','.txt','.rst','.toml','.yaml','.yml','.json','.sql','.html','.htm','.css','.ps1','.sh','.ini','.cfg','.xml','.vue','.svelte','.pdf','.docx','.xlsx','.pptx','.eml','.rtf','.csv','.tsv','.epub'}
try:
    from project_search.extractors_more import MORE_EXT as _MORE_EXT
    from project_search.extractors_ole import OLE_EXT as _OLE_EXT
    EXT |= set(_MORE_EXT) | set(_OLE_EXT)
except Exception:
    _MORE_EXT = _OLE_EXT = frozenset()
_FX_EXT = {'.xlsx','.pptx','.eml','.html','.htm','.rtf','.csv','.tsv','.epub'} | (set(_MORE_EXT) - {'.rst'}) | set(_OLE_EXT)
_IMG_EXT = {'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'}  # text via OCR; extractors.extract() returns None when OCR is disabled
_FX_EXT = _FX_EXT | _IMG_EXT
EXT = EXT | _IMG_EXT
SECRET={'credentials.json','secrets.json','api_keys.json','token.json','tokens.json','auth.json','localdocs_config.json','id_rsa','id_ed25519','.pem','.key','.pfx','.p12'}
SECRET_SUFFIX={'.pem', '.key', '.pfx', '.p12'}
# 512 KB used to skip every PDF/DOCX above it silently (half of a real PDF corpus).
MAX_BYTES = int(os.environ.get('LOCALDOCS_MAX_BYTES', str(200 * 1024 * 1024)))
_SKIP = [s for s in os.environ.get('LOCALDOCS_SKIP', '').split('|') if s]
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cpu_ready=False
_embedding_lock=threading.Lock()
_CACHE={}
_CACHE_LOCK=threading.Lock()


def _load_cached(directory):
    """Process-level index cache: reload from disk only when files changed."""
    if (directory/'publish.journal').exists():
        with locked(directory):pass  # recovers an interrupted publish
    try:
        sig=tuple((f, (directory/f).stat().st_mtime_ns, (directory/f).stat().st_size) for f in
                  ('index_meta.json','meta.sqlite','index.npy','lexical.sqlite','index.faiss')
                  if (directory/f).exists())
    except OSError:
        sig=None
    from project_search.neuro_vector_store import VectorStore
    with _CACHE_LOCK:
        hit=_CACHE.get(str(directory))
        if hit is not None and hit[0]==sig:
            return hit[1]
    store=VectorStore.load(str(directory/'index'))
    with _CACHE_LOCK:
        _CACHE[str(directory)]=(sig,store)
        if len(_CACHE)>32:
            _CACHE.pop(next(iter(_CACHE)))
    return store

def cpu_setup():
    global _cpu_ready
    if not _cpu_ready:
        import torch,faiss
        n=thread_count()
        torch.set_num_threads(n);faiss.omp_set_num_threads(n)
        _cpu_ready=True

def location(folder):
    root=Path(folder).expanduser().resolve(strict=True)
    if not root.is_dir():raise ValueError('Project path must be a folder')
    key=hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()[:20]
    return root,BASE/key

@contextmanager
def locked(directory):
    import msvcrt
    directory.mkdir(parents=True,exist_ok=True)
    lockfile = directory/'access.lock'
    if not lockfile.exists():
        lockfile.write_bytes(b'0')
    with open(lockfile,'r+b') as f:
        start=time.monotonic()
        while True:
            try:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                if time.monotonic()-start>60:raise RuntimeError('Index is busy; retry after the current operation')
                time.sleep(.2)
        try:
            recover_publish(directory)
            yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

LAST_SKIPS={}  # reason -> count for the latest files() walk (coverage report, like openai/math 300/719)

# Junk dirs beyond SKIP (SKIP entries keep reason skipped_dir regardless of
# LOCALDOCS_JUNK_FILTER); junk-only dirs are skipped only when the filter is on.
_JUNK_DIRS_EXTRA={'vendor','.cache'}
_JUNK_EXT={'.js','.json','.css','.map'}
_LOCK_NAMES={'package-lock.json','yarn.lock','pnpm-lock.yaml'}


def _junk_enabled():
    return os.environ.get('LOCALDOCS_JUNK_FILTER','1')!='0'


def _secret_scan_enabled():
    return os.environ.get('LOCALDOCS_SECRET_SCAN', '1') != '0'


_SECRET_HEAD_BYTES = 64 * 1024
_SECRET_PRIVKEY_RE = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----')
_SECRET_AKIA_RE = re.compile(r'AKIA[0-9A-Z]{16}')
_SECRET_ASSIGN_RE = re.compile(
    r'''(?i)\b(api_key|secret_key|access_token|password)\b\s*[:=]\s*['"]?([A-Za-z0-9_\-./+]{20,})['"]?''')

# Binary/extracted formats are not raw-scanned (they go through extractors/OCR);
# the content guard only reads extensions already treated as plain text.
_SECRET_SCAN_SKIP_EXT = frozenset({'.pdf', '.docx'}) | _FX_EXT


def _secret_reason(sample):
    """Return a short reason when the head text holds an obvious secret, else None."""
    if _SECRET_PRIVKEY_RE.search(sample):
        return 'private-key'
    if _SECRET_AKIA_RE.search(sample):
        return 'aws-key'
    if _SECRET_ASSIGN_RE.search(sample):
        return 'assigned-token'
    return None


def _file_has_secret(path):
    """Peek at most the first 64 KB; None = no secret (or scan not applicable)."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        with open(path, 'rb') as f:
            head = f.read(_SECRET_HEAD_BYTES)
    except OSError:
        return None
    if not head or b'\0' in head:
        return None  # binary: leave to the normal binary skip
    try:
        sample = head.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:
            sample = head.decode('cp1251')
        except UnicodeDecodeError:
            sample = head.decode('utf-8-sig', errors='replace')
    return _secret_reason(sample)


def _ignore_patterns(root):
    try:
        lines=(Path(root)/'.localdocsignore').read_text(encoding='utf-8',errors='replace').splitlines()
    except OSError:
        return []
    return [s for s in (ln.strip() for ln in lines) if s and not s.startswith('#')]


def _first_line_long(path,limit=2000):
    try:
        with open(path,'r',encoding='utf-8',errors='replace') as f:
            return len(f.readline(4096).rstrip('\r\n'))>limit
    except OSError:
        return False


def files(root):
    root=Path(root)
    LAST_SKIPS.clear()
    def _skip(why):LAST_SKIPS[why]=LAST_SKIPS.get(why,0)+1
    pats=_ignore_patterns(root)
    junk=_junk_enabled()
    app_private = [Path(ROOT_DIR)/name for name in ('runtime','models','logs','indexes')]+[WRITE_DIR/name for name in ('logs','indexes')]
    selected_config = Path(os.environ.get('LOCALDOCS_CONFIG', str(Path(ROOT_DIR)/'localdocs_config.json'))).resolve()
    rg=shutil.which('rg')
    if rg:
        r=subprocess.run([rg,'--files','--hidden','-g','!.git/**'],cwd=root,capture_output=True,encoding='utf8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if r.returncode not in (0,1):raise RuntimeError(r.stderr[:500])
        candidates=(root/x for x in r.stdout.splitlines())
    else:
        candidates=[]
        for base,dirs,names in os.walk(root,followlinks=False):
            keep=[]
            for d in dirs:
                if (Path(base)/d).is_symlink():continue
                low=d.lower()
                if low in SKIP:_skip('skipped_dir');continue
                if junk and low in _JUNK_DIRS_EXTRA:_skip('skipped_dir');continue
                keep.append(d)
            dirs[:]=keep
            candidates.extend(Path(base)/n for n in names)
    for p in sorted(candidates):
        rel=p.relative_to(root)
        posix=rel.as_posix()
        resolved=p.resolve()
        if resolved.is_relative_to(BASE):continue
        if resolved == selected_config:_skip('secret');continue
        if p.name.lower() == '.localdocs-fs-journal.json':_skip('app_private');continue  # legacy undo journal of older versions
        if any(resolved.is_relative_to(private) for private in app_private):_skip('app_private');continue
        if any(x.lower() in SKIP for x in rel.parts):_skip('skipped_dir');continue
        if junk and any(x.lower() in _JUNK_DIRS_EXTRA for x in rel.parts):_skip('skipped_dir');continue
        if pats and any(s in posix for s in pats):_skip('ignore_file');continue
        if p.suffix.lower() in SECRET_SUFFIX or p.name.lower() in SECRET or p.name.lower().startswith('.env'):_skip('secret');continue
        if junk and p.name.lower() in _LOCK_NAMES:_skip('lockfile');continue
        if junk and p.suffix.lower() in _JUNK_EXT and ('.min.' in p.name.lower() or _first_line_long(p)):_skip('minified');continue
        if p.suffix.lower() not in EXT:_skip('unsupported_extension');continue
        if _SKIP and any(s in posix for s in _SKIP):_skip('user_skip_list');continue
        try:
            if p.is_symlink() or not p.is_file() or not resolved.is_relative_to(root):_skip('symlink_or_outside');continue
            if p.stat().st_size>MAX_BYTES:_skip('too_large');continue
            if _secret_scan_enabled() and p.suffix.lower() not in _SECRET_SCAN_SKIP_EXT:
                _why = _file_has_secret(p)
                if _why:_skip('skipped_secret');continue
            if junk and p.suffix.lower() in _DATA_EXT and p.stat().st_size>150_000 and _looks_like_data(p):_skip('data_file');continue
        except OSError:continue
        yield p,posix

_DATA_EXT={'.txt','.log','.csv','.tsv','.dat','.jsonl','.ndjson','.json'}
def _looks_like_data(p):
    """Large machine-generated text (model dumps, BPE merges, numeric tables): no natural language to search."""
    try:
        with open(p,'rb') as f:sample=f.read(65536).decode('utf-8','ignore')
    except OSError:return False
    lines=[x for x in sample.splitlines() if x.strip()]
    if len(lines)<20:return False
    body=''.join(sample.split())
    if not body:return True
    letters=sum(c.isalpha() for c in body)/len(body)
    avg=sum(len(x) for x in lines)/len(lines)
    words=sample.split()
    avgw=sum(len(w) for w in words)/max(1,len(words))
    # prose: mostly letters, lines/words of normal length; dumps: digits/punctuation or tiny repetitive tokens
    return letters<0.5 or avg<12 or (avgw<3.2 and len(set(lines))<0.6*len(lines))

_MIN_CHUNK=150
def chunks(text,path,stamp):
    """Chunks of one file; tiny remnants (page tails, split leftovers) are merged into the previous chunk
    so they do not become noisy, unreadable search hits."""
    pending=None
    for m in _chunks_raw(text,path,stamp):
        if pending is None:
            pending=m;continue
        if len(m['text'])<_MIN_CHUNK and len(pending['text'])+len(m['text'])+1<=1000:
            pending['text']=pending['text']+chr(10)+m['text']
            pending['end_line']=max(pending['end_line'],m['end_line'])
            pending['key']=hashlib.sha256((path+'\0'+str(pending['start_line'])+'\0'+pending['text']).encode()).hexdigest()
            continue
        yield pending;pending=m
    if pending is not None:yield pending

def _chunks_raw(text,path,stamp):
    lines=text.splitlines();i=0
    while i<len(lines):
        j=i;size=0
        while j<len(lines) and j-i<35 and (size<650 or j==i):
            size+=len(lines[j])+1;j+=1
        content='\n'.join(lines[i:j])
        if content.strip():
            # Extremely long source lines are bounded, rather than silently
            # discarded or allowed to exceed the embedder's useful input.
            for offset in range(0,len(content),800):
                part=content[offset:offset+800]
                first_line=i+1+content[:offset].count('\n')
                key=hashlib.sha256((path+'\0'+str(first_line)+'\0'+str(offset)+'\0'+part).encode()).hexdigest()
                yield {'path':path,'start_line':first_line,'end_line':first_line+part.count('\n'),'text':part,'key':key,'mtime_ns':stamp}
        i=j if j-i<=2 else j-2

def _lex_norm(s):
    """FTS normalisation: lowercase + yo/ye (U+0451/U+0401 -> U+0435) so queries match."""
    return s.lower().replace('ё', 'е').replace('Ё', 'е')


def _hit_page(m):
    """Citation locator: PDF chunks carry 'page'; text chunks do not."""
    return m.get('page')


def _hit_locator(m, page):
    if page is not None:
        return f"p.{page}"
    return f"lines {m['start_line']}-{m['end_line']}"


# --- Chunks v2 (sandbox-only experiment; main backend untouched) -----------------
# Added metadata per chunk: title / section / table_header / prev_key / next_key /
# chunk_version. Ranking inputs are UNCHANGED: `key` is still hashed from the raw
# text only, the semantic embedding input stays `path + text`, and _hit_page /
# _hit_locator above are byte-identical to main. Only the FTS lexical copy may
# prepend table_header (table files only; the synthetic corpus has no tables, so
# its FTS index is bit-identical to main).
CHUNK_VERSION = 2


def _doc_title_section(full_text):
    """title = first non-empty line; section = Branch:/Reference: head line when
    present (synthetic corpus), else the second non-empty line (real docs: first
    heading-ish line)."""
    nonempty = [ln.strip() for ln in (full_text or '').splitlines() if ln.strip()]
    if not nonempty:
        return '', ''
    title = nonempty[0]
    section = ''
    if len(nonempty) > 1:
        head = nonempty[:4]
        br = next((ln for ln in head
                   if ln.startswith('Branch:') or ln.startswith('Reference:')), None)
        section = br if br is not None else nonempty[1]
    return title, section


def _is_header_row(line):
    """A serialised table row: docx tables join cells with ' | ', markdown uses
    pipes. Pure prose never qualifies."""
    s = (line or '').strip()
    if not s:
        return False
    if ' | ' in s:
        return True
    if s.startswith('|') and s.count('|') >= 2:
        return True
    return False


def _run_header(lines, idx_exclusive):
    """Header of the table run ending before line idx_exclusive: walk back over
    consecutive table rows (blank lines tolerated); the run's first row counts
    as a header only when it holds no digits (data rows carry codes/numbers).
    A header-less table (e.g. the reference single-row docx_table) yields None
    instead of a neighbouring data row."""
    j = idx_exclusive - 1
    while j >= 0 and (not lines[j].strip() or _is_header_row(lines[j])):
        j -= 1
    for k in range(j + 1, idx_exclusive):
        if k < 0 or k >= len(lines):
            continue
        if lines[k].strip():
            cand = lines[k].strip()
            if _is_header_row(cand) and not any(ch.isdigit() for ch in cand):
                return cand
            return None
    return None


def _table_header_before(page_text, char_off):
    """Page-level variant for 800-char pdf/docx slices: the table run is found
    in the full page lines, so mid-row cuts still resolve to the real header."""
    text = page_text or ''
    lines = text.splitlines()
    idx = text[:max(0, char_off)].count('\n')
    return _run_header(lines, idx)


def _table_header_for(start_line_1based, file_lines, chunk_text, suffix):
    """Line-based variant for plain-text/fx chunk lists (same run rule)."""
    lines = file_lines or []
    header = _run_header(lines, (start_line_1based or 1) - 1)
    if header is None and suffix in ('.csv', '.tsv', '.xlsx'):
        for i, ln in enumerate(file_lines or []):
            if ln.strip():
                if i + 1 < (start_line_1based or 1):
                    header = ln.strip()
                break
    if header and header in (chunk_text or ''):
        return None
    return header


def _finalize_file_chunks(file_chunks, full_text, suffix):
    """Attach v2 metadata to one file's chunks (text/keys untouched) and link
    prev/next neighbours in file order."""
    title, section = _doc_title_section(full_text)
    file_lines = (full_text or '').splitlines()
    for m in file_chunks:
        m['title'] = title
        m['section'] = section
        m['chunk_version'] = CHUNK_VERSION
        if m.get('table_header') is None:
            m['table_header'] = _table_header_for(
                m.get('start_line') or 1, file_lines, m.get('text') or '', suffix)
        if m.get('table_header') and m['table_header'] in (m.get('text') or ''):
            m['table_header'] = None
    for i, m in enumerate(file_chunks):
        m['prev_key'] = file_chunks[i - 1]['key'] if i > 0 else None
        m['next_key'] = file_chunks[i + 1]['key'] if i < len(file_chunks) - 1 else None
    return file_chunks


def _fts_text(m):
    """Lexical index copy: data-row chunks carry their table header as a prefix
    so header terms are findable; ranking-relevant stored text is unchanged."""
    h = m.get('table_header')
    t = m.get('text') or ''
    if h and h not in t:
        return h + '\n' + t
    return t


def _drop_legacy_v2(cum_meta, cum_vec, prog=None):
    """Migration: chunks without chunk_version==2 are dropped from the committed
    store so the walk below re-extracts them with v2 fields (vectors are
    re-embedded for those chunks only). Returns (meta, vec, dropped_count)."""
    keep = [i for i, m in enumerate(cum_meta)
            if m.get('chunk_version') == CHUNK_VERSION]
    dropped = len(cum_meta) - len(keep)
    if dropped:
        cum_meta = [cum_meta[i] for i in keep]
        if cum_vec is not None:
            cum_vec = cum_vec[keep]
        if prog is not None:
            try:
                prog(f"migrated chunks v2: re-extract {dropped} legacy chunks")
            except OSError:
                pass
    return cum_meta, cum_vec, dropped
# --- /Chunks v2 ---------------------------------------------------------------


def _line_spans(lines, text):
    """[(line, char_start, char_end)] for OCR lines inside one page's text.

    OCR page text is built as "\n".join(line texts), so offsets are derived
    from that join instead of guessing with text.find() (which assigned repeat
    lines to the first occurrence and drifted chunk geometry).
    """
    valid = [line for line in lines or []
             if isinstance(line, dict) and str(line.get('text') or '')]
    if not valid:
        return []
    joined = "\n".join(str(line.get('text') or '') for line in valid)
    spans = []
    if text == joined or text.startswith(joined):
        offset = 0
        for line in valid:
            value = str(line.get('text') or '')
            spans.append((line, offset, offset + len(value)))
            offset += len(value) + 1
        return spans
    # Fallback: sequential lookup only (never search from index 0 again).
    cursor = 0
    for line in valid:
        value = str(line.get('text') or '')
        start = text.find(value, cursor)
        if start < 0:
            continue
        spans.append((line, start, start + len(value)))
        cursor = start + len(value)
    return spans


def _attach_page_geometry(chunk, spans, page, start, end):
    """Attach only lines overlapping this chunk plus a compact page reference."""
    page_no = page.get('page')
    matched = [{'text': line.get('text'), 'box': line.get('box'),
                'conf': line.get('conf')}
               for line, line_start, line_end in spans
               if line_start < end and line_end > start]
    if matched:
        chunk['ocr_lines'] = matched
    chunk['ocr_page'] = {'page': page_no,
                         'source': page.get('source'),
                         'coord': 'render_px',
                         'line_count': len(spans)}
    if page.get('parts'):
        chunk['ocr_page']['parts'] = page['parts']
    if page.get('layout'):
        chunk['ocr_page']['layout'] = page.get('layout')
    if page.get('geom_chain'):
        chunk['ocr_page']['geom_chain'] = page.get('geom_chain')
    if page.get('orientation') is not None:
        chunk['ocr_page']['orientation'] = page.get('orientation')
    return chunk


def _image_page_geometry(result):
    """Page record for a single-image OCR result (page identity is always 1).

    Spread scans keep their parts: lines carry ``part`` L/R, and the compact
    page reference records each part's source box (never a silent cut).
    """
    parts = []
    for part in (result or {}).get('pages') or []:
        if isinstance(part, dict):
            parts.append({'part': part.get('part'), 'page': part.get('page', 1),
                          'box': part.get('box')})
    return {'page': 1, 'text': str((result or {}).get('text') or ''),
            'source': 'ocr:' + str((result or {}).get('source') or 'image'),
            'lines': (result or {}).get('lines') or [], 'parts': parts,
            'layout': (result or {}).get('layout'),
            'geom_chain': (result or {}).get('geom_chain')}


def lexical_db(path,metadata):
    conn=sqlite3.connect(path)
    try:
        conn.execute('CREATE VIRTUAL TABLE docs USING fts5(path,text)')
        conn.executemany('INSERT INTO docs(rowid,path,text) VALUES(?,?,?)',((i+1,m['path'],_lex_norm(_fts_text(m))) for i,m in enumerate(metadata)))
        conn.commit()
    finally:conn.close()

def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


EMBED_WORKERS = _env_int("LOCALDOCS_EMBED_WORKERS", 1)
EMBED_BATCH = _env_int("LOCALDOCS_EMBED_BATCH", 32)
SHARD_CHUNKS = _env_int("LOCALDOCS_SHARD_CHUNKS", 0)  # 0 = off; e.g. 100000
SEARCH_FANOUT = _env_int("LOCALDOCS_SEARCH_FANOUT", 8)
# Tuned on bench/tune_fusion.py (100 math PDFs, 292 queries): BM25 hits are far more
# precise than dense for term-bearing queries, so the lexical RRF weight is 5 (was 2).
LEX_WEIGHT = _env_int("LOCALDOCS_LEX_WEIGHT", 5)
CAND_DEPTH = _env_int("LOCALDOCS_CAND_DEPTH", 60)
EXACT_BOOST = _env_int("LOCALDOCS_EXACT_BOOST", 20)  # legacy blanket digit multiplier (kept for env override; targeted codes use IDENT_BONUS)
IDENT_BONUS = float(os.environ.get("LOCALDOCS_IDENT_BONUS", "0.6"))  # RRF-space bonus for exact code hits
# Identifiers: letter+digit mixes (BR105, KX50291, STUDY804000), long digit
# strings, hyphenated codes. Plain years/amounts are NOT identifiers.
_IDENT_RE = re.compile(r"[A-Za-z]*\d+[A-Za-z\d\-]*")


def _decode_plain(data):
    """BOM-aware plain-text decode for indexed files. None = binary/empty.

    UTF-8/16/32 BOM first, then utf-8, cp1251, cp1250. Raw NUL bytes are
    only accepted as BOM-less UTF-16 text with a plausible letter ratio;
    otherwise the file is honestly skipped (never errors="ignore" masking).
    """
    try:
        if not data:
            return None
        import codecs as _c
        if data.startswith((_c.BOM_UTF16_LE, _c.BOM_UTF16_BE)):
            try:
                _t = data.decode("utf-16")
                return _t if _t.strip() and "\x00" not in _t else None
            except Exception:
                return None
        if data.startswith((_c.BOM_UTF32_LE, _c.BOM_UTF32_BE)):
            try:
                _t = data.decode("utf-32")
                return _t if _t.strip() and "\x00" not in _t else None
            except Exception:
                return None
        if b"\x00" in data:
            for _enc in ("utf-16-le", "utf-16-be"):
                try:
                    _cand = data.decode(_enc)
                except Exception:
                    continue
                if "\x00" in _cand:
                    continue
                _s = _cand.strip().lstrip("\ufeff")
                if _s and sum(ch.isalpha() or ch.isspace() for ch in _s[:2000]) / max(1, len(_s[:2000])) > 0.5:
                    return _cand.lstrip("\ufeff")
            return None
        for _enc in ("utf-8-sig", "utf-8", "cp1251", "cp1250"):
            try:
                _t = data.decode(_enc)
                return _t if _t.strip() else None
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        try:
            _t = data.decode("utf-8-sig", errors="replace")
            return _t if _t.strip() else None
        except Exception:
            return None
    except Exception:
        return None


def _parse_query(query):
    """Full-query parse (no 12-token window).

    Returns dict: phrases (quoted), codes (identifiers anywhere in query),
    must (codes+phrases), excluded (-term), terms (all FTS tokens).
    Codes are letter+digit mixes with >=2 digits (BR105, KX50291,
    STUDY804000). Pure numbers - years, amounts, counts - stay plain terms
    and never become mandatory filters.
    """
    q = query or ""
    phrases = re.findall(r'"([^"]+)"', q)
    q_wo_quotes = re.sub(r'"[^"]+"', " ", q)
    excluded = set(m.lower() for m in re.findall(r"(?<!\w)-([A-Za-zА-Яа-яЁё0-9]{2,})", q_wo_quotes))
    raw_tokens = re.findall(r"[\w\-]+", q_wo_quotes, flags=re.UNICODE)
    codes = []
    for tok in raw_tokens:
        t = tok.strip("-_")
        if not t or t.lower() in excluded:
            continue
        has_digit = sum(ch.isdigit() for ch in t)
        has_alpha = any(ch.isalpha() for ch in t)
        # Letter+digit mix with at least 2 digits: BR105, KX50291.
        # Single-digit suffixes (word1, v2) and pure numbers are plain terms.
        if has_alpha and has_digit >= 2 and len(t) >= 4:
            codes.append(t)    # De-duplicate preserving order (case-insensitive)
    seen_c, uniq_codes = set(), []
    for c in codes:
        if c.lower() not in seen_c:
            seen_c.add(c.lower())
            uniq_codes.append(c)
    terms = [_lex_norm(t) for t in re.findall(r"\w+", q)]
    must_terms = [_lex_norm(c) for c in uniq_codes] + [_lex_norm(p) for p in phrases if p.strip()]
    return {"phrases": phrases, "codes": uniq_codes, "must": must_terms,
            "excluded": excluded, "terms": terms, "raw": q,
            "translit_terms": _translit_terms(q)}


_RU_TRANSLIT = [
    ("shch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("yu", "ю"), ("ya", "я"),
    ("jo", "ё"), ("yo", "ё"), ("ye", "е"), ("y", "ы"),
    ("a", "а"), ("b", "б"), ("v", "в"), ("g", "г"), ("d", "д"), ("e", "е"),
    ("z", "з"), ("i", "и"), ("j", "й"), ("k", "к"), ("l", "л"), ("m", "м"),
    ("n", "н"), ("o", "о"), ("p", "п"), ("r", "р"), ("s", "с"), ("t", "т"),
    ("u", "у"), ("f", "ф"), ("h", "х"), ("c", "ц"), ("'", "ь"),
]


def _translit_terms(query):
    """Latin-translit Russian -> Cyrillic variant tokens (expansion only).

    Only when the query has no Cyrillic at all (pure translit like
    'Kuda otnesti'). Returns lex-normalised Cyrillic tokens, never codes.
    """
    try:
        q = str(query or "")
        if not q or re.search(r"[\u0400-\u04FF]", q):
            return []
        if not re.search(r"[A-Za-z]", q):
            return []
        out_words = []
        for tok in re.findall(r"[A-Za-z']+", q):
            low = tok.lower()
            if len(low) < 3:
                continue
            s, i = "", 0
            while i < len(low):
                hit = None
                for lat, cyr in _RU_TRANSLIT:
                    if low.startswith(lat, i):
                        hit = (lat, cyr)
                        break
                if hit:
                    s += hit[1]
                    i += len(hit[0])
                else:
                    s += low[i]
                    i += 1
            if re.search(r"[\u0400-\u04FF]", s) and len(s) >= 3:
                out_words.append(_lex_norm(s))
        seen, uniq = set(), []
        for t in out_words:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq[:20]
    except Exception:
        return []


def route(query):
    """Deterministic route: exact / semantic / mixed. Search-time only.

    exact: only codes and/or quoted phrases (+ optional exclusions) - lexical
      only, semantic intentionally off (pure codes get blurred by dense).
    mixed: code/phrase + meaning terms - RRF both branches + mandatory code
      filter; code position (start/middle/end) does not matter.
    semantic: no codes/phrases - hybrid RRF as before.
    """
    p = _parse_query(query)
    codes, phrases = p["codes"], p["phrases"]
    raw = p["raw"] or ""
    q_wo = re.sub(r'"[^"]+"', " ", raw)
    tokens = [t.strip("-_") for t in re.findall(r"[\w\-]+", q_wo, flags=re.UNICODE)]
    tokens = [t for t in tokens if t]
    excl = {e.lower() for e in p["excluded"]}
    content = [t for t in tokens if t.lower() not in excl]
    cover = {t.lower() for t in content for c in codes if t.lower() == c.lower()}
    n_phrase_tokens = sum(len(re.findall(r"\w+", ph)) for ph in phrases)
    if (codes or phrases) and not phrases and len(content) <= max(1, len(cover)):
        # Pure code query (single or few codes, nothing else meaningful).
        if len(content) <= max(1, len(cover)):
            only_codes = all(t.lower() in cover for t in content)
            if only_codes:
                return "exact"
    if phrases and not codes and len(content) <= n_phrase_tokens:
        return "exact"
    if codes or phrases:
        return "mixed"
    return "semantic"


_OCR_MAP = str.maketrans({"O": "0", "o": "0", "I": "1", "i": "1", "l": "1", "L": "1"})


def ocr_variants(code):
    """Homoglyph variants (0/O, 1/I/l). Expansion only, never a filter.

    At most 6 variants, rank-only with half weight and no IDENT_BONUS.
    """
    base = code.translate(_OCR_MAP)
    out = []
    if base.lower() != code.lower():
        out.append(base)
    for i, ch in enumerate(code):
        if ch in "0Oo":
            for r in ("0", "O", "o"):
                v = code[:i] + r + code[i + 1:]
                if v.lower() != code.lower() and v.lower() != base.lower():
                    out.append(v)
        elif ch in "1IiLl":
            for r in ("1", "I", "l"):
                v = code[:i] + r + code[i + 1:]
                if v.lower() != code.lower() and v.lower() != base.lower():
                    out.append(v)
    seen, uniq = set(), []
    for v in out:
        if v.lower() not in seen:
            seen.add(v.lower())
            uniq.append(v)
    return uniq[:6]


def _embed_worker_init():
    import torch as _t
    _t.set_num_threads(1)
    _t.set_num_interop_threads(1)


def _embed_worker_job(payload):
    from project_search.neuro_vector_store import embed_texts as _et
    idx, texts, model_name = payload
    return idx, _et(texts, "passage", model_name)


def embed_parallel(indexed_texts, workers=1, batch=32, model_name=None):
    """indexed_texts: list of (position, text). Returns dict position -> vector.
    workers=1 keeps the old sequential path (safest). >1 uses process pool,
    one model copy per worker, torch pinned to 1 thread each."""
    from project_search.neuro_vector_store import embed_texts as _et
    if workers <= 1 or len(indexed_texts) < workers * 2:
        out = {}
        with _embedding_lock:
            for start in range(0, len(indexed_texts), batch):
                part = indexed_texts[start:start + batch]
                enc = _et([t for _, t in part], "passage", model_name)
                for (pos, _), v in zip(part, enc):
                    out[pos] = v
        return out
    import concurrent.futures as _cf
    jobs = [indexed_texts[i::workers] for i in range(workers)]
    jobs = [(n, [t for _, t in j], model_name) for n, j in enumerate(jobs) if j]
    out = {}
    try:
        with _cf.ProcessPoolExecutor(max_workers=workers, initializer=_embed_worker_init) as ex:
            for n, enc in ex.map(_embed_worker_job, jobs):
                for (pos, _), v in zip([p for p in indexed_texts[n::workers]], enc):
                    out[pos] = v
    except Exception:
        with _embedding_lock:
            from project_search.neuro_vector_store import embed_texts as _et2
            for start in range(0, len(indexed_texts), batch):
                part = indexed_texts[start:start + batch]
                enc = _et2([t for _, t in part], "passage", model_name)
                for (pos, _), v in zip(part, enc):
                    out[pos] = v
    return out


class IndexLimitError(ValueError):
    """A configured indexing bound was reached; never report a truncated index as complete."""


def index_project_chunked(folder, group_files=400, max_files=0, max_chunks=0, should_stop=None):
    """Crash-safe indexing: commits cumulative store every group_files new chunks.
    Same index dir/format as index_project (compatible, shares reuse map).
    A rerun after a crash reuses committed vectors, re-extracts text only.
    should_stop: optional callable returning True to cooperatively stop at the
    next group boundary; on stop a partial commit is flushed and a report
    with stopped=True is returned. Default None preserves old behavior."""
    # Probe entitlement: index_project_chunked
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        from probe_text_index import index as _free_index
        return _free_index(folder, should_stop)
    import datetime as _dt
    max_files=max(0, _env_int('LOCALDOCS_MAX_FILES', max_files))
    max_chunks=max(0, _env_int('LOCALDOCS_MAX_CHUNKS', max_chunks))
    root, directory = location(folder)
    from project_search.neuro_vector_store import VectorStore
    import numpy as np
    cpu_setup()

    def prog(msg):
        try:
            with open(os.path.join(str(WRITE_DIR), 'logs', 'index_progress.log'), 'a', encoding='utf-8') as f:
                f.write(f"{_dt.datetime.now().strftime('%H:%M:%S')} [chunked] {msg}\n")
        except OSError:
            pass

    with locked(directory):
        store = VectorStore.load(str(directory / 'index'))
        mdl = _pin_model(store)
        if store.metadata and (store.model or MODEL_NAME) != mdl:
            store = VectorStore(model=mdl)  # explicit model change: old vectors are unusable
        cum_meta = store.metadata.to_list()
        cum_vec = store.vectors
        have = {m['key'] for m in cum_meta}
    del store  # release the original FAISS copy and vectors before growing the cumulative store
    # Chunks-v2 migration (same pattern as the geom_version backfill below):
    # committed chunks without chunk_version==2 are dropped here, so the walk
    # re-extracts them with v2 fields. LOCALDOCS_REEXTRACT=1 forces the same
    # for every file via the reuse predicate.
    cum_meta, cum_vec, _dropped_v2 = _drop_legacy_v2(cum_meta, cum_vec, prog)
    if _dropped_v2:
        have = {m['key'] for m in cum_meta}
    unchanged = {}
    for m in cum_meta:
        entry = unchanged.setdefault(m['path'], {'mtime': m.get('mtime_ns'), 'keys': []})
        if entry['mtime'] != m.get('mtime_ns'):
            entry['mtime'] = None
        _suffix = Path(m['path']).suffix.lower()
        if ((_suffix in ('.pdf', '.docx') or _suffix in _IMG_EXT)
                and not m.get('geom_version')):
            entry['mtime'] = None  # one-time re-extract to backfill geometry
        entry['keys'].append(m['key'])
    reused_files = 0
    skipped = 0
    needs_ocr = 0
    count = 0
    group = []
    total_new = 0
    seen = {}
    seen_keys = set()

    def add_chunk(m):
        seen_keys.add(m['key'])
        if m['key'] in have:
            return
        if max_chunks and len(cum_meta)+len(group)>=max_chunks:
            raise IndexLimitError('Configured fragment limit reached')
        group.append(m)
        have.add(m['key'])

    def flush(force=False):
        nonlocal cum_meta, cum_vec, total_new
        if not group and not force:
            return
        if group:
            texts = [(m['key'], m['path'] + '\n' + m['text']) for m in group]
            got = embed_parallel(texts, EMBED_WORKERS, EMBED_BATCH, mdl)
            new_vecs = np.array([got[m['key']] for m in group], dtype='float32')
            cum_vec = new_vecs if cum_vec is None else np.vstack([cum_vec, new_vecs])
            cum_meta.extend(group)
            total_new += len(group)
        st = VectorStore(model=mdl)
        if cum_vec is not None:
            st._ensure_index(cum_vec.shape[1], len(cum_meta))
            if st.index_type == "ivfpq" and len(cum_vec):
                st._build_ivfpq(cum_vec)
            elif len(cum_vec):
                st._index.add(cum_vec)
        st.vectors = cum_vec
        st.metadata = cum_meta
        with locked(directory):
            st.save(str(directory / 'pending'))
            lex = directory / 'pending.sqlite'
            if lex.exists():
                lex.unlink()
            lexical_db(str(lex), cum_meta)
            _publish(directory)
            if not cum_meta:
                for name in ('index.npy', 'index.faiss'):
                    if (directory / name).exists():
                        (directory / name).unlink()
            if (directory / 'shards.json').exists():
                (directory / 'shards.json').unlink()
            (directory / 'project.json').write_text(json.dumps(
                {'folder': str(root), 'files': count, 'chunks': len(cum_meta),
                 'partial': True, 'model': mdl,
                 'index_directory': str(directory)}, ensure_ascii=False, indent=2), encoding='utf8')
        group.clear()
        prog(f"committed chunks={len(cum_meta)}")

    def _asked():
        try:
            return bool(should_stop()) if callable(should_stop) else False
        except Exception:
            return False

    def _stopped_report():
        flush()
        prog("stopped by request")
        return {'folder': str(root), 'files': count, 'chunks': len(cum_meta),
                'partial': True, 'new_embeddings': total_new, 'stopped': True,
                'mode': 'chunked', 'model': mdl,
                'index_directory': str(directory)}

    for p, rel in files(root):
        count += 1
        if max_files and count > max_files:
            raise IndexLimitError('Configured file limit reached')
        try:
            seen[rel] = p.stat().st_mtime_ns
        except OSError:
            pass
        prior = unchanged.get(rel)
        if (prior and prior['mtime'] is not None and
                prior['mtime'] == seen.get(rel) and
                p.suffix.lower() != '.rtf' and
                os.environ.get('LOCALDOCS_REEXTRACT', '0') != '1'):
            seen_keys.update(prior['keys'])
            reused_files += 1
            if count % 25 == 1:
                prog(f"file {count}: reused {rel[:90]}")
            continue
        if p.suffix.lower() in ('.pdf', '.docx'):
            try:
                from project_search.localdocs import extract_pages as _ld
                pages = _ld(p)
                empty = sum(1 for page in pages if not (page.get('text') or '').strip())
                _doc_full = '\n\n'.join(page.get('text') or '' for page in pages)
                _fchunks = []
                for page in pages:
                    pg = page.get('page')
                    pt = page.get('text') or ''
                    spans = _line_spans(page.get('lines'), pt)
                    for off in range(0, max(len(pt), 1), 800):
                        part = pt[off:off + 800]
                        if part.strip():
                            key = hashlib.sha256((rel + '\0' + str(pg) + '\0' + str(off) + '\0' + part).encode()).hexdigest()
                            chunk = {'path': rel, 'start_line': pg, 'end_line': pg,
                                     'page': pg, 'text': part, 'key': key,
                                     'mtime_ns': p.stat().st_mtime_ns, 'geom_version': 1,
                                     'table_header': _table_header_before(pt, off)}
                            if spans:
                                _attach_page_geometry(chunk, spans, page, off, off + len(part))
                            _fchunks.append(chunk)
                _finalize_file_chunks(_fchunks, _doc_full, p.suffix.lower())
                for _m in _fchunks:
                    add_chunk(_m)
                if empty == len(pages):
                    skipped += 1
                    needs_ocr += 1
            except IndexLimitError:
                raise
            except Exception:
                skipped += 1
            if count % 25 == 1:
                prog(f"file {count}: {rel[:90]}")
            if len(group) >= group_files:
                flush()
                if _asked():
                    return _stopped_report()
            continue
        if p.suffix.lower() in _IMG_EXT:
            try:
                from project_search.extractors import _image_ocr_result as _img_ocr
                _img = _img_ocr(p)
            except Exception:
                _img = None
            if not _img or not str(_img.get('text') or '').strip():
                skipped += 1
            else:
                _page = _image_page_geometry(_img)
                _spans = _line_spans(_page['lines'], _page['text'])
                _cursor = 0
                _fchunks = []
                for m in chunks(_page['text'], rel, p.stat().st_mtime_ns):
                    m['geom_version'] = 1
                    if _spans:
                        _start = _page['text'].find(m['text'], _cursor)
                        if _start < 0:
                            _start = _page['text'].find(m['text'])
                        if _start >= 0:
                            _cursor = _start + len(m['text'])
                            _attach_page_geometry(m, _spans, _page, _start, _start + len(m['text']))
                    _fchunks.append(m)
                _finalize_file_chunks(_fchunks, _page['text'], p.suffix.lower())
                for _m in _fchunks:
                    add_chunk(_m)
            if count % 25 == 1:
                prog(f"file {count}: {rel[:90]}")
            if len(group) >= group_files:
                flush()
                if _asked():
                    return _stopped_report()
            continue
        if p.suffix.lower() in _FX_EXT:
            try:
                from project_search.extractors import extract as _fx_extract
                _ftext = _fx_extract(p)
            except Exception:
                _ftext = None
            if not _ftext or not _ftext.strip():
                skipped += 1
            else:
                _fchunks = _finalize_file_chunks(
                    list(chunks(_ftext, rel, p.stat().st_mtime_ns)), _ftext, p.suffix.lower())
                for _m in _fchunks:
                    add_chunk(_m)
            if count % 25 == 1:
                prog(f"file {count}: {rel[:90]}")
            if len(group) >= group_files:
                flush()
                if _asked():
                    return _stopped_report()
            continue
        try:
            data = p.read_bytes()
        except OSError:
            skipped += 1
            continue
        text = _decode_plain(data)
        if not text:
            skipped += 1
            continue
        _fchunks = _finalize_file_chunks(
            list(chunks(text, rel, p.stat().st_mtime_ns)), text, p.suffix.lower())
        for _m in _fchunks:
            add_chunk(_m)
        if count % 25 == 1:
            prog(f"file {count}: {rel[:90]}")
        if len(group) >= group_files:
            flush()
            if _asked():
                return _stopped_report()
    if _asked():
        return _stopped_report()
    keep_idx = [i for i, m in enumerate(cum_meta)
                if m['path'] in seen and m['key'] in seen_keys]
    pruned = len(cum_meta) - len(keep_idx)
    if pruned:
        cum_meta = [cum_meta[i] for i in keep_idx]
        if cum_vec is not None:
            cum_vec = cum_vec[keep_idx]
        prog(f"pruned zombies={pruned}")
    refreshed = False
    for m in cum_meta:
        if m['mtime_ns'] != seen[m['path']]:
            m['mtime_ns'] = seen[m['path']]
            refreshed = True
    # Pruning must be published even when no new chunks need embedding, including
    # deletion of the last source file. A legacy no-op rerun also migrates format.
    flush(force=bool(pruned or refreshed or not (directory / 'meta.sqlite').exists()))
    report = {'folder': str(root), 'files': count, 'chunks': len(cum_meta), 'partial': False,
              'new_embeddings': total_new, 'pruned': pruned, 'skipped_binary_or_encoding': skipped,
              'needs_ocr': needs_ocr, 'mode': 'chunked', 'model': mdl, 'skipped_by_reason': dict(LAST_SKIPS),
              'cpu_threads': thread_count(), 'reused_files': reused_files,
              'index_type': json.loads((directory / 'index_meta.json').read_text(encoding='utf8')).get('index_type', 'flat'),
              'index_directory': str(directory)}
    (directory / 'project.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    return report


def index_project(folder, max_files=10000, max_chunks=30000):
    """Create/update a CPU-only code/document index for one user-selected folder; reuse unchanged vectors. Does not edit source files."""
    # Probe entitlement: index_project
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        from probe_text_index import index as _free_index
        return _free_index(folder, None)
    max_files = _env_int("LOCALDOCS_MAX_FILES", max_files)
    max_chunks = _env_int("LOCALDOCS_MAX_CHUNKS", max_chunks)
    if max_files <= 0:
        max_files = float("inf")
    if max_chunks <= 0:
        max_chunks = float("inf")
    root, directory = location(folder)
    from project_search.neuro_vector_store import VectorStore,embed_texts
    import numpy as np
    cpu_setup()
    with locked(directory):
        previous=VectorStore.load(str(directory/'index'))
        mdl=_pin_model(previous)
        if previous.metadata and (previous.model or MODEL_NAME)!=mdl:
            previous=VectorStore(model=mdl)  # explicit model change: re-embed everything
        previous.metadata = previous.metadata.to_list()
        # Chunks-v2 migration (same as index_project_chunked): committed chunks
        # without chunk_version==2 are dropped so the walk below re-extracts
        # them with v2 fields. LOCALDOCS_REEXTRACT=1 is honoured by the
        # chunked path; here legacy chunks always migrate once.
        try:
            previous.metadata, previous.vectors, _ = _drop_legacy_v2(
                previous.metadata, previous.vectors)
        except Exception:
            pass
        old={m['key']:i for i,m in enumerate(previous.metadata)}
        old_vectors=previous.vectors
        if old_vectors is None and previous._index is not None and previous.size:
            try:old_vectors=previous._index.reconstruct_n(0,previous.size)
            except Exception:old_vectors=None
        metadata=[];count=0;skipped=0;needs_ocr=0
        import datetime as _dt
        prog = os.path.join(str(WRITE_DIR), 'logs', 'index_progress.log')
        def _prog(msg):
            try:
                with open(os.path.normpath(prog), 'a', encoding='utf-8') as f:
                    f.write(f"{_dt.datetime.now().strftime('%H:%M:%S')} {msg}\n")
            except OSError:
                pass
        for p,rel in files(root):
            count+=1
            if count % 25 == 1:
                _prog(f"file {count}: {rel[:90]}")
            if count>max_files:raise ValueError('File limit reached; choose a narrower project folder')
            if p.suffix.lower() in ('.pdf', '.docx'):
                try:
                    from project_search.localdocs import extract_pages as _ld_extract
                    _pages = _ld_extract(p)
                    _empty = sum(1 for page in _pages if not (page.get('text') or '').strip())
                    _doc_full = '\n\n'.join(page.get('text') or '' for page in _pages)
                    _fchunks = []
                    for page in _pages:
                        _pg = page.get('page')
                        _pt = page.get('text') or ''
                        _spans = _line_spans(page.get('lines'), _pt)
                        for _off in range(0, max(len(_pt), 1), 800):
                            _part = _pt[_off:_off + 800]
                            if _part.strip():
                                _key = hashlib.sha256((rel + '\0' + str(_pg) + '\0' + str(_off) + '\0' + _part).encode()).hexdigest()
                                _chunk = {'path': rel, 'start_line': _pg, 'end_line': _pg, 'page': _pg, 'text': _part, 'key': _key, 'mtime_ns': p.stat().st_mtime_ns, 'geom_version': 1,
                                          'table_header': _table_header_before(_pt, _off)}
                                if _spans:
                                    _attach_page_geometry(_chunk, _spans, page, _off, _off + len(_part))
                                _fchunks.append(_chunk)
                    _finalize_file_chunks(_fchunks, _doc_full, p.suffix.lower())
                    metadata.extend(_fchunks)
                    if _empty == len(_pages):
                        skipped += 1
                        needs_ocr += 1
                    continue
                except Exception:
                    skipped += 1
                    continue
            if p.suffix.lower() in _IMG_EXT:
                try:
                    from project_search.extractors import _image_ocr_result as _img_ocr
                    _img = _img_ocr(p)
                except Exception:
                    _img = None
                if not _img or not str(_img.get('text') or '').strip():
                    skipped += 1
                    continue
                _page = _image_page_geometry(_img)
                _spans = _line_spans(_page['lines'], _page['text'])
                _cursor = 0
                _fchunks = []
                for _m in chunks(_page['text'], rel, p.stat().st_mtime_ns):
                    _m['geom_version'] = 1
                    if _spans:
                        _start = _page['text'].find(_m['text'], _cursor)
                        if _start < 0:
                            _start = _page['text'].find(_m['text'])
                        if _start >= 0:
                            _cursor = _start + len(_m['text'])
                            _attach_page_geometry(_m, _spans, _page, _start, _start + len(_m['text']))
                    _fchunks.append(_m)
                _finalize_file_chunks(_fchunks, _page['text'], p.suffix.lower())
                metadata.extend(_fchunks)
                continue
            if p.suffix.lower() in _FX_EXT:
                try:
                    from project_search.extractors import extract as _fx_extract
                    _ftext = _fx_extract(p)
                except Exception:
                    _ftext = None
                if not _ftext or not _ftext.strip():
                    skipped += 1
                else:
                    metadata.extend(_finalize_file_chunks(
                        list(chunks(_ftext, rel, p.stat().st_mtime_ns)), _ftext, p.suffix.lower()))
                    if len(metadata) > max_chunks:
                        raise ValueError('Chunk limit reached; choose a narrower folder')
                continue
            data=p.read_bytes()
            text=_decode_plain(data)
            if not text:skipped+=1;continue
            metadata.extend(_finalize_file_chunks(
                list(chunks(text,rel,p.stat().st_mtime_ns)), text, p.suffix.lower()))
            if len(metadata)>max_chunks:raise ValueError('Chunk limit reached; choose a narrower folder')
        new_positions=[i for i,m in enumerate(metadata) if m['key'] not in old]
        newkeys={m['key'] for m in metadata}
        deleted=[k for k in old if k not in newkeys]
        mode='rebuild'
        if (previous.index_type == 'hnsw' and not deleted and previous.vectors is not None
                and new_positions and previous._index is not None):
            parts = [(i, metadata[i]['path'] + '\n' + metadata[i]['text']) for i in new_positions]
            got = embed_parallel(parts, EMBED_WORKERS, EMBED_BATCH, mdl)
            order = [new_positions[n] for n in sorted(range(len(new_positions)),
                                                      key=lambda n: new_positions[n])]
            newvecs = np.array([got[i] for i in order], dtype='float32')
            previous._index.add(newvecs)
            previous.vectors = np.vstack([previous.vectors, newvecs])
            current_by_key = {m['key']: m for m in metadata}
            previous.metadata = ([current_by_key[m['key']] for m in previous.metadata if m['key'] in newkeys]
                                 + [m for m in metadata if m['key'] not in old])
            metadata = previous.metadata
            prefix = str(directory / 'pending')
            previous.save(prefix)
            mode = 'incremental'
            idx_type = previous.index_type
        else:
            parts=[(i,metadata[i]['path']+'\n'+metadata[i]['text'])
                   for i in (new_positions if old_vectors is not None
                             else list(range(len(metadata))))]
            got=embed_parallel(parts,EMBED_WORKERS,EMBED_BATCH,mdl)
            dim=len(next(iter(got.values()))) if got else (old_vectors.shape[1] if old_vectors is not None else 384)
            vectors=np.empty((len(metadata),dim),dtype='float32')
            for i,m in enumerate(metadata):
                if m['key'] in old and old_vectors is not None:vectors[i]=old_vectors[old[m['key']]]
            for i,v in got.items():vectors[i]=v
            store=VectorStore(model=mdl)
            if len(metadata):
                store._ensure_index(dim, len(metadata))
                if store.index_type == "ivfpq":
                    store._build_ivfpq(vectors)
                else:
                    store._index.add(vectors)
                store.vectors = vectors
            store.metadata=metadata
            prefix=str(directory/'pending')
            store.save(prefix)
            idx_type = store.index_type
            if SHARD_CHUNKS > 0 and len(metadata) > SHARD_CHUNKS:
                idx_type = _save_shards(directory, metadata, vectors, mdl)
                mode = 'sharded'
        lex=directory/'pending.sqlite'
        if lex.exists():lex.unlink()
        lexical_db(lex,metadata)
        _publish(directory)
        if not metadata and (directory/'index.faiss').exists():(directory/'index.faiss').unlink()
        if not metadata and (directory/'index.npy').exists():(directory/'index.npy').unlink()
        if mode != 'sharded' and (directory/'shards.json').exists():
            (directory/'shards.json').unlink()
        report={'folder':str(root),'files':count,'chunks':len(metadata),'new_embeddings':len(new_positions),'reused_embeddings':len(metadata)-len(new_positions),'skipped_binary_or_encoding':skipped,'needs_ocr':needs_ocr,'skipped_by_reason':dict(LAST_SKIPS),'mode':mode,'index_type':idx_type,'model':mdl,'device':'cpu','cpu_threads':os.cpu_count(),'index_directory':str(directory)}
        (directory/'project.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        return report

_PUBLISH_PAIRS = (('pending_meta.json', 'index_meta.json'), ('pending_meta.sqlite', 'meta.sqlite'),
                  ('pending.npy', 'index.npy'),
                  ('pending.sqlite', 'lexical.sqlite'), ('pending.faiss', 'index.faiss'))


def recover_publish(directory):
    """Finish an interrupted _publish (roll forward). Call with the dir lock held."""
    journal = directory / 'publish.journal'
    if not journal.exists():
        return False
    for a, b in _PUBLISH_PAIRS:
        if (directory / a).exists():
            _replace_retry(directory / a, directory / b)
    journal.unlink()
    return True


def _publish(directory):
    """Journaled multi-file swap: all pending files are complete before the
    journal is written; a crash mid-rename is rolled forward by recover_publish,
    so meta/vectors/FTS/faiss never stay mixed from two generations."""
    journal = directory / 'publish.journal'
    todo = [a for a, _ in _PUBLISH_PAIRS if (directory / a).exists()]
    tmp = directory / 'publish.journal.tmp'
    tmp.write_text(json.dumps(todo), encoding='utf8')
    os.replace(tmp, journal)
    recover_publish(directory)


def _replace_retry(src, dst, tries=5):
    import time as _t
    for n in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if n == tries - 1:
                raise
            _t.sleep(0.5 * (n + 1))


def _save_shards(directory, metadata, vectors, model=None):
    """Split a big flat index into exact per-shard indexes (SHARD_CHUNKS each).
    Shards stay FlatIP: exact search, parallel fan-out at query time."""
    from project_search.neuro_vector_store import VectorStore
    import numpy as _np
    names = []
    if hasattr(metadata, 'to_list'):
        metadata = metadata.to_list()
    for n, start in enumerate(range(0, len(metadata), SHARD_CHUNKS)):
        part_meta = metadata[start:start + SHARD_CHUNKS]
        part_vec = vectors[start:start + SHARD_CHUNKS]
        sub = directory / f"shard_{n:03d}"
        sub.mkdir(parents=True, exist_ok=True)
        st = VectorStore(model=model)
        st._dim = part_vec.shape[1]
        st.index_type = "flat"
        import faiss as _faiss
        st._index = _faiss.IndexFlatIP(st._dim)
        st._index.add(part_vec)
        st.vectors = part_vec
        st.metadata = part_meta
        st.save(str(sub / 'pending'))
        lex = sub / 'pending.sqlite'
        if lex.exists():
            lex.unlink()
        lexical_db(str(lex), part_meta)
        _publish(sub)
        names.append(sub.name)
        (sub / 'project.json').write_text(json.dumps(
            {"folder": str(sub), "files": 0, "chunks": len(part_meta), "shard_of": str(directory)},
            ensure_ascii=False), encoding='utf8')
    manifest = {"shards": names, "total_chunks": len(metadata), "shard_size": SHARD_CHUNKS}
    (directory / 'shards.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding='utf8')
    return "flat-sharded"


def search_project(folder,query,k=6):
    """Search an indexed folder by meaning and exact names. Return bounded excerpts, original paths, lines and stale flags."""
    # Probe entitlement: search_project
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        from probe_text_index import search as _free_search
        return _free_search(folder, query, k)
    root,directory=location(folder)
    if not (directory/'project.json').exists():raise ValueError('Project is not indexed; call index_project first')
    if not query.strip():return []
    from project_search.neuro_vector_store import VectorStore
    cpu_setup();k=max(1,min(k,100))
    try:
        from project_search import rerank as _rerank
    except ImportError:
        _rerank=None
    _use_rerank=bool(_rerank is not None and _rerank.enabled())
    _rn=k
    if _use_rerank:
        try:_rn=max(1,int(os.environ.get('LOCALDOCS_RERANK_N','40')))
        except ValueError:_rn=20
    fetch=max(k,_rn) if _use_rerank else k
    manifest = directory / 'shards.json'
    if manifest.exists():
        import concurrent.futures as _cf
        try:
            names = json.loads(manifest.read_text(encoding='utf8'))['shards']
        except (OSError, ValueError, KeyError):
            names = []
        if names:
            allhits = []
            with _cf.ThreadPoolExecutor(max_workers=min(SEARCH_FANOUT, len(names))) as ex:
                futs = {ex.submit(_search_dir, root, directory / n, query, fetch): n for n in names}
                for fu in _cf.as_completed(futs):
                    try:
                        allhits.extend(fu.result())
                    except (ValueError, RuntimeError):
                        pass
            merged, per_file = [], Counter()
            for h in sorted(allhits, key=lambda x: x['score'], reverse=True):
                key = h['file']
                if per_file[key] >= 4:
                    continue
                per_file[key] += 1
                merged.append(h)
                if len(merged) >= fetch:
                    break
            if _use_rerank:
                merged=_rerank.rerank(query, merged, _rn)[:k]
            return merged
    found=_search_dir(root, directory, query, fetch)
    if _use_rerank:
        found=_rerank.rerank(query, found, _rn)[:k]
    else:
        found=found[:k]
    return found


def _search_dir(root, directory, query, k):
    """Search one materialized index dir; root resolves relative metadata paths.

    Deterministic router: exact (codes/phrases only) = lexical only;
    semantic (no codes) = hybrid RRF; mixed = RRF + mandatory code filter.
    OCR homoglyph variants expand candidates (half weight, no bonus, never
    a filter) and only when exact codes find nothing.
    """
    from project_search.neuro_vector_store import VectorStore
    with locked(directory):
        store=_load_cached(directory)
        _route=route(query)
        if _route == "exact":
            semantic=[]
        else:
            with _embedding_lock:semantic=store.search(query,max(CAND_DEPTH,k*4))
        ranks={m['key']:{'metadata':m,'score':1/(60+rank),'semantic_score':score} for rank,(m,score) in enumerate(semantic,1)}
        parsed=_parse_query(query)
        terms=parsed["terms"]
        codes=parsed["codes"]
        excluded=parsed["excluded"]
        lexw=LEX_WEIGHT
        code_keys=set()
        ocr_used=False
        if terms or codes:
            try:
                conn=sqlite3.connect(directory/'lexical.sqlite')
            except sqlite3.Error:
                conn=None
            if conn is not None:
                try:
                    # Mandatory codes first: exact branch/code must survive mixing.
                    if codes:
                        try:
                            code_and=' AND '.join('"'+_lex_norm(c).replace('"','""')+'"' for c in codes)
                            code_rows=conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',(code_and,CAND_DEPTH)).fetchall()
                        except sqlite3.Error:
                            code_rows=[]
                        for rank,(rowid,) in enumerate(code_rows,1):
                            if not (1 <= rowid <= len(store.metadata)):continue
                            m=store.metadata[rowid-1]
                            code_keys.add(m['key'])
                            item=ranks.setdefault(m['key'],{'metadata':m,'score':0,'semantic_score':None})
                            item['score']+=lexw/(60+rank)+IDENT_BONUS/(60+rank)
                        # OCR-tolerant expansion only when exact codes miss.
                        if not code_keys:
                            for c in codes:
                                for v in ocr_variants(c):
                                    try:
                                        v_rows=conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',
                                                            ('"'+_lex_norm(v).replace('"','""')+'"',CAND_DEPTH)).fetchall()
                                    except sqlite3.Error:
                                        v_rows=[]
                                    if v_rows:
                                        ocr_used=True
                                        for rank,(rowid,) in enumerate(v_rows,1):
                                            if not (1 <= rowid <= len(store.metadata)):continue
                                            m=store.metadata[rowid-1]
                                            item=ranks.setdefault(m['key'],{'metadata':m,'score':0,'semantic_score':None})
                                            item['score']+=(lexw*0.5)/(60+rank+10)
                                        break
                                if ocr_used:
                                    break
                    if _route != "exact":
                        expression_and=' AND '.join('"'+term.replace('"','""')+'"' for term in terms)
                        rows=conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',(expression_and,CAND_DEPTH)).fetchall()
                        if len(rows)<3:
                            expression=' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
                            rows=conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',(expression,CAND_DEPTH)).fetchall()
                    else:
                        # Exact route: lexical over codes/phrases only, no semantic blur.
                        must=[_lex_norm(c) for c in codes]+[_lex_norm(p) for p in parsed["phrases"] if p.strip()] or terms
                        expression_and=' AND '.join('"'+t.replace('"','""')+'"' for t in must)
                        rows=conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',(expression_and,CAND_DEPTH)).fetchall()
                        if len(rows)<3:
                            expression=' OR '.join('"'+t.replace('"','""')+'"' for t in must)
                            rows=conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',(expression,CAND_DEPTH)).fetchall()
                    # Translit expansion (RU latin -> Cyrillic), rank-only, never a filter.
                    translit = parsed.get("translit_terms") or []
                    if translit and _route == "semantic" and len(rows) < 3:
                        try:
                            t_rows=conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',
                                (' OR '.join('"'+t.replace('"','""')+'"' for t in translit),CAND_DEPTH)).fetchall()
                        except sqlite3.Error:
                            t_rows=[]
                        for rank,(rowid,) in enumerate(t_rows,1):
                            if not (1 <= rowid <= len(store.metadata)):continue
                            m=store.metadata[rowid-1]
                            item=ranks.setdefault(m['key'],{'metadata':m,'score':0,'semantic_score':None})
                            item['score']+=(lexw*0.5)/(60+rank+10)
                except sqlite3.Error:
                    rows=[]
                finally:conn.close()
                for rank,(rowid,) in enumerate(rows,1):
                    if not (1 <= rowid <= len(store.metadata)):continue
                    m=store.metadata[rowid-1]
                    item=ranks.setdefault(m['key'],{'metadata':m,'score':0,'semantic_score':None})
                    item['score']+=lexw/(60+rank)
                    if m['key'] in code_keys:
                        item['score']+=IDENT_BONUS/(60+rank)
        # Mandatory code filter: when the query names explicit codes and the
        # index contains them, chunks without any code must not push the
        # correct branch out of the final limit.
        if codes and code_keys:
            def _has_code(text):
                low=text.lower()
                for c in codes:
                    if c.lower() in low:
                        return True
                return False
            filtered={key:item for key,item in ranks.items() if _has_code(item['metadata']['text']) or key in code_keys}
            if filtered:
                ranks=filtered
        if excluded:
            ranks={key:item for key,item in ranks.items()
                   if not any(x in item['metadata']['text'].lower() for x in excluded)}
        qlow=query.lower()
        for item in ranks.values():
            if qlow in item['metadata']['text'].lower():
                item['score']+=0.05
        found=[];per_file=Counter()
        for item in sorted(ranks.values(),key=lambda x:x['score'],reverse=True):
            m=item['metadata'];p=root/m['path']
            if per_file[m['path']]>=4:continue
            per_file[m['path']]+=1
            _snippet=m['text']
            if m.get('table_header') and m['table_header'] not in _snippet:
                _snippet=m['table_header']+'\n'+_snippet  # display prefix; stored text/ranking untouched
            hit={'file':str(p),'start_line':m['start_line'],'end_line':m['end_line'],'snippet':_snippet,'score':round(item['score'],6),'semantic_score':item['semantic_score'],'changed_since_index':not p.exists() or p.stat().st_mtime_ns!=m['mtime_ns'],'page':_hit_page(m),'locator':_hit_locator(m,_hit_page(m))}
            if m.get('ocr_lines'):
                hit['ocr_lines']=m['ocr_lines']
            if m.get('ocr_page'):
                hit['ocr_page']=m['ocr_page']
            for _kk in ('title','section','table_header','prev_key','next_key','chunk_version'):
                if m.get(_kk) is not None:
                    hit[_kk]=m[_kk]
            found.append(hit)
            if len(found)>=k:break
        return found

def list_projects():
    """List locally indexed project folders, chunk counts and index locations."""
    out = []
    import probe_licensing as _pl
    records = {p.parent: p for p in BASE.glob('*/project.json')}
    if not _pl.allowed('semantic'):
        records.update({p.parent: p for p in BASE.glob('*/free_project.json')})
    for p in sorted(records.values()):
        try:
            out.append(json.loads(p.read_text(encoding='utf8')))
        except (OSError, ValueError):
            continue
    return out


def search_all(query, k=6, per_shard=4, hot_hours=0, allowed_folders=None):
    """Federated search across all indexed shards. hot_hours>0 searches only
    shards indexed within that window first (hot tier), then the rest."""
    import time as _t
    projs = list_projects()
    if allowed_folders is not None:
        allowed = {os.path.normcase(str(Path(f).resolve())) for f in allowed_folders}
        projs = [p for p in projs if os.path.normcase(str(Path(p['folder']).resolve())) in allowed]
    now = _t.time()

    def hot(p):
        try:
            age = (now - Path(p['index_directory']).stat().st_mtime) / 3600
            return age <= hot_hours
        except (OSError, KeyError):
            return False

    ordered = sorted(projs, key=lambda p: (not hot(p)) if hot_hours else 0)
    if hot_hours:
        ordered = [p for p in ordered if hot(p)] + [p for p in projs if not hot(p)]
    allhits = []
    for p in ordered:
        try:
            hits = search_project(p['folder'], query, per_shard)
        except (ValueError, RuntimeError, OSError, IndexError, sqlite3.Error, KeyError):
            continue
        for h in hits:
            h['project'] = p['folder']
            allhits.append(h)
    return sorted(allhits, key=lambda h: h['score'], reverse=True)[:max(1, min(k, 24))]


# --- Search status taxonomy (display-time only; ranking untouched) -----------
# search_project/search_all keep returning plain hit lists (positional callers
# are safe). Status is derived afterwards from (folder, query, hits) plus a
# read-only lexical re-probe with the same FTS construction as _search_dir.
# Rule: an exact miss of an ad-hoc string (e.g. ZZ99999) is an honest
# "no match"; a MISSING registry-family code (BR/KX/STUDY...) SHOULD be in the
# index — its absence means a stale/partial index or a missing OCR layer, so
# the honest signal is "re-index", not silence.

STATUSES = ('ok', 'no_matches', 'index_incomplete', 'engine_unavailable', 'bad_query')

REGISTRY_FAMILIES = ('BR', 'KX', 'STUDY')


def is_registry_code(code):
    """Registry-minted identifier (branch / reference codes)."""
    c = (code or '').strip().upper()
    return any(c.startswith(p) and any(ch.isdigit() for ch in c[len(p):len(p) + 4])
               for p in REGISTRY_FAMILIES)


def _lexical_code_probe(directory, codes):
    """(n_code_keys, variant_hit) with the same FTS construction as the
    _search_dir code branch. Read-only; sqlite errors propagate."""
    conn = sqlite3.connect(str(Path(directory) / 'lexical.sqlite'))
    try:
        n = conn.execute('SELECT COUNT(*) FROM docs').fetchone()[0]
        if n == 0:
            return 0, False
        code_and = ' AND '.join('"' + _lex_norm(c).replace('"', '""') + '"' for c in codes)
        try:
            n_code = conn.execute(
                'SELECT COUNT(*) FROM docs WHERE docs MATCH ?', (code_and,)).fetchone()[0]
        except sqlite3.Error:
            n_code = 0
        variant_hit = False
        if n_code == 0:
            for c in codes:
                for v in ocr_variants(c):
                    try:
                        r = conn.execute(
                            'SELECT COUNT(*) FROM docs WHERE docs MATCH ?',
                            ('"' + _lex_norm(v).replace('"', '""') + '"',)).fetchone()[0]
                    except sqlite3.Error:
                        r = 0
                    if r:
                        variant_hit = True
                        break
                if variant_hit:
                    break
        return n_code, variant_hit
    finally:
        conn.close()


def _zero_hit_status(query_route, codes, present):
    """Status for zero hits given lexical presence. Never raises."""
    if not codes:
        return 'no_matches', 'route=%s; zero candidates after filters' % (query_route,)
    if present:
        return 'no_matches', ('route=%s; code(s) %s are indexed but removed '
                              'by filters' % (query_route, codes))
    if any(is_registry_code(c) for c in codes):
        return 'index_incomplete', ('registry code(s) %s absent from lexical index '
                                    '(stale/partial index or OCR text not indexed); '
                                    're-index (enable OCR)'
                                    % ([c for c in codes if is_registry_code(c)],))
    return 'no_matches', 'route=%s; zero candidates after filters' % (query_route,)


def search_status(folder, query, hits):
    """Read-only status for a completed search. Never raises; ranking untouched."""
    # Probe entitlement: search_status
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        return ('ok', 'exact words') if hits else ('no_matches', 'exact words')
    q = (query or '').strip()
    if not q:
        return 'bad_query', 'empty query'
    if hits:
        try:
            r = route(q)
        except Exception:
            r = 'semantic'
        return 'ok', 'route=%s, %d hit(s)' % (r, len(hits))
    try:
        root, directory = location(folder)
    except Exception as e:
        return 'engine_unavailable', 'location: %r' % (e,)
    if not (directory / 'project.json').exists():
        return 'engine_unavailable', 'not indexed; call index_project first'
    try:
        r = route(q)
    except Exception:
        r = 'semantic'
    codes = (_parse_query(q).get('codes') or [])
    try:
        n_code, variant_hit = _lexical_code_probe(directory, codes)
    except Exception as e:
        return 'engine_unavailable', 'lexical probe: %s: %s' % (type(e).__name__, repr(e)[:200])
    return _zero_hit_status(r, codes, bool(n_code or variant_hit))


def federation_search_status(folders, query):
    """Federation-level status for /search_all. Never raises."""
    # Probe entitlement: federation_search_status
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        return ('no_matches', 'exact words; no matching terms')
    q = (query or '').strip()
    if not q:
        return 'bad_query', 'empty query'
    try:
        r = route(q)
    except Exception:
        r = 'semantic'
    codes = (_parse_query(q).get('codes') or [])
    if not codes:
        return 'no_matches', 'route=%s; zero candidates after filters (federated)' % (r,)
    present, probed = False, 0
    for folder in folders or []:
        try:
            root, directory = location(folder)
        except Exception:
            continue
        if not (directory / 'project.json').exists():
            continue
        probed += 1
        try:
            n_code, variant_hit = _lexical_code_probe(directory, codes)
        except Exception:
            continue
        if n_code or variant_hit:
            present = True
            break
    if probed == 0:
        return 'engine_unavailable', 'no searchable project indexes'
    st, det = _zero_hit_status(r, codes, present)
    return st, det + ' (federated over %d index(es))' % (probed,)


def _neighbour_view(nm):
    """Display view of one neighbour chunk. Never raises."""
    try:
        text = nm.get('text') or ''
        head = nm.get('table_header')
        snippet = (head + '\n' + text) if (head and head not in text) else text
        return {'key': nm.get('key'), 'path': nm.get('path'), 'text': text,
                'snippet': snippet, 'start_line': nm.get('start_line'),
                'end_line': nm.get('end_line'),
                'locator': _hit_locator(nm, _hit_page(nm))}
    except Exception:
        return None


def get_neighbours(folder, key):
    """+/-1 chunk context by key (display-time; never duplicated into the index).
    Read-only, no migration: v2 prev_key/next_key are used when present,
    otherwise neighbours resolve by same-file chunk order.
    Returns {'prev': chunk_or_None, 'next': chunk_or_None}."""
    try:
        root, directory = location(folder)
    except Exception:
        return {'prev': None, 'next': None}
    try:
        store = _load_cached(directory)
        metas = store.metadata.to_list() if hasattr(store.metadata, 'to_list') else list(store.metadata)
    except Exception:
        return {'prev': None, 'next': None}
    by_key = {}
    for m in metas:
        try:
            by_key[m.get('key')] = m
        except (AttributeError, TypeError):
            continue
    m = by_key.get(key)
    if m is None:
        return {'prev': None, 'next': None}
    try:
        same = sorted((x for x in metas if x.get('path') == m.get('path')),
                      key=lambda x: (x.get('start_line') or 0, x.get('key') or ''))
    except Exception:
        return {'prev': None, 'next': None}
    idx = next((i for i, x in enumerate(same) if x.get('key') == key), None)
    if idx is None:
        return {'prev': None, 'next': None}
    order_prev = same[idx - 1] if idx > 0 else None
    order_next = same[idx + 1] if idx + 1 < len(same) else None
    # Explicit v2 links win as a whole when the chunk carries them (chunk
    # windows may overlap, so chain order can differ from line order; mixing
    # the two would pair inconsistent neighbours). Chunks stored without any
    # links (legacy, no migration) fall back to same-file order.
    if m.get('prev_key') or m.get('next_key'):
        linked_prev = by_key.get(m.get('prev_key')) if m.get('prev_key') else None
        linked_next = by_key.get(m.get('next_key')) if m.get('next_key') else None
        out = {'prev': _neighbour_view(linked_prev) if linked_prev else None,
               'next': _neighbour_view(linked_next) if linked_next else None}
        return out
    out = {'prev': _neighbour_view(order_prev) if order_prev else None,
           'next': _neighbour_view(order_next) if order_next else None}
    return out


def search_twostage(folder, query, k=6, width=60):
    """Two-stage: coarse candidates (vector top-N + FTS top-N), then exact
    cosine rerank on stored vectors + lexical RRF. Returns same shape as search."""
    root, directory = location(folder)
    if not (directory / 'project.json').exists():
        raise ValueError('Project is not indexed; call index_project first')
    if not query.strip():
        return []
    manifest = directory / 'shards.json'
    if manifest.exists():
        try:
            names = json.loads(manifest.read_text(encoding='utf8'))['shards']
        except (OSError, ValueError, KeyError):
            names = []
        if names:
            import concurrent.futures as _cf
            allhits = []
            with _cf.ThreadPoolExecutor(max_workers=min(SEARCH_FANOUT, len(names))) as ex:
                futs = {ex.submit(_twostage_dir, root, directory / n, query, k, width): n
                        for n in names}
                for fu in _cf.as_completed(futs):
                    try:
                        allhits.extend(fu.result())
                    except (ValueError, RuntimeError, OSError, IndexError, sqlite3.Error):
                        pass
            merged, per_file = [], Counter()
            for h in sorted(allhits, key=lambda x: x['score'], reverse=True):
                if per_file[h['file']] >= 4:
                    continue
                per_file[h['file']] += 1
                merged.append(h)
                if len(merged) >= k:
                    break
            return merged
    import numpy as _np
    from project_search.neuro_vector_store import VectorStore, embed_texts
    cpu_setup()
    k = max(1, min(k, 12))
    return _twostage_dir(root, directory, query, k, width)


def _twostage_dir(root, directory, query, k, width):
    import numpy as _np
    from project_search.neuro_vector_store import VectorStore, embed_texts
    with locked(directory):
        store = _load_cached(directory)
        with _embedding_lock:
            qvec = embed_texts([query], 'query', store.model)
            sem = store.search(query, width, return_indices=True)
        cand = {}
        for rank, (m, s, position) in enumerate(sem, 1):
            cand[m['key']] = {'metadata': m, 'sem_rank': rank, 'position': position}
        parsed = _parse_query(query)
        terms = parsed["terms"]
        codes = parsed["codes"]
        if terms:
            try:
                conn = sqlite3.connect(directory / 'lexical.sqlite')
            except sqlite3.Error:
                conn = None
            if conn is not None:
                try:
                    expression_and = ' AND '.join('"' + t.replace('"', '""') + '"' for t in terms)
                    rows = conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',
                                        (expression_and, width)).fetchall()
                    if len(rows) < 3:
                        expression = ' OR '.join('"' + t.replace('"', '""') + '"' for t in terms)
                        rows = conn.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',
                                            (expression, width)).fetchall()
                except sqlite3.Error:
                    rows = []
                finally:
                    conn.close()
                for rank, (rowid,) in enumerate(rows, 1):
                    if 1 <= rowid <= len(store.metadata):
                        m = store.metadata[rowid - 1]
                        cand.setdefault(m['key'], {'metadata': m, 'sem_rank': None,
                                                   'position': rowid - 1})['lex_rank'] = rank
        keys = list(cand)
        exact = {}
        if store.vectors is not None:
            for key in keys:
                i = cand[key]['position']
                if i is not None and i < len(store.vectors):
                    exact[key] = float(_np.dot(store.vectors[i], qvec[0]))
        qlow = query.lower()
        scored = []
        for key, c in cand.items():
            m = c['metadata']
            s = 0.7 * exact.get(key, -1.0)
            if 'lex_rank' in c:
                s += 0.6 / (60 + c['lex_rank'])
            if qlow in m['text'].lower():
                s += 0.05
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        found, per_file = [], Counter()
        for s, m in scored:
            p = root / m['path']
            if per_file[m['path']] >= 4:
                continue
            per_file[m['path']] += 1
            hit = {'file': str(p), 'start_line': m['start_line'], 'end_line': m['end_line'],
                   'snippet': (m['table_header'] + '\n' + m['text'] if m.get('table_header') and m['table_header'] not in (m.get('text') or '') else m['text']), 'score': round(s, 6),
                   'changed_since_index': not p.exists() or p.stat().st_mtime_ns != m['mtime_ns'],
                   'page': _hit_page(m), 'locator': _hit_locator(m, _hit_page(m))}
            if m.get('ocr_lines'):
                hit['ocr_lines'] = m['ocr_lines']
            if m.get('ocr_page'):
                hit['ocr_page'] = m['ocr_page']
            for _kk in ('title', 'section', 'table_header', 'prev_key', 'next_key', 'chunk_version'):
                if m.get(_kk) is not None:
                    hit[_kk] = m[_kk]
            found.append(hit)
            if len(found) >= k:
                break
        return found
