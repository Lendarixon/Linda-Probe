"""Free exact-word content search, no embeddings, OCR, or network downloads.

Uses a separate SQLite file so changing edition never destroys a Pro index.
"""
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid
from contextlib import closing

def index(folder, should_stop=None):
    from project_search import backend as B
    _, directory = B.location(folder)
    with B.locked(directory):
        return _index_locked(folder, should_stop)

def _index_locked(folder, should_stop=None):
    from project_search import backend as B
    root, directory = B.location(folder)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / ('free-' + uuid.uuid4().hex + '.sqlite')
    report = {'folder': str(root), 'files': 0, 'chunks': 0, 'partial': False,
              'model': 'exact-words', 'index_directory': str(directory), 'skipped': 0, 'needs_ocr': 0}
    try:
        with closing(sqlite3.connect(tmp)) as db:
            db.execute('CREATE VIRTUAL TABLE docs USING fts5(path UNINDEXED,page UNINDEXED,text)')
            for p, rel in B.files(root):
                if should_stop and should_stop():
                    return dict(report, partial=True, stopped=True)
                if p.suffix.lower() in B._IMG_EXT: continue
                try:
                    if p.suffix.lower() == '.pdf':
                        from pypdf import PdfReader
                        pages = [(i + 1, pg.extract_text() or '') for i, pg in enumerate(PdfReader(str(p)).pages)]
                        report['needs_ocr'] += sum(not text.strip() for _, text in pages)
                    elif p.suffix.lower() == '.docx':
                        from project_search.localdocs import extract_pages
                        pages = [(pg['page'], pg['text']) for pg in extract_pages(p)]
                    elif p.suffix.lower() in B._FX_EXT:
                        from project_search.extractors import extract
                        pages = [(1, extract(p) or '')]
                    else:
                        pages = [(1, B._decode_plain(p.read_bytes()) or '')]
                    n = 0
                    for page, text in pages:
                        for offset in range(0, len(text), 700):
                            part = text[offset:offset + 900]
                            if part.strip():
                                db.execute('INSERT INTO docs(path,page,text) VALUES(?,?,?)', (rel, page, part)); n += 1
                    if n: report['files'] += 1
                    else: report['skipped'] += 1
                    report['chunks'] += n
                except Exception as exc:
                    if isinstance(exc, sqlite3.Error): raise
                    report['skipped'] += 1
            db.commit()
        os.replace(tmp, directory / 'free_text.sqlite')
        metadata = directory / 'free_project.json'
        temp_meta = metadata.with_suffix('.tmp')
        temp_meta.write_text(json.dumps(report), encoding='utf-8'); os.replace(temp_meta, metadata)
        return report
    finally: tmp.unlink(missing_ok=True)

def search(folder, query, limit=6, match='all'):
    """match='all': every word must occur (default, file search). match='any': BM25 over any word of 3+ letters
    (natural-language questions, used by Linda Sejf; a question never has all its words in one passage)."""
    from project_search import backend as B
    root, directory = B.location(folder)
    path = directory / 'free_text.sqlite'
    if not path.is_file(): raise ValueError('Exact-word index is missing; update this folder index')
    terms = re.findall(r'\w+', str(query), flags=re.UNICODE)
    if match == 'any':
        terms = list(dict.fromkeys(t.lower() for t in terms if len(t) >= 3 or t.isdigit()))
    if not terms: return []
    match = (' OR ' if match == 'any' else ' AND ').join('"' + t.replace('"', '""') + '"' for t in terms)
    out = []
    with closing(sqlite3.connect('file:' + path.as_posix() + '?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT path,page,text,bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?',
                          (match, max(1, min(int(limit), 200)))).fetchall()
    for rel, page, text, rank in rows:
        full = (root / rel).resolve()
        if not full.is_relative_to(root) or not full.is_file(): continue
        out.append({'file': str(full), 'path': rel, 'page': page, 'start_line': page, 'end_line': page,
            'snippet': text, 'text': text, 'score': -rank, 'source': 'text', 'stale': False})
    return out
