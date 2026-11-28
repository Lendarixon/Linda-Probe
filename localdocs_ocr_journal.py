"""Persistent OCR outcomes (not a text cache), with serialized SQLite access.

One row per file version/requested engine. error_streak counts consecutive
failures; successful outcomes and explicit skipped records reset the streak.
'uncertain' rows (low-confidence output) behave like errors: a few automatic
attempts, then skip until an explicit retry. Stale rows are ignored when
the caller reports a different model identity.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import threading
import time

ROOT = Path(__file__).resolve().parent
_LOCK = threading.RLock()
_STATUSES = {'ok', 'empty', 'error', 'skipped', 'uncertain'}
_RETRYABLE = ('error', 'uncertain')
_TTL_DAYS = 180


def signature(path):
    path = os.path.normcase(os.path.abspath(os.fspath(path)))
    st = os.stat(path)
    return path, st.st_mtime_ns, st.st_size


@contextmanager
def _db():
    with _LOCK:
        target = Path(os.environ.get('LOCALDOCS_OCR_JOURNAL') or
                      Path(os.environ.get('LOCALDOCS_WRITE_DIR') or ROOT) / 'indexes' / '_ocr_journal.sqlite3')
        target.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(target), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS ocr_log (
                path TEXT, mtime_ns INTEGER, size INTEGER, engine TEXT,
                device TEXT, lang TEXT,
                status TEXT CHECK(status IN ('ok','empty','error','skipped','uncertain')),
                chars INTEGER, seconds REAL, error TEXT, at TEXT,
                error_streak INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT '',
                dpi REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(path, mtime_ns, size, engine))''')
            _migrate(db)
            try:
                db.commit()
            except Exception:
                pass
            # Serialize writers with BEGIN IMMEDIATE + backoff (cross-process).
            last = None
            for delay in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
                try:
                    db.execute('BEGIN IMMEDIATE')
                    last = None
                    break
                except sqlite3.OperationalError as exc:
                    msg = str(exc).lower()
                    if 'locked' in msg or 'busy' in msg:
                        last = exc
                        time.sleep(delay)
                        continue
                    raise
            if last is not None:
                db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                # TTL auto-cleanup on exit (freshly written rows have new `at`).
                try:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=_TTL_DAYS)).isoformat()
                    db.execute("DELETE FROM ocr_log WHERE at IS NOT NULL AND at < ?", (cutoff,))
                except Exception:
                    pass
                try:
                    db.commit()
                except Exception:
                    pass
            except BaseException:
                try:
                    db.rollback()
                except Exception:
                    pass
                raise
        finally:
            try:
                db.close()
            except Exception:
                pass


def _migrate(db):
    """Bring pre-existing journals to the current schema, preserving rows."""
    info = [row['name'] for row in db.execute('PRAGMA table_info(ocr_log)')]
    if 'model' not in info:
        db.execute("ALTER TABLE ocr_log ADD COLUMN model TEXT NOT NULL DEFAULT ''")
        info.append('model')
    if 'dpi' not in info:
        db.execute("ALTER TABLE ocr_log ADD COLUMN dpi REAL NOT NULL DEFAULT 0")
        info.append('dpi')
    sql = db.execute(
        "SELECT sql FROM sqlite_master WHERE name='ocr_log'").fetchone()[0] or ''
    if 'uncertain' not in sql:
        db.execute('''CREATE TABLE ocr_log_new (
            path TEXT, mtime_ns INTEGER, size INTEGER, engine TEXT,
            device TEXT, lang TEXT,
            status TEXT CHECK(status IN ('ok','empty','error','skipped','uncertain')),
            chars INTEGER, seconds REAL, error TEXT, at TEXT,
            error_streak INTEGER NOT NULL DEFAULT 0,
            model TEXT NOT NULL DEFAULT '',
            dpi REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(path, mtime_ns, size, engine))''')
        cols = 'path,mtime_ns,size,engine,device,lang,status,chars,seconds,error,at,error_streak,model,dpi'
        # Old tables lack dpi: select 0 as dpi for preserved rows.
        old_cols = 'path,mtime_ns,size,engine,device,lang,status,chars,seconds,error,at,error_streak,model'
        if 'dpi' in info and 'model' in info:
            db.execute(f'INSERT INTO ocr_log_new ({cols}) SELECT {cols} FROM ocr_log')
        elif 'model' in info:
            db.execute(f'INSERT INTO ocr_log_new ({cols}) SELECT {old_cols},0 FROM ocr_log')
        else:
            base = 'path,mtime_ns,size,engine,device,lang,status,chars,seconds,error,at,error_streak'
            db.execute(f'INSERT INTO ocr_log_new ({cols}) SELECT {base},\'\',0 FROM ocr_log')
        db.execute('DROP TABLE ocr_log')
        db.execute('ALTER TABLE ocr_log_new RENAME TO ocr_log')


def record(path, status, engine, device, lang, chars=0, seconds=0.0, error='',
           *, file_signature=None, model=None, dpi=None):
    """Record the version captured before OCR, even if the input changed meanwhile.

    model is an opaque caller-provided identity of the recogniser behind
    engine; empty/error/uncertain rows are skipped only when it matches, so a
    model change invalidates stale failed/empty outcomes. None keeps the
    legacy behaviour (stored as '' and matched unconditionally). dpi/lang/device
    work the same way for empty rows: None means legacy match-all, otherwise a
    mismatch forces one fresh attempt.
    """
    if status not in _STATUSES:
        raise ValueError('Invalid OCR status')
    key = file_signature if file_signature is not None else signature(path)
    dpi_value = 0.0 if dpi is None else float(dpi)
    with _db() as db:
        db.execute('''INSERT INTO ocr_log
            (path,mtime_ns,size,engine,device,lang,status,chars,seconds,error,at,error_streak,model,dpi)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path,mtime_ns,size,engine) DO UPDATE SET
            device=excluded.device, lang=excluded.lang, status=excluded.status,
            chars=excluded.chars, seconds=excluded.seconds, error=excluded.error,
            at=excluded.at, model=excluded.model, dpi=excluded.dpi,
            error_streak=CASE WHEN excluded.status
                IN ('error','uncertain') THEN ocr_log.error_streak+1 ELSE 0 END''',
            (*key, engine, device, lang, status, int(chars), float(seconds), str(error),
             datetime.now(timezone.utc).isoformat(), int(status in _RETRYABLE),
             '' if model is None else str(model), dpi_value))


def _row_model(row):
    try:
        return row['model'] or ''
    except (IndexError, KeyError):
        return ''


def _row_lang(row):
    try:
        return row['lang']
    except (IndexError, KeyError):
        return None


def _row_device(row):
    try:
        return row['device']
    except (IndexError, KeyError):
        return None


def _row_dpi(row):
    try:
        value = row['dpi']
    except (IndexError, KeyError):
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def should_skip(path, engine=None, *, include_ok=False, model=None,
                lang=None, device=None, dpi=None):
    """Indexing skips only empty/exhausted errors; studio may include ok.

    Uncertain (low-confidence) rows are retried automatically until their
    streak is exhausted, like errors. Passing model skips empty/error/
    uncertain rows only when the stored model identity matches, so a model
    change forces one fresh attempt; model=None preserves legacy matching.
    Passing lang/device/dpi skips empty rows only when they match, so a
    request change (language, device, DPI) forces one fresh attempt;
    None (default) preserves legacy matching for old callers.
    """
    try:
        key = signature(path)
    except OSError:
        return False
    sql = ('SELECT status,error_streak,model,lang,device,dpi FROM ocr_log '
           'WHERE path=? AND mtime_ns=? AND size=?')
    if engine is not None:
        sql += ' AND engine=?'
        key += (engine,)
    want_dpi = None
    if dpi is not None:
        try:
            want_dpi = float(dpi)
        except (TypeError, ValueError):
            want_dpi = None
    with _db() as db:
        for row in db.execute(sql, key):
            if row['status'] == 'ok':
                if include_ok:
                    return True
            elif row['status'] in ('empty',) + _RETRYABLE:
                if model is not None and _row_model(row) != model:
                    continue
                if row['status'] == 'empty':
                    if lang is not None and _row_lang(row) != lang:
                        continue
                    if device is not None and _row_device(row) != device:
                        continue
                    if want_dpi is not None:
                        stored = _row_dpi(row)
                        if stored is None or stored == 0:
                            continue  # legacy unknown DPI: force one attempt
                        if abs(stored - want_dpi) > 1e-6:
                            continue
                    return True
                if row['error_streak'] >= 3:
                    return True
    return False


def already_done(path, engine=None, *, model=None, lang=None, device=None, dpi=None):
    return should_skip(path, engine, include_ok=True, model=model,
                       lang=lang, device=device, dpi=dpi)


def retry(path=None, engine=None):
    """Explicit retry: drop error/uncertain rows so the next pass re-attempts them.

    ok/empty outcomes are kept. Returns the number of rows removed.
    """
    with _db() as db:
        sql = "DELETE FROM ocr_log WHERE status IN ('error','uncertain')"
        args = []
        if path is not None:
            sql += ' AND path=?'
            args.append(os.path.normcase(os.path.abspath(os.fspath(path))))
        if engine is not None:
            sql += ' AND engine=?'
            args.append(engine)
        return db.execute(sql, args).rowcount


def recent(limit=200, status=None, query=None):
    clauses, args = [], []
    if status is not None:
        clauses.append('status=?')
        args.append(status)
    if query:
        clauses.append('(instr(lower(path),lower(?))>0 OR instr(lower(error),lower(?))>0)')
        args.extend((str(query), str(query)))
    sql = 'SELECT * FROM ocr_log'
    if clauses:
        sql += ' WHERE ' + ' AND '.join(clauses)
    sql += ' ORDER BY at DESC, path LIMIT ?'
    args.append(max(0, int(limit)))
    with _db() as db:
        return [dict(row) for row in db.execute(sql, args)]


def stats():
    result = dict(total=0, ok=0, empty=0, error=0, skipped=0, chars=0, seconds=0.0)
    with _db() as db:
        for row in db.execute('SELECT status,COUNT(*) AS n,SUM(chars) AS chars,'
                              'SUM(seconds) AS seconds FROM ocr_log GROUP BY status'):
            result[row['status']] = row['n']
            result['total'] += row['n']
            result['chars'] += row['chars'] or 0
            result['seconds'] += row['seconds'] or 0.0
    return result


def forget(path=None):
    with _db() as db:
        if path is None:
            db.execute('DELETE FROM ocr_log')
        else:
            db.execute('DELETE FROM ocr_log WHERE path=?',
                       (os.path.normcase(os.path.abspath(os.fspath(path))),))


def cleanup_missing(limit=200):
    """Inspect at most limit distinct paths, oldest outcomes first."""
    removed = 0
    with _db() as db:
        paths = list(db.execute('SELECT path FROM ocr_log GROUP BY path '
                               'ORDER BY MAX(at) LIMIT ?', (max(0, int(limit)),)))
        for row in paths:
            try:
                os.stat(row['path'])
            except FileNotFoundError:
                removed += db.execute('DELETE FROM ocr_log WHERE path=?',
                                      (row['path'],)).rowcount
            except OSError:
                pass  # Permission and transient network errors are not missing files.
    return removed
