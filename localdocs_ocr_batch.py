"""Local OCR studio: one background worker, cooperative page cancellation.

No downloads or subprocesses. Optional PDF dependencies are imported lazily.
Timestamps are Unix seconds; done counts completed, failed and skipped inputs.
Cancelled in-flight inputs have error='stopped'; queued inputs remain unprocessed.
"""

import copy
import ctypes
import io
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import warnings

import localdocs_ocr_journal as journal


SUPPORTED = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
TEXT_LIMIT = 5 * 1024 * 1024  # UTF-8 bytes across all text results in one job.
_WORKER = threading.Lock()  # Also prevents concurrent GPU work in separate Jobs.


class _Stopped(Exception):
    pass


def _low_priority():
    if os.name == 'nt':
        try:
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentThread.restype = ctypes.c_void_p
            kernel.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            kernel.SetThreadPriority(kernel.GetCurrentThread(), -1)
        except Exception:
            pass


def _model_for(engine):
    """Model identity for journal gating; None = legacy match-all (see journal)."""
    try:
        import localdocs_ocr as _ocr_mod
        ident = _ocr_mod._model_identity(engine)
        return None if ident == "unknown" else ident
    except Exception:
        return None


def _pages(src, dpi, check):
    """Yield owned RGB images, closed as soon as the consumer advances."""
    from PIL import Image, ImageOps
    if src.suffix.lower() == '.pdf':
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(str(src)) as document:
            for index in range(len(document)):
                check()
                page = document[index]
                try:
                    bitmap = page.render(scale=dpi / 72)
                    try:
                        image = bitmap.to_pil()
                        try:
                            yield image
                        finally:
                            image.close()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
    else:
        with Image.open(src) as source:
            for index in range(getattr(source, 'n_frames', 1)):
                check()
                source.seek(index)
                frame = ImageOps.exif_transpose(source)
                try:
                    rgba = frame.convert('RGBA')
                    try:
                        image = Image.new('RGB', rgba.size, 'white')
                        image.paste(rgba, mask=rgba.getchannel('A'))
                        try:
                            yield image
                        finally:
                            image.close()
                    finally:
                        rgba.close()
                finally:
                    if frame is not source:
                        frame.close()


def _raster_pdf(src, target, dpi, check):
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    try:
        for image in _pages(src, dpi, check):
            check()
            with io.BytesIO() as buffer:
                rgb = image.convert('RGB')
                try:
                    rgb.save(buffer, format='PDF', resolution=dpi)
                finally:
                    rgb.close()
                buffer.seek(0)
                reader = PdfReader(buffer)
                writer.add_page(reader.pages[0])
        check()
        with target.open('wb') as output:
            writer.write(output)
    finally:
        writer.close()


def _publish(staged, directory, stem, suffix):
    """Exclusive creation protects existing files even against concurrent writers."""
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        tail = '' if number == 1 else f' ({number})'
        target = directory / (stem + tail + suffix)
        try:
            output = target.open('xb')
        except FileExistsError:
            number += 1
            continue
        try:
            with output, staged.open('rb') as source:
                shutil.copyfileobj(source, output)
        except BaseException:
            target.unlink()  # Only the new file exclusively created above.
            raise
        return str(target)


class Job:
    def __init__(self):
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._state = dict(phase='idle', total=0, done=0, skipped_count=0, current=None,
                           started_at=None, finished_at=None, results=[], stopped=False)

    def status(self):
        """Return a detached, thread-safe snapshot."""
        with self._lock:
            return copy.deepcopy(self._state)

    def stop(self):
        """Request cancellation; an active OCR page is allowed to finish."""
        with self._lock:
            if self._state['phase'] == 'running':
                self._stop.set()
                self._state['stopped'] = True

    def _check(self):
        if self._stop.is_set():
            raise _Stopped('stopped')

    def start(self, items, mode, outdir, lang, device, engine, dpi=200, skip_done=True,
              *, rotation=0, spreads=False):
        # Probe entitlement: start
        import probe_licensing as _pl
        _pl.require('ocr')
        if mode not in {'pdf_layer', 'pdf_new', 'txt', 'md', 'text'}:
            raise ValueError('Invalid mode')
        if device not in {'cpu', 'gpu'}:
            raise ValueError('Invalid device')
        if engine not in {'auto', 'own', 'rapid', 'tesseract'}:
            raise ValueError('Invalid engine')
        if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or dpi <= 0:
            raise ValueError('dpi must be finite and positive')
        if rotation not in (0, 90, 180, 270):
            raise ValueError('rotation must be 0/90/180/270')
        if not isinstance(spreads, bool):
            raise ValueError('spreads must be bool')
        paths = [Path(item).absolute() for item in items]
        destination = Path(outdir).absolute() if outdir else None
        with self._lock:
            if self._state['phase'] == 'running' or not _WORKER.acquire(blocking=False):
                raise RuntimeError('An OCR batch is already running')
            self._stop.clear()
            self._state = dict(phase='running', total=0, done=0, skipped_count=0, current=None,
                               started_at=time.time(), finished_at=None, results=[], stopped=False)
            self._thread = threading.Thread(target=self._run,
                args=(paths, mode, destination, lang, device, engine, dpi, skip_done, rotation, spreads),
                name='localdocs-ocr-batch', daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._state.update(phase='done', finished_at=time.time())
                _WORKER.release()
                raise

    def _run(self, items, mode, outdir, lang, device, engine, dpi, skip_done=True, rotation=0, spreads=False):
        _low_priority()
        used = 0
        try:
            # Snapshot before output creation, so generated files cannot re-enter.
            expanded = []
            seen = set()
            for item in items:
                self._check()
                if item.is_dir():
                    def walk_error(error):
                        expanded.append((Path(error.filename or item), str(error)))
                    for root, dirs, files in os.walk(item, followlinks=False, onerror=walk_error):
                        self._check()
                        dirs.sort()
                        expanded.extend((Path(root) / name, None) for name in sorted(files))
                else:
                    expanded.append((item, None))
            queue = []
            for path, error in expanded:
                key = os.path.normcase(str(path.resolve()))
                if key not in seen:
                    seen.add(key)
                    queue.append((path, error))
            with self._lock:
                self._state['total'] = len(queue)
            for src, discovery_error in queue:
                self._check()
                started = time.perf_counter()
                file_key = None
                geometry_default = (int(rotation or 0) % 360 == 0 and not spreads)
                result = dict(src=str(src), dst=None, kind=mode, chars=0, seconds=0.0, error=None)
                with self._lock:
                    self._state['current'] = str(src)
                try:
                    if discovery_error:
                        raise OSError(discovery_error)
                    if not src.is_file():
                        raise FileNotFoundError(str(src))
                    geometry_default = (int(rotation or 0) % 360 == 0 and not spreads)
                    try:
                        file_key = journal.signature(src)
                        done = bool(skip_done and geometry_default and journal.already_done(
                            src, engine, model=_model_for(engine),
                            lang=lang, device=device, dpi=dpi))
                    except Exception as error:
                        done = False
                        warnings.warn('OCR journal unavailable: ' + str(error), RuntimeWarning)
                    if src.suffix.lower() not in SUPPORTED:
                        result.update(kind='skipped', skipped=True)
                    elif done:
                        result.update(kind='skipped', skipped=True, journal_done=True)
                    else:
                        used += self._process(src, mode, outdir, lang, device, engine, dpi,
                                              result, TEXT_LIMIT - used, rotation, spreads)
                except Exception as error:
                    result['error'] = str(error) or type(error).__name__
                result['seconds'] = time.perf_counter() - started
                # Preserve ok/empty/error streaks when merely skipping a done file.
                # Cancellation is not a recognition failure and must remain retryable.
                if (file_key is not None and geometry_default and not result.get('journal_done')
                        and result['error'] != 'stopped'):
                    if result['error']:
                        status = 'error'
                    elif result.get('skipped'):
                        status = 'skipped'
                    elif int(result.get('_ocr_uncertain') or 0) > 0:
                        status = 'uncertain'
                    elif result.get('_ocr_empty', not result['chars']):
                        status = 'empty'
                    else:
                        status = 'ok'
                    err_text = result['error'] or (str(result.get('_ocr_reason') or '') if status == 'uncertain' else '')
                    try:
                        journal.record(src, status, engine, result.get('device', device), lang, chars=result['chars'],
                                       seconds=result['seconds'], error=err_text,
                                       file_signature=file_key, model=_model_for(engine), dpi=dpi)
                    except Exception as error:
                        warnings.warn('OCR journal write failed: ' + str(error), RuntimeWarning)
                if (not result['error'] and not result.get('skipped') and not result.get('journal_done')
                        and int(result.get('_ocr_uncertain') or 0) > 0):
                    result['uncertain_pages'] = int(result.get('_ocr_uncertain') or 0)
                    if result.get('_ocr_reason'):
                        result['uncertain_reason'] = str(result.get('_ocr_reason'))
                if (not result['error'] and not result.get('skipped') and not result.get('journal_done')
                        and result.get('_ocr_pages') is not None):
                    try:
                        result['pages'] = [dict(uncertain=bool(p.get('uncertain')),
                                               conf=float(p.get('conf', 0.0)))
                                         for p in (result.get('_ocr_pages') or [])]
                    except Exception:
                        pass
                result.pop('_ocr_empty', None)
                result.pop('_ocr_uncertain', None)
                result.pop('_ocr_reason', None)
                result.pop('_ocr_pages', None)
                with self._lock:
                    self._state['results'].append(result)
                    self._state['done'] += 1
                    self._state['skipped_count'] += int(bool(result.get('skipped')))
        except _Stopped:
            pass
        except Exception as error:
            with self._lock:
                self._state['results'].append(dict(src='', dst=None, kind=mode,
                    chars=0, seconds=0.0, error=str(error) or type(error).__name__))
        finally:
            with self._lock:
                self._state.update(phase='done', current=None, finished_at=time.time(),
                                   stopped=self._stop.is_set())
                _WORKER.release()

    def _process(self, src, mode, outdir, lang, device, engine, dpi, result, budget,
                 rotation=0, spreads=False):
        directory = outdir if outdir is not None else src.parent
        if mode in {'text', 'txt', 'md'}:
            import localdocs_ocr
            texts = []
            size = 0
            uncertain_pages = 0
            uncertain_reason = ''
            pages = []
            for image in _pages(src, dpi, self._check):
                self._check()
                recognized = localdocs_ocr.ocr_image(image, lang, device, engine,
                                                         rotation=rotation or None, spreads=spreads)
                try:
                    _st = str(recognized.get('status') or '')
                    _cf = float(recognized.get('confidence', 0.0) or 0.0)
                except Exception:
                    _st, _cf = '', 0.0
                pages.append(dict(uncertain=(_st == 'uncertain'), conf=_cf))
                if _st == 'uncertain':
                    uncertain_pages += 1
                    if not uncertain_reason:
                        uncertain_reason = str(recognized.get('reason') or '')
                text = str(recognized['text'])
                result['device'] = recognized.get('device', device)
                self._check()
                size += len(text.encode('utf-8'))
                if mode == 'text' and size > budget:
                    raise ValueError('Combined text exceeds 5 MiB UTF-8 limit')
                texts.append(text)
            result['_ocr_empty'] = not any(part.strip() for part in texts)
            result['_ocr_uncertain'] = uncertain_pages
            result['_ocr_reason'] = uncertain_reason
            result['_ocr_pages'] = pages
            if mode == 'md':
                text = '# ' + src.stem + '\n\n' + '\n\n'.join(
                    f'## Страница {i}\n\n{part}' for i, part in enumerate(texts, 1)) + '\n'
            elif len(texts) > 1:
                text = ''.join(f'\n\n— страница {i} —\n\n{part}'
                               for i, part in enumerate(texts, 1))
            else:
                text = texts[0] if texts else ''
            self._check()
            if mode == 'text':
                size = len(text.encode('utf-8'))
                if size > budget:
                    raise ValueError('Combined text exceeds 5 MiB UTF-8 limit')
                result.update(text=text, chars=sum(map(len, texts)))
                return size
            with tempfile.TemporaryDirectory(prefix='localdocs-batch-') as temporary:
                staged = Path(temporary) / 'text'
                staged.write_text(text, encoding='utf-8')
                self._check()
                result['dst'] = _publish(staged, directory, src.stem, '.' + mode)
            result['chars'] = sum(map(len, texts))
        else:
            import localdocs_ocr
            from localdocs_pdf_ocr import make_searchable_pdf
            uncertain_pages = 0
            uncertain_reason = ''
            pages = []
            _orig_ocr = localdocs_ocr.ocr_image
            def _counting_ocr(image, *args, **kwargs):
                nonlocal uncertain_pages, uncertain_reason
                rec = _orig_ocr(image, *args, **kwargs)
                try:
                    _st = str(rec.get('status') or '')
                    try:
                        _cf = float(rec.get('confidence', 0.0) or 0.0)
                    except Exception:
                        _cf = 0.0
                    pages.append(dict(uncertain=(_st == 'uncertain'), conf=_cf))
                    if _st == 'uncertain':
                        uncertain_pages += 1
                        if not uncertain_reason:
                            uncertain_reason = str(rec.get('reason') or '')
                except Exception:
                    pass
                return rec
            with tempfile.TemporaryDirectory(prefix='localdocs-batch-') as temporary:
                folder = Path(temporary)
                source = src
                if mode == 'pdf_new' or src.suffix.lower() != '.pdf':
                    source = folder / 'raster.pdf'
                    _raster_pdf(src, source, dpi, self._check)
                self._check()
                staged = folder / 'searchable.pdf'
                localdocs_ocr.ocr_image = _counting_ocr
                try:
                    stats = make_searchable_pdf(source, staged, lang=lang, device=device,
                        engine=engine, dpi=dpi, max_pages=None,
                        progress=lambda _status: self._check(), force=(mode == 'pdf_new'),
                        rotation=rotation or None, spreads=spreads)
                finally:
                    localdocs_ocr.ocr_image = _orig_ocr
                self._check()
                stem = src.stem + ('.ocr' if mode == 'pdf_layer' else ' (OCR)')
                result['dst'] = _publish(staged, directory, stem, '.pdf')
                result['chars'] = stats['chars']
            result['_ocr_empty'] = not result['chars']
            result['_ocr_uncertain'] = uncertain_pages
            result['_ocr_reason'] = uncertain_reason
            result['_ocr_pages'] = pages
        return 0


_JOB = Job()


def get_job():
    return _JOB
