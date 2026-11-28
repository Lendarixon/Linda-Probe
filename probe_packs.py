"""Explicit, sequential preparation of optional Pro models. Never loads GPU."""
import threading
import probe_licensing as L

_lock = threading.Lock()
_job = {'phase': 'idle', 'items': [], 'current': '', 'error': ''}
_cancel = threading.Event()
MODEL = 'intfloat/multilingual-e5-small'

def status():
    with _lock: return dict(_job, items=[dict(x) for x in _job['items']])

def stop():
    _cancel.set()
    return status()

def start(selected=None):
    L.require('semantic')
    selected = list(dict.fromkeys(selected if selected is not None else ['semantic', 'ocr', 'photos']))
    if not selected or any(x not in L.FEATURES for x in selected): raise ValueError('invalid packs')
    with _lock:
        if _job['phase'] == 'downloading': return {'started': False, **status_unlocked()}
        _cancel.clear(); _job.update(phase='downloading', items=[], current='', error='')
    threading.Thread(target=_run, args=(selected,), name='probe-pack-download', daemon=True).start()
    return {'started': True, **status()}

def status_unlocked(): return dict(_job)

def _run(selected):
    import localdocs_models as M
    try:
        for pack in selected:
            if _cancel.is_set(): break
            L.require(pack)
            with _lock: _job.update(current=pack, done_bytes=0, total_bytes=None)
            if pack == 'semantic':
                from huggingface_hub import snapshot_download
                snapshot_download(MODEL, allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.model', '1_Pooling/*'], max_workers=1)
            else:
                ids = ['rapid-detector', 'rapid-rec-en', 'rapid-rec-latin', 'rapid-rec-cyrillic', 'linda-ocr-hf'] if pack == 'ocr' else ['clip-vit-b32-en', 'clip-multilingual-text']
                for model_id in ids:
                    L.require(pack)
                    if _cancel.is_set(): break
                    M.download(model_id, progress=_progress, should_stop=lambda: _cancel.is_set() or not L.allowed(pack))
                    entry = M._entry(model_id)
                    if model_id.startswith('rapid-rec-'): continue
                    for kind in entry.get('supported_kinds', [entry['kind']]):
                        if model_id == 'clip-vit-b32-en' and kind == 'text_embed_clip': continue
                        M.set_active(kind, model_id)
            L.require(pack)
            with _lock: _job['items'].append({'pack': pack, 'ready': not _cancel.is_set()})
        with _lock: _job.update(phase='cancelled' if _cancel.is_set() else 'done', current='')
    except Exception as exc:
        with _lock: _job.update(phase='cancelled' if _cancel.is_set() else 'error', error='pack_download_failed', detail=type(exc).__name__ + ': ' + str(exc)[:180], current='')

def _progress(value):
    with _lock: _job.update(done_bytes=value.get('done_bytes'), total_bytes=value.get('total_bytes'))
