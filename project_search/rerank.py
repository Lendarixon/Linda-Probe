"""Optional CrossEncoder reranker. Disabled unless LOCALDOCS_RERANK=1/on/true."""
import os

_MODEL = None
_MODEL_NAME = None

DEFAULT_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'


def _model_name():
    return os.environ.get('LOCALDOCS_RERANK_MODEL', DEFAULT_MODEL)


def _default_top_n():
    try:
        return int(os.environ.get('LOCALDOCS_RERANK_N', '40'))
    except ValueError:
        return 20


import contextlib
import threading

_FORCE = threading.local()


@contextlib.contextmanager
def precise(on=True):
    """Per-request, thread-local switch for the precise (cross-encoder) mode."""
    previous = getattr(_FORCE, 'on', False)
    _FORCE.on = bool(on)
    try:
        yield
    finally:
        _FORCE.on = previous


def enabled():
    if getattr(_FORCE, 'on', False):
        return True
    return os.environ.get('LOCALDOCS_RERANK', '').strip().lower() in ('1', 'on', 'true')


def _get_model():
    global _MODEL, _MODEL_NAME
    name = _model_name()
    if _MODEL is not None and _MODEL_NAME == name:
        return _MODEL
    from sentence_transformers import CrossEncoder
    _MODEL = CrossEncoder(name, device='cpu', max_length=256)
    _MODEL_NAME = name
    return _MODEL


def rerank(query, hits, top_n=None):
    """Re-score first top_n hits by (query, snippet) pairs, sort head desc."""
    try:
        if not hits:
            return hits
        if top_n is None:
            top_n = _default_top_n()
        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            top_n = _default_top_n()
        n = min(max(top_n, 0), len(hits))
        if n <= 0:
            return hits
        model = _get_model()
        pairs = [(query, (h.get('snippet', '') if isinstance(h, dict) else ''))
                 for h in hits[:n]]
        scores = []
        for i in range(0, len(pairs), 16):
            batch = model.predict(pairs[i:i + 16])
            scores.extend(list(batch))
        for h, s in zip(hits[:n], scores):
            h['rerank_score'] = float(s)
        head = sorted(hits[:n], key=lambda x: x.get('rerank_score', float('-inf')),
                      reverse=True)
        return head + hits[n:]
    except Exception:
        return hits
