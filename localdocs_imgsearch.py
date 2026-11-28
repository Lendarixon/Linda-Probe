"""Offline CLIP image search. No sessions are created until embedding is requested.

embed_images returns (float32 vectors for valid images, boolean input mask).
Cancellation raises ImageSearchCancelled and leaves an existing index untouched.
Progress callbacks receive {done, total, path}. Inference is serialized globally;
ORT sessions use one CPU worker. models_dir overrides the active model directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re as _re
from pathlib import Path
import threading
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps

# Additive P4: script-based hint, not a language identifier. Any Cyrillic or
# Polish-specific diacritic suggests the multilingual text module.
_MULTI_HINT = _re.compile('[\u0400-\u04FF\u0104\u0106\u0118\u0141\u0143\u00D3\u015A\u0179\u017B\u0105\u0107\u0119\u0142\u0144\u00F3\u015B\u017A\u017C]')


def needs_multilingual_text(query):
    """True when the query contains Cyrillic or Polish diacritics."""
    return bool(_MULTI_HINT.search(str(query or '')))


def active_text_info():
    """Which photo text module would embed_text use. No inference, no network."""
    try:
        import localdocs_models
        model_id = localdocs_models.get_active('text_embed_clip')
        if not model_id:
            return {'model_id': None, 'multilingual': False, 'reason': 'no-active-text-model'}
        state = localdocs_models.status(model_id)
        if not state.get('installed', False):
            return {'model_id': model_id, 'multilingual': False, 'reason': 'text-model-not-installed'}
        directory = Path(state['path'])
        multi = any((directory / d / 'config.json').is_file() for d in ('2_Dense', '1_Dense'))
        return {'model_id': model_id, 'multilingual': bool(multi),
                'reason': '' if multi else 'standard-clip-text-only'}
    except Exception as exc:
        return {'model_id': None, 'multilingual': False, 'reason': str(exc)}

_INFERENCE_LOCK = threading.RLock()
_EMBED_LOCK = _INFERENCE_LOCK  # inference/embedding (same object, backward compatible)
_SEARCH_LOCK = threading.RLock()  # vector search + normalized-vector cache
_GPU_VERDICT = {}
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.heic')


def image_paths(paths, limit=None, should_stop=None):
    """Expand selections without following symlinks/junctions. Deduplicate overlaps."""
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError('paths must be a nonempty list')
    found = {}
    def linked(path):
        st = path.lstat()
        return path.is_symlink() or bool(getattr(st, 'st_file_attributes', 0) & 0x400)
    def add(path):
        _stop(should_stop)
        if not linked(path) and path.is_file() and path.suffix.lower() in EXTENSIONS:
            found[os.path.normcase(str(path))] = path
            if limit is not None and len(found) > limit:
                raise OverflowError('selection exceeds %d image files; select fewer files' % limit)
    for raw in paths:
        _stop(should_stop)
        if not isinstance(raw, (str, os.PathLike)) or not os.path.isabs(raw):
            raise ValueError('paths must be absolute')
        path = Path(os.path.abspath(raw))
        if not path.exists():
            raise ValueError('path not found: ' + str(path))
        if any(linked(p) for p in [path] + list(path.parents)):
            raise ValueError('reparse points are not allowed')
        if path.is_dir():
            for directory, dirs, files in os.walk(path, followlinks=False):
                _stop(should_stop)
                dirs[:] = sorted(d for d in dirs if not linked(Path(directory) / d))
                for name in sorted(files):
                    add(Path(directory) / name)
        else:
            add(path)
    return sorted(found.values())


class ImageSearchCancelled(RuntimeError):
    """The caller cancelled an embedding or indexing operation."""


def _stop(callback):
    if callback is not None and callback():
        raise ImageSearchCancelled('Image search cancelled')


def _json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Encoder must return finite [N,D] embeddings')
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _find(directory, pattern, prefer=()):
    matches = sorted(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f'Model file missing: {pattern}')
    for word in prefer:  # e.g. fp16 graphs run on DirectML but crash the CPU optimizer; int8 ones the reverse
        for match in matches:
            if word in match.name:
                return match
    return matches[0]


def _dense_weights(path):
    """Read float32 safetensors with numpy only; never unpickle downloaded code."""
    import struct
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if header_size > 1024 * 1024:
            raise ValueError('Invalid safetensors header')
        header = json.loads(stream.read(header_size))
        payload = stream.read()
    tensors = {}
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        start, end = spec['data_offsets']
        shape = spec['shape']
        if (spec['dtype'] != 'F32' or not 0 <= start <= end <= len(payload)
                or any(type(n) is not int or n < 0 for n in shape)
                or end - start != int(np.prod(shape)) * 4):
            raise ValueError('Invalid Dense tensor')
        tensors[name] = np.frombuffer(payload[start:end], dtype='<f4').reshape(shape).copy()
    return tensors


class _MultilingualText:
    """SentenceTransformer Transformer -> masked mean -> Dense, CPU only."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.graph = _find(self.directory / 'onnx', '*.onnx', ('quint8_avx2', 'qint8', 'model.onnx'))
        # The upstream model uses 2_Dense; accept 1_Dense for compatible exports.
        dense = next((self.directory / name for name in ('2_Dense', '1_Dense')
                      if (self.directory / name / 'config.json').is_file()), None)
        if dense is None:
            raise FileNotFoundError('Dense configuration missing')
        self.config = _json(dense / 'config.json')
        tensors = _dense_weights(dense / 'model.safetensors')
        self.weight = tensors['linear.weight']
        self.bias = tensors.get('linear.bias')
        if self.weight.shape != (self.config['out_features'], self.config['in_features']):
            raise ValueError('Dense weight shape mismatch')
        if self.config.get('bias', True):
            if self.bias is None or self.bias.shape != (self.weight.shape[0],):
                raise ValueError('Dense bias missing or invalid')
        elif self.bias is not None:
            raise ValueError('Unexpected Dense bias')
        self.activation = self.config['activation_function'].rsplit('.', 1)[-1]
        if self.activation not in ('Identity', 'Tanh', 'ReLU'):
            raise ValueError('Unsupported Dense activation: ' + self.activation)
        self.length = int(_json(self.directory / 'sentence_bert_config.json').get('max_seq_length', 128))
        if not 0 < self.length <= 512:
            raise ValueError('Invalid text sequence length')
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=self.length)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('[PAD]') or 0, pad_token='[PAD]')
        self.session = None

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.weight.shape[0]), np.float32)
        if self.session is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session = ort.InferenceSession(str(self.graph), sess_options=options,
                                               providers=['CPUExecutionProvider'])
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        ids = np.asarray([e.ids for e in encoded], np.int64)
        mask = np.asarray([e.attention_mask for e in encoded], np.int64)
        inputs = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': np.zeros_like(ids)}
        names = [n.name for n in self.session.get_inputs()]
        if any(n not in inputs for n in names):
            raise ValueError('Unsupported multilingual encoder input')
        outputs = self.session.get_outputs()
        name = next((n.name for n in outputs if n.name == 'last_hidden_state'), outputs[0].name)
        values = np.asarray(self.session.run([name], {n: inputs[n] for n in names})[0], np.float32)
        if values.shape != (*ids.shape, self.weight.shape[1]):
            raise ValueError('Expected token embeddings before mean pooling and Dense')
        values = (values * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
        values = values @ self.weight.T
        if self.bias is not None:
            values += self.bias
        if self.activation == 'Tanh':
            values = np.tanh(values)
        elif self.activation == 'ReLU':
            values = np.maximum(values, 0)
        return _normalize(values)


def available():
    """Check local prerequisites without loading an encoder or accessing network."""
    try:
        if importlib.util.find_spec('onnxruntime') is None:
            return {'ready': False, 'reason': 'onnxruntime is not installed'}
        import localdocs_models
        model_id = localdocs_models.get_active('image_embed')
        if not model_id:
            return {'ready': False, 'reason': 'No active image_embed model'}
        directory = Path(localdocs_models.status(model_id)['path'])
        for pattern in ('vision_model*.onnx', 'text_model*.onnx', 'tokenizer.json'):
            _find(directory, pattern)
        return {'ready': True, 'reason': ''}
    except Exception as exc:
        return {'ready': False, 'reason': str(exc)}


class ImageSearch:
    def __init__(self, models_dir=None, device='cpu'):
        # Probe entitlement: __init__
        import probe_licensing as _pl
        _pl.require('photos')
        if device not in ('cpu', 'gpu'):
            raise ValueError('device must be cpu or gpu')
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.device = device
        self.device_used = None
        self._key = None
        self._sessions = {}
        self._files = {}
        self._dimension = 0
        self._text_key = None
        self._text_engine = None
        self._text_override = None
        self._norm_cache = {}

    def _active_text(self):
        import localdocs_models
        model_id = localdocs_models.get_active('text_embed_clip')
        if not model_id:
            self._text_key, self._text_engine, self._text_override = None, None, None
            return None
        state = localdocs_models.status(model_id)
        if not state.get('installed', False):
            self._text_key, self._text_engine, self._text_override = None, None, None
            return None
        directory = Path(state['path'])
        signature = [(str(p.relative_to(directory)), p.stat().st_mtime_ns, p.stat().st_size)
                     for p in sorted(directory.rglob('*')) if p.is_file()]
        key = (model_id, str(directory.resolve()), tuple(signature))
        if key != self._text_key:
            if any((directory / d / 'config.json').is_file() for d in ('2_Dense', '1_Dense')):
                engine = _MultilingualText(directory)
            else:
                # A separately selected standard CLIP text graph remains supported.
                engine = ImageSearch(models_dir=directory, device=self.device)
            self._text_engine, self._text_key = engine, key
            self._text_override = directory
        return self._text_engine

    def _prepare(self):
        if self.models_dir is None:
            import localdocs_models
            model_id = localdocs_models.get_active('image_embed')
            if not model_id:
                raise RuntimeError('No active image_embed model')
            directory = Path(localdocs_models.status(model_id)['path'])
        else:
            model_id, directory = 'directory', self.models_dir
        prefer = ('fp16', 'quantized') if self.device == 'gpu' else ('quantized',)
        files = {kind: _find(directory, pattern, prefer) for kind, pattern in (
            ('text', 'text_model*.onnx'), ('tokenizer', 'tokenizer.json'))}
        if any(directory.rglob('vision_model*.onnx')):
            files['vision'] = _find(directory, 'vision_model*.onnx', prefer)
        # Also fingerprint sidecars and external ONNX data to prevent stale reuse.
        signature = [(str(p.relative_to(directory)), p.stat().st_mtime_ns, p.stat().st_size)
                     for p in sorted(directory.rglob('*')) if p.is_file()]
        key = hashlib.sha256(json.dumps([model_id, str(directory.resolve()), signature]).encode()).hexdigest()
        if key == self._key:
            return
        self._files, self._sessions = files, {}
        self._dimension = 0
        self.device_used = None
        self._manifest = _json(directory / 'model.json')
        config = _json(directory / 'preprocessor_config.json')
        size = config.get('crop_size', config.get('size', 224))
        if isinstance(size, dict):
            self._height = int(size.get('height', size.get('shortest_edge', 224)))
            self._width = int(size.get('width', size.get('shortest_edge', 224)))
        else:
            self._height = self._width = int(size)
        resize = config.get('size', min(self._height, self._width))
        self._resize = int(resize.get('shortest_edge', min(self._height, self._width))) if isinstance(resize, dict) else int(resize)
        self._mean = np.asarray(config.get('image_mean', [0.48145466, 0.4578275, 0.40821073]), np.float32)
        self._std = np.asarray(config.get('image_std', [0.26862954, 0.26130258, 0.27577711]), np.float32)
        if min(self._height, self._width, self._resize) <= 0 or self._mean.shape != (3,) or self._std.shape != (3,) or np.any(self._std <= 0):
            raise ValueError('Invalid image preprocessing configuration')
        self._tokenizer = None
        self._key = key

    def _gpu_safe(self, path):
        """DirectML can crash the whole process (access violation) on some graphs, e.g. int8-quantized ones:
        try the session in a throw-away child process first and remember the verdict."""
        key = (str(path), os.path.getmtime(path))
        if key in _GPU_VERDICT:
            return _GPU_VERDICT[key]
        import subprocess, sys
        code = ("import onnxruntime as o,sys\n"
                "o.InferenceSession(sys.argv[1], providers=['DmlExecutionProvider','CPUExecutionProvider'])\n")
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        try:
            ok = subprocess.run([sys.executable, '-I', '-c', code, str(path)], capture_output=True,
                                timeout=120, creationflags=flags).returncode == 0
        except Exception:
            ok = False
        _GPU_VERDICT[key] = ok
        return ok

    def _session(self, kind, cpu=False):
        import onnxruntime as ort
        if not cpu and self.device == 'gpu' and kind not in self._sessions and not self._gpu_safe(self._files[kind]):
            cpu = True
        if cpu:
            self._sessions.clear()
            self.device_used = 'cpu'
        if kind not in self._sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ['CPUExecutionProvider']
            if self.device == 'gpu' and self.device_used != 'cpu':
                if 'DmlExecutionProvider' in ort.get_available_providers():
                    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                    options.enable_mem_pattern = False
            try:
                session = ort.InferenceSession(str(self._files[kind]), sess_options=options, providers=providers)
            except Exception:
                if len(providers) == 1:
                    raise
                return self._session(kind, cpu=True)
            self.device_used = 'gpu' if 'DmlExecutionProvider' in session.get_providers() else 'cpu'
            self._sessions[kind] = session
        return self._sessions[kind]

    def _names(self, kind, field, actual):
        manifest = self._manifest
        graph = manifest.get('compatibility', {}).get('graphs', {}).get(self._files[kind].name, {})
        graph = manifest.get('graphs', {}).get(self._files[kind].name, graph)
        value = graph.get(field, manifest.get(field, {}))
        if isinstance(value, dict):
            value = value.get(kind, value.get(self._files[kind].name, []))
        if isinstance(value, str):
            value = [value]
        names = [v.get('name') if isinstance(v, dict) else v for v in value] if isinstance(value, list) else []
        # Metadata supplies ordering, while the graph remains authoritative.
        return [n for n in names if n in actual] + [n for n in actual if n not in names]

    def _run(self, kind, make_feed):
        session = self._session(kind)
        def run(current):
            inputs = self._names(kind, 'inputs', [n.name for n in current.get_inputs()])
            outputs = self._names(kind, 'outputs', [n.name for n in current.get_outputs()])
            return current.run([outputs[0]], make_feed(inputs, current))[0]
        try:
            return run(session)
        except Exception:
            if self.device_used != 'gpu':
                raise
            return run(self._session(kind, cpu=True))

    def _image(self, path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            width, height = image.size
            scale = max(self._resize / min(width, height), self._width / width, self._height / height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
            left, top = (image.width - self._width) // 2, (image.height - self._height) // 2
            image = image.crop((left, top, left + self._width, top + self._height))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        return ((pixels - self._mean) / self._std).transpose(2, 0, 1)

    def embed_images(self, paths, batch=16, progress=None, should_stop=None):
        """Return (valid vectors, mask); unreadable images have mask=False."""
        # Probe entitlement: embed_images
        import probe_licensing as _pl
        _pl.require('photos')
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('batch must be a positive integer')
        paths = list(paths)
        with _EMBED_LOCK:
            self._prepare()
            mask = np.zeros(len(paths), dtype=bool)
            vectors, pixels = [], []
            def flush():
                if pixels:
                    _stop(should_stop)
                    data = np.stack(pixels).astype(np.float32)
                    result = _normalize(self._run('vision', lambda names, session: {names[0]: data}))
                    if len(result) != len(pixels):
                        raise ValueError('Vision encoder batch mismatch')
                    vectors.append(result)
                    self._dimension = result.shape[1]
                    pixels.clear()
            for i, path in enumerate(paths):
                _stop(should_stop)
                try:
                    value = self._image(path)
                except (OSError, ValueError, SyntaxError):
                    value = None
                if value is not None:
                    pixels.append(value)
                    mask[i] = True
                if len(pixels) >= batch:
                    flush()
                if progress:
                    progress({'done': i + 1, 'total': len(paths), 'path': str(path)})
            flush()
            _stop(should_stop)
            return (np.concatenate(vectors) if vectors else np.empty((0, self._dimension), np.float32)), mask

    def embed_text(self, texts):
        # Probe entitlement: embed_text
        import probe_licensing as _pl
        _pl.require('photos')
        texts = [texts] if isinstance(texts, str) else list(texts)
        with _EMBED_LOCK:
            # Explicit directory instances are self-contained (including the
            # standard-CLIP delegate above); default instances honor active text.
            if self.models_dir is None:
                encoder = self._active_text()
                if encoder is not None:
                    return encoder.encode(texts) if isinstance(encoder, _MultilingualText) else encoder.embed_text(texts)
            self._prepare()
            if self._tokenizer is None:
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(self._files['tokenizer']))
                tokenizer.no_padding()
                tokenizer.enable_truncation(max_length=77)
                self._tokenizer = tokenizer
            tokenizer = self._tokenizer
            pad = tokenizer.token_to_id('[PAD]')
            if pad is None:
                pad = tokenizer.token_to_id('<|endoftext|>')
            pad = 0 if pad is None else pad
            ids = np.full((len(texts), 77), pad, dtype=np.int64)
            mask = np.zeros_like(ids)
            bos, eos = tokenizer.token_to_id('<|startoftext|>'), tokenizer.token_to_id('<|endoftext|>')
            for row, encoded in enumerate(tokenizer.encode_batch(texts, add_special_tokens=True)):
                tokens = encoded.ids
                if bos is not None and (not tokens or tokens[0] != bos):
                    tokens = [bos] + tokens
                if eos is not None and (not tokens or tokens[-1] != eos):
                    tokens = tokens + [eos]
                tokens = tokens[:77]
                if eos is not None and tokens:
                    tokens[-1] = eos
                ids[row, :len(tokens)] = tokens
                mask[row, :len(tokens)] = 1
            if not texts:
                return np.empty((0, self._dimension), np.float32)
            def feed(names, session):
                result = {}
                for name in names:
                    if 'attention_mask' in name:
                        result[name] = mask
                    elif 'position_ids' in name:
                        result[name] = np.broadcast_to(np.arange(77, dtype=np.int64), ids.shape).copy()
                    elif 'token_type_ids' in name:
                        result[name] = np.zeros_like(ids)
                    else:
                        result[name] = ids
                return result
            values = self._run('text', feed)
            if values.ndim == 3:
                values = values[np.arange(len(ids)), ids.argmax(axis=1)]
            values = _normalize(values)
            if len(values) != len(texts):
                raise ValueError('Text encoder batch mismatch')
            self._dimension = values.shape[1]
            return values

    def _load(self, store_path):
        with np.load(store_path, allow_pickle=False) as data:
            paths = data['paths'].astype(str).tolist()
            mtimes, sizes, vectors = data['mtimes'].copy(), data['sizes'].copy(), data['vectors'].astype(np.float32)
            key = str(data['model_key'].item()) if 'model_key' in data else None
        if vectors.ndim != 2 or not (len(paths) == len(mtimes) == len(sizes) == len(vectors)):
            raise ValueError('Invalid image index')
        return paths, mtimes, sizes, vectors, key

    def index_paths(self, paths, store_path, should_stop=None, progress=None):
        selected = image_paths(paths, should_stop=should_stop)
        return self.index_folder(None, store_path, should_stop=should_stop, progress=progress, _paths=selected)

    def index_folder(self, folder, store_path, extensions=EXTENSIONS, should_stop=None, progress=None, _paths=None):
        with _EMBED_LOCK:
            self._prepare()
            _stop(should_stop)
            old, reused = {}, 0
            if Path(store_path).is_file():
                paths, mtimes, sizes, vectors, key = self._load(store_path)
                if key == self._key:
                    old = {p: (m, s, v) for p, m, s, v in zip(paths, mtimes, sizes, vectors)}
            allowed = {e.lower() if e.startswith('.') else '.' + e.lower() for e in extensions}
            if _paths is not None and Path(store_path).is_file() and key != self._key:
                raise ValueError('Active model changed; rebuild the custom index')
            records, pending = (dict(old) if _paths is not None else {}), []
            reused = len(records)
            for path in (_paths if _paths is not None else sorted(Path(folder).rglob('*'))):
                _stop(should_stop)
                if not path.is_file() or path.suffix.lower() not in allowed:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                name = str(path)
                if name in old and old[name][:2] == (stat.st_mtime_ns, stat.st_size):
                    records[name] = (stat.st_mtime_ns, stat.st_size, old[name][2])
                    if _paths is None:
                        reused += 1
                else:
                    if name in records:
                        records.pop(name)
                        reused -= 1
                    pending.append((name, stat.st_mtime_ns, stat.st_size))
            if pending:
                embedded, mask = self.embed_images([p[0] for p in pending], progress=progress, should_stop=should_stop)
                for record, vector in zip((p for p, valid in zip(pending, mask) if valid), embedded):
                    # A file changed during embedding must be retried next time.
                    try:
                        stat = Path(record[0]).stat()
                    except OSError:
                        continue
                    if (stat.st_mtime_ns, stat.st_size) == record[1:]:
                        records[record[0]] = (record[1], record[2], vector)
            _stop(should_stop)
            names = sorted(records)
            vectors = np.stack([records[p][2] for p in names]) if names else np.empty((0, self._dimension), np.float32)
            target = Path(store_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.' + uuid4().hex + '.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, paths=np.asarray(names, dtype=str),
                                        mtimes=np.asarray([records[p][0] for p in names], np.int64),
                                        sizes=np.asarray([records[p][1] for p in names], np.int64),
                                        vectors=vectors.astype(np.float16), model_key=np.asarray(self._key))
                    stream.flush()
                    os.fsync(stream.fileno())
                _stop(should_stop)
                os.replace(temporary, target)
            finally:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            return {'indexed': len(names), 'reused': reused, 'skipped': len(pending) - (len(names) - reused)}

    def _norm_cached(self, store_path):
        """Cached (paths, normalized vectors); no per-query copies or re-normalization."""
        try:
            skey = str(store_path)
            fst = os.stat(skey)
            sig = (fst.st_mtime_ns, fst.st_size)
        except OSError:
            sig = None
        with _EMBED_LOCK:
            cur_key = self._key
        ck = (skey, sig, cur_key)
        with _SEARCH_LOCK:
            hit = self._norm_cache.get(ck)
            if hit is not None:
                return hit
        paths, _, _, vectors, key = self._load(store_path)
        if key != cur_key:
            raise ValueError('Active model changed; reindex the folder')
        normed = _normalize(vectors)
        with _SEARCH_LOCK:
            # Keep at most a few stores; callers fan out over <=8 stores.
            if len(self._norm_cache) >= 16:
                try:
                    self._norm_cache.pop(next(iter(self._norm_cache)))
                except StopIteration:
                    pass
            self._norm_cache[ck] = (paths, normed)
            return self._norm_cache[ck]

    def _search(self, store_path, vector, limit, min_score=None, exclude=None):
        if limit < 0:
            raise ValueError('limit must be nonnegative')
        if limit == 0:
            return []
        paths, normed = self._norm_cached(store_path)
        if not paths:
            return []
        qv = vector if isinstance(vector, np.ndarray) and vector.dtype == np.float32 else np.asarray(vector, dtype=np.float32)
        if qv.ndim != 1 or qv.shape[0] != normed.shape[1]:
            raise ValueError(
                'Photo text vector dim %s does not match photo index dim '
                '%d; the query text module is incompatible with this photo '
                'index (re-encode with the matching text module, do not reuse vectors)' % (tuple(qv.shape), normed.shape[1]))
        scores = normed @ qv
        results = []
        for i in np.argsort(-scores, kind='stable'):
            if exclude is not None and os.path.normcase(os.path.abspath(paths[i])) == exclude:
                continue
            score = float(np.clip(scores[i], -1, 1))
            if min_score is not None and score < min_score:
                continue
            results.append({'path': paths[i], 'score': score})
            if len(results) >= limit:
                break
        return results

    def search_text(self, store_path, query, limit=60, min_score=None):
        with _EMBED_LOCK:
            self._prepare()
            vector = self.embed_text([query])[0]
        return self._search(store_path, vector, limit, min_score)

    def search_image(self, store_path, image_path, limit=60):
        with _EMBED_LOCK:
            vectors, mask = self.embed_images([image_path])
            if not mask[0]:
                raise ValueError('Cannot decode query image: ' + str(image_path))
            exclude = os.path.normcase(os.path.abspath(image_path))
            vector = vectors[0]
        return self._search(store_path, vector, limit, exclude=exclude)


# ===== P6b combined photo+tag search (additive; existing search_text/search_tags untouched) =====
def parse_photo_query(query):
    """Split one field into (text, tags). Tags: tag:xxx or #xxx (quotes allow spaces)."""
    import re
    raw = str(query or "")
    tags = []

    def _norm_tag(value):
        t = re.sub(r"\s+", " ", str(value).strip().strip('"\'' ).replace("_", " ")).strip().lower()
        if not t or len(t) > 60 or "http" in t:
            return ""
        return t

    # tag:"multi word", tag:'multi word', tag:single
    def _sub_tag(m):
        word = m.group(1) or m.group(2) or m.group(3) or ""
        norm = _norm_tag(word)
        if norm:
            tags.append(norm)
        return " "

    raw = re.sub(r'tag:\s*"([^"]{1,60})"\s*|tag:\s*\'([^\']{1,60})\'\s*|tag:\s*([^\s#,;]+)', _sub_tag, raw)

    def _sub_hash(m):
        norm = _norm_tag(m.group(1) or "")
        if norm:
            tags.append(norm)
        return " "

    raw = re.sub(r'#([^\s#,;]{1,60})', _sub_hash, raw)
    text = re.sub(r"\s+", " ", raw).strip()
    seen, uniq = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return text, uniq


def merge_photo_results(sem, tag_hits, limit=60, tag_filter="all", assign_map=None):
    """Merge semantic + tag hits: intersection first, then rest by score, deduped.

    sem: [{path, score}], tag_hits: [{path, score, tags?}].
    tag_filter: all/with (only tagged)/without (only untagged).
    assign_map: {path: [tags]} or {normpath: [tags]} for filter accuracy; None skips any-tag knowledge.
    """
    import os
    if tag_filter not in ("all", "with", "without"):
        raise ValueError('tag_filter must be all/with/without')
    try:
        limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        raise ValueError('limit must be an integer')

    def _key(p):
        return os.path.normcase(os.path.normpath(str(p or "")))

    def _score(r):
        try:
            return float(r.get("score", 0))
        except (TypeError, ValueError):
            return 0.0

    sem_map, sem_order = {}, []
    for r in sem or []:
        k = _key(r.get("path", ""))
        if not k or k in sem_map:
            continue
        sem_map[k] = r
        sem_order.append(k)
    tag_map = {}
    for r in tag_hits or []:
        k = _key(r.get("path", ""))
        if not k or k in tag_map:
            continue
        tag_map[k] = r

    norm_assign = {}
    if assign_map:
        for p, ts in (assign_map or {}).items():
            names = [str(t) for t, _ in ts] if ts and isinstance(ts[0], (list, tuple)) else [str(t) for t in (ts or [])]
            norm_assign[_key(p)] = names

    def _tags_for(k):
        if k in tag_map and isinstance(tag_map[k].get("tags"), list) and tag_map[k]["tags"]:
            return [str(t) for t in tag_map[k]["tags"]]
        return list(norm_assign.get(k, []))

    def _has_any_tag(k):
        if k in norm_assign:
            return bool(norm_assign[k])
        if k in tag_map:
            got = tag_map[k].get("tags")
            return bool(got)
        return False

    inter, rest = [], []
    for k in list(sem_map.keys() | tag_map.keys()):
        s = _score(sem_map[k]) if k in sem_map else None
        t = _score(tag_map[k]) if k in tag_map else None
        score = max([v for v in (s, t) if v is not None] or [0.0])
        src = "both" if (k in sem_map and k in tag_map) else ("semantic" if k in sem_map else "tag")
        path = (sem_map.get(k) or tag_map.get(k) or {}).get("path", "")
        item = {"path": path, "score": score, "source": src, "tags": _tags_for(k)}
        (inter if (k in sem_map and k in tag_map) else rest).append(item)
    inter.sort(key=lambda r: -r["score"])
    rest.sort(key=lambda r: -r["score"])
    ordered = inter + rest
    if tag_filter == "with":
        ordered = [r for r in ordered if _has_any_tag(_key(r["path"])) or bool(r.get("tags"))]
    elif tag_filter == "without":
        ordered = [r for r in ordered if (not _has_any_tag(_key(r["path"]))) and not r.get("tags")]
    return ordered[:limit]


def search_combined(engine, store_path, query, limit=60, tag_mode="any", tag_filter="all",
                    tag_search=None, available_fn=None, assign_map=None):
    """One query over meaning + tags. Returns {results, reason, detail, text, tags, device_used}.

    reason: '' ok | no-model | no-photos | no-tag | no-match | filtered | empty-query
    Never changes search_text/search_tags behaviour; only orchestrates them.
    Synthetic-friendly: engine/tag_search/available_fn/assign_map injectable (no ONNX needed).
    """
    import os
    if tag_mode not in ("any", "all"):
        raise ValueError('tag_mode must be any/all')
    if tag_filter not in ("all", "with", "without"):
        raise ValueError('tag_filter must be all/with/without')
    limit = max(1, min(int(limit), 200))
    text, tags = parse_photo_query(query)
    if tag_search is None:
        from localdocs_tags import search_tags as tag_search
    if available_fn is None:
        available_fn = available
    device_used = str(getattr(engine, "device_used", "") or "")
    # Model gate only when semantic text is actually needed.
    ready_info = {"ready": True, "reason": ""}
    if text:
        try:
            ready_info = dict(available_fn() or {"ready": True, "reason": ""})
        except Exception as exc:
            ready_info = {"ready": False, "reason": str(exc)}
        if not ready_info.get("ready"):
            return {"results": [], "reason": "no-model", "detail": str(ready_info.get("reason", "")),
                    "text": text, "tags": tags, "device_used": device_used}
    # Photo count (best effort; fake engines may lack _load).
    photos = None
    try:
        loader = getattr(engine, "_load", None)
        if loader is not None:
            photos = len((loader(str(store_path))[0]) or [])
        else:
            photos = None
    except (OSError, ValueError):
        photos = 0
    except Exception:
        photos = None
    if photos == 0:
        return {"results": [], "reason": "no-photos", "detail": "",
                "text": text, "tags": tags, "device_used": device_used}
    sem, tag_hits = [], []
    if text:
        sem = list(engine.search_text(store_path, text, limit) or [])
    if tags:
        tag_hits = list(tag_search(store_path, tags, tag_mode, limit) or [])
    if not text and not tags:
        return {"results": [], "reason": "empty-query", "detail": "",
                "text": text, "tags": tags, "device_used": device_used}
    # Assign map for with/without filter (any-tag knowledge).
    amap = assign_map
    if amap is None and tag_filter != "all":
        try:
            from localdocs_tags import load as _tag_load
            data = _tag_load(store_path) or {}
            raw = data.get("assign", {}) or {}
            amap = {}
            for p, picked in raw.items():
                if picked and isinstance(picked[0], (list, tuple)):
                    amap[p] = [str(t) for t, _ in picked]
                else:
                    amap[p] = [str(t) for t in (picked or [])]
        except Exception:
            amap = {}
    merged_all = merge_photo_results(sem, tag_hits, limit, "all", amap)
    merged = merge_photo_results(sem, tag_hits, limit, tag_filter, amap) if tag_filter != "all" else merged_all
    if merged:
        return {"results": merged, "reason": "", "detail": "",
                "text": text, "tags": tags, "device_used": device_used or str(getattr(engine, "device_used", "") or "")}
    if merged_all and not merged:
        return {"results": [], "reason": "filtered", "detail": tag_filter,
                "text": text, "tags": tags, "device_used": device_used}
    if tags and not tag_hits:
        return {"results": [], "reason": "no-tag", "detail": ", ".join(tags),
                "text": text, "tags": tags, "device_used": device_used}
    return {"results": [], "reason": "no-match", "detail": "",
            "text": text, "tags": tags, "device_used": device_used} 
