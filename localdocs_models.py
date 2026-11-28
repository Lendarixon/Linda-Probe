"""Linda Probe model storage. Import/list/status never access the network.

Only explicit download and update-check APIs perform HTTP requests. Progress
callbacks receive a dict with file, done_bytes and total_bytes (whole model).
Cancellation raises ModelDownloadCancelled and retains the current .part file.
Unknown file sizes are None; catalog size_mb values may be estimates.
"""
from __future__ import annotations

import hashlib
import http.client
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
import math
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "models_catalog.json"
KINDS = frozenset(("ocr_rec", "ocr_det", "image_embed", "text_embed_clip", "embed"))
_STATE_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.Lock()
_JOB_LOCK = threading.Lock()
_JOB = {"phase": "idle", "model_id": None, "file": None,
        "done_bytes": 0, "total_bytes": 0, "error": None}
_CHUNK = 64 * 1024
_UPDATE = {"checked_at": None, "updates": {}, "new_models": []}
_UPDATE_THREAD = None
_UPDATE_WAKE = threading.Event()


def _https(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Catalog URLs must use HTTPS without credentials")
    return url


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_entry(item, remote=False):
    """Shared validation for bundled, cached remote and custom records."""
    if not isinstance(item, dict):
        raise ValueError("Model entry must be an object")
    _safe_id(item.get("id"))
    if item.get("kind") not in KINDS or item.get("source") not in ("hf", "bundled", "custom"):
        raise ValueError("Invalid model kind or source")
    if not isinstance(item.get("license"), str) or not item["license"].strip():
        raise ValueError("Model license required")
    if "commercial_ok" not in item or not (isinstance(item["commercial_ok"], bool) or item["commercial_ok"] == "unknown"):
        raise ValueError("Model commercial permission required")
    if remote and item["source"] != "hf":
        raise ValueError("Remote catalogs may only register HF downloads")
    if not isinstance(item.get("files"), list) or not item["files"]:
        raise ValueError("Model files required")
    if not set(item.get("supported_kinds", [item["kind"]])) <= KINDS:
        raise ValueError("Invalid supported kinds")
    if "version" in item:
        _version(item["version"])
    seen = set()
    for spec in item["files"]:
        path = spec["path"]
        _child(ROOT, path)
        if re.search(r'[<>"|?*\x00-\x1f]', path):
            raise ValueError("Invalid Windows filename")
        # Windows normalizes trailing dots/spaces and reserves device filenames.
        for part in PurePosixPath(path).parts:
            if part.endswith((".", " ")) or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", part):
                raise ValueError("Unsafe Windows filename")
        key = path.casefold()
        if key in seen or key in ("model.json", "active.json", "remote_catalog.json") or key.endswith((".part", ".part.json")):
            raise ValueError("Duplicate or reserved model filename")
        seen.add(key)
        if remote:
            _https(spec.get("url", ""))
        size = spec.get("size")
        if size is not None and (type(size) is not int or size <= 0):
            raise ValueError("Invalid file size")
        digest = spec.get("sha256")
        if digest and (not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)):
            raise ValueError("Invalid SHA256")
    if any(any(other.startswith(path + '/') for other in seen) for path in seen):
        raise ValueError("Model file cannot also be a parent directory")
    return item


def _version(value):
    """Numeric dotted versions; absent version is zero. No lexical 10 < 2 bug."""
    if isinstance(value, bool) or not re.fullmatch(r"\d+(?:\.\d+)*", str(value)):
        raise ValueError("Version must be numeric or dotted numeric")
    parts = [int(n) for n in str(value).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


class ModelDownloadError(RuntimeError):
    """A model could not be downloaded or verified."""


class ModelDownloadCancelled(ModelDownloadError):
    """The caller cancelled a download; partial files are resumable."""


def models_dir():
    override = os.environ.get("LOCALDOCS_MODELS_DIR")
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    # Packaged Windows hosts merge real AppData with a private LocalCache.
    # Resolve that storage explicitly before enforcing child-path containment;
    # otherwise an existing virtualized child appears to escape its real parent.
    if os.name == 'nt' and not override:
        import ctypes
        size = ctypes.c_uint32(0)
        kernel = ctypes.windll.kernel32
        if kernel.GetCurrentPackageFamilyName(ctypes.byref(size), None) == 122:
            family = ctypes.create_unicode_buffer(size.value)
            if kernel.GetCurrentPackageFamilyName(ctypes.byref(size), family) == 0:
                base = base / 'Packages' / family.value / 'LocalCache' / 'Local'
    path = Path(override).expanduser() if override else base / "LocalDocs" / "models"
    path.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt' and not override:
        # Some desktop hosts virtualize descendants without giving their child
        # Python process package identity. Only recognize the OS's exact private
        # AppData mirror, never arbitrary symlinks/junctions or other destinations.
        original_base = Path(os.environ.get('LOCALAPPDATA', ''))
        packages = (original_base / 'Packages').resolve()
        for child in path.iterdir():
            if not child.is_dir() or child.is_symlink() or child.is_junction(): continue
            actual = child.resolve().parent
            try: parts = actual.relative_to(packages).parts
            except ValueError: continue
            if len(parts) == 5 and tuple(p.lower() for p in parts[1:]) == ('localcache','local','localdocs','models'):
                path = actual
                break
    return path.resolve()


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise ValueError("Invalid model id")
    return value


def _child(root, relative):
    # Reject Windows drive/ADS/UNC and traversal even when tested on Unix.
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Invalid model file path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or parts == PurePosixPath("."):
        raise ValueError("Model file path must stay inside its directory")
    root = root.resolve()
    target = (root / Path(*parts.parts)).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Model file path escapes its directory")
    return target


def load_catalog():
    """Return fresh catalog entries, including locally registered custom models."""
    entries = _read_json(Path(CATALOG_PATH), [])
    if not isinstance(entries, list):
        raise ValueError("Model catalog must be a list")
    merged = {}
    for item in entries:
        _validate_entry(item)
        if item["id"] in merged:
            raise ValueError("Duplicate model id")
        merged[item["id"]] = item
    remote = _read_json(models_dir() / "remote_catalog.json", [])
    if not isinstance(remote, list):
        raise ValueError("Remote catalog must be a list")
    for item in remote:
        _validate_entry(item, remote=True)
        old = merged.get(item["id"])
        if old is None or _version(item.get("version", 0)) > _version(old.get("version", 0)):
            merged[item["id"]] = item
    entries = list(merged.values())
    custom = models_dir() / "custom"
    if custom.is_dir():
        for path in sorted(custom.glob("*/model.json")):
            if path.resolve().is_relative_to(custom.resolve()):
                item = _read_json(path)
                if isinstance(item, dict) and item.get("source") == "custom":
                    _safe_id(item["id"])
                    if path.parent.name == item["id"]:
                        entries.append(item)
    seen = set()
    for item in entries:
        _validate_entry(item)
        model_id = _safe_id(item["id"])
        if model_id in seen or item["kind"] not in KINDS:
            raise ValueError("Duplicate model id or invalid kind: " + model_id)
        seen.add(model_id)
    return entries


def refresh_remote_catalog(url=None, timeout=5):
    """Fetch metadata only; reject unsafe entries and atomically replace cache.

    With no configured source this is a no-op. Errors leave the cache intact.
    Returns accepted records. No implicit network access from load_catalog.
    """
    if url is None:
        url = os.environ.get("LOCALDOCS_CATALOG_URL")
        if url is None:
            url = _read_json(Path(CATALOG_PATH).with_name("catalog_source.json"), {}).get("url", "")
    if not url:
        return []
    _https(url)
    with build_opener(_HTTPSRedirect()).open(Request(url), timeout=timeout) as response:
        _https(response.geturl())
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Remote catalog is too large")
    entries = json.loads(data.decode("utf-8-sig"))
    if not isinstance(entries, list):
        raise ValueError("Remote catalog must be a list")
    accepted = {}
    for item in entries:
        try:
            _validate_entry(item, remote=True)
        except (ValueError, KeyError, TypeError, OSError):
            continue
        old = accepted.get(item["id"])
        if old is None or _version(item.get("version", 0)) > _version(old.get("version", 0)):
            accepted[item["id"]] = item
    with _STATE_LOCK:
        _write_json(models_dir() / "remote_catalog.json", list(accepted.values()))
    return list(accepted.values())


def check_all_updates():
    results = {}
    for entry in load_catalog():
        if entry["source"] == "hf" and _status(entry)["installed"]:
            try:
                results[entry["id"]] = check_updates(entry["id"])["update"]
            except Exception:
                results[entry["id"]] = None
    return results


def update_state():
    with _STATE_LOCK:
        return {"checked_at": _UPDATE["checked_at"], "updates": dict(_UPDATE["updates"]),
                "new_models": list(_UPDATE["new_models"])}


def _auto_update_once():
    before = {e["id"] for e in load_catalog()}
    try:
        refresh_remote_catalog()
    except Exception:
        pass  # Offline: still check previously known installed models.
    after = {e["id"] for e in load_catalog()}
    results = check_all_updates()
    with _STATE_LOCK:
        _UPDATE.update(checked_at=datetime.now(timezone.utc).isoformat(),
                       updates=results, new_models=sorted(after - before))


def start_auto_update_check(period_hours=24):
    """Start one opt-in daemon; check immediately, then periodically. No downloads."""
    global _UPDATE_THREAD
    seconds = float(period_hours) * 3600
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("period_hours must be positive and finite")
    def worker():
        while True:
            try:
                _auto_update_once()
            except Exception:
                pass  # A damaged local record must not kill future checks.
            if _UPDATE_WAKE.wait(seconds):
                return
    with _STATE_LOCK:
        if _UPDATE_THREAD is not None and _UPDATE_THREAD.is_alive():
            return False
        _UPDATE_WAKE.clear()
        thread = threading.Thread(target=worker, name="localdocs-model-updates", daemon=True)
        thread.start()
        _UPDATE_THREAD = thread
    return True


def _entry(model_id):
    _safe_id(model_id)
    for entry in load_catalog():
        if entry["id"] == model_id:
            return entry
    raise ValueError("Unknown model: " + model_id)


def _directory(entry):
    if entry["source"] == "bundled":
        return ROOT
    if entry["source"] == "custom":
        return _child(models_dir(), "custom/" + entry["id"])
    return _child(models_dir(), entry["id"])


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_ok(path, spec):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    size = spec.get("size")
    if size is not None and size > 0 and path.stat().st_size != size:
        return False
    expected = spec.get("sha256")
    return not expected or _sha256(path).lower() == expected.lower()


def _active_map():
    value = _read_json(models_dir() / "active.json", {})
    if not isinstance(value, dict):
        raise ValueError("Invalid active.json")
    return value


def _status(entry):
    directory = _directory(entry)
    checks = {spec["path"]: _file_ok(_child(directory, spec["path"]), spec)
              for spec in entry["files"]}
    installed = bool(checks) and all(checks.values())
    size = sum(_child(directory, p).stat().st_size for p in checks
               if _child(directory, p).is_file()) / 1_000_000
    active = _active_map()
    return {"installed": installed, "path": str(directory), "files_ok": checks,
            "size_mb": size, "active_for": [k for k, v in active.items()
                                             if v == entry["id"] and installed]}


def status(model_id):
    return _status(_entry(model_id))


def list_models(kind=None):
    if kind is not None and kind not in KINDS:
        raise ValueError("Invalid model kind")
    return [dict(entry, status=(state := _status(entry)), active=bool(state["active_for"]))
            for entry in load_catalog()
            if kind is None or kind in _supported(entry)]


def _supported(entry):
    return set(entry.get("supported_kinds", [entry["kind"]]))


def set_active(kind, model_id):
    # Probe entitlement: set_active
    import probe_licensing as _pl
    _pl.require('semantic')
    with _STATE_LOCK:
        entry = _entry(model_id)
        if kind not in KINDS or kind not in _supported(entry):
            raise ValueError("Model does not support kind: " + str(kind))
        if entry.get("commercial_ok") is not True:
            raise ValueError("Commercial permission is not confirmed for this model")
        if not _status(entry)["installed"]:
            raise ValueError("Model is not installed: " + model_id)
        active = _active_map()
        active[kind] = model_id
        _write_json(models_dir() / "active.json", active)


def get_active(kind):
    """Return the explicitly selected installed model id, or None."""
    # Probe entitlement: get_active
    import probe_licensing as _pl
    if not _pl.allowed('semantic'):
        return None
    if kind not in KINDS:
        raise ValueError("Invalid model kind")
    with _STATE_LOCK:
        model_id = _active_map().get(kind)
        if model_id:
            try:
                entry = _entry(model_id)
                if (entry.get("commercial_ok") is True and kind in _supported(entry)
                        and _status(entry)["installed"]):
                    return model_id
            except (ValueError, OSError):
                pass
    return None


def _headers(response):
    return {"etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified")}


def _url(spec):
    url = spec.get("url", "")
    if not url.startswith(("http://", "https://")):
        raise ModelDownloadError("No HTTP download URL for " + spec["path"])
    return url


def _stop(should_stop):
    if should_stop is not None and should_stop():
        raise ModelDownloadCancelled("Download cancelled; partial files kept")


def _transfer(spec, target, progress, should_stop, completed, total, prior):
    part = target.with_name(target.name + ".part")
    resume_info = part.with_name(part.name + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    if offset:
        saved = _read_json(resume_info, {})
        if saved.get("url") == _url(spec):
            prior = saved
        elif saved:
            # A catalog URL change must not concatenate unrelated byte streams.
            offset = 0
    # A complete .part may have been left just before atomic replacement.
    if offset and spec.get("sha256") and _file_ok(part, spec):
        _stop(should_stop)
        os.replace(part, target)
        return prior
    headers = {"User-Agent": "Linda-Probe-models/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        validator = prior.get("etag") or prior.get("last_modified")
        if validator:
            headers["If-Range"] = validator
    _stop(should_stop)
    try:
        response = urlopen(Request(_url(spec), headers=headers), timeout=30)
    except HTTPError as exc:
        # A stale/oversized partial file needs a fresh full transfer.
        if exc.code != 416 or not offset:
            raise
        exc.close()
        headers.pop("Range", None)
        headers.pop("If-Range", None)
        response = urlopen(Request(_url(spec), headers=headers), timeout=30)
    with response:
        metadata = _headers(response)
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)",
                                 response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset:
                raise ModelDownloadError("Invalid Range response for " + spec["path"])
            end = int(match[2])
            if end < offset:
                raise ModelDownloadError("Invalid Range length")
            remote_size = int(match[3]) if match[3] != "*" else None
            expected_read = end - offset + 1
            for key in ("etag", "last_modified"):
                if prior.get(key) and metadata.get(key) != prior[key]:
                    raise ModelDownloadError("Remote file changed; discard the stale .part before retrying")
            mode = "ab"
        elif response.status == 200:
            offset, mode = 0, "wb"  # Server ignored Range or If-Range changed.
            length = response.headers.get("Content-Length")
            remote_size = expected_read = int(length) if length else None
        else:
            raise ModelDownloadError("Unexpected HTTP status: " + str(response.status))
        received = 0
        with part.open(mode) as stream:
            _write_json(resume_info, dict(metadata, url=_url(spec)))
            while True:
                _stop(should_stop)
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                stream.write(chunk)
                received += len(chunk)
                if progress:
                    progress({"file": spec["path"], "done_bytes": completed + offset + received,
                              "total_bytes": total})
            stream.flush()
            os.fsync(stream.fileno())
        _stop(should_stop)
        if expected_read is not None and received != expected_read:
            raise ModelDownloadError("Incomplete download: " + spec["path"])
        if remote_size is not None and part.stat().st_size != remote_size:
            raise ModelDownloadError("Incomplete Range download: " + spec["path"])
    if not _file_ok(part, spec):
        # Keep evidence and avoid repeatedly resuming a poisoned complete part.
        os.replace(part, part.with_name(part.name + ".invalid-" + uuid4().hex))
        raise ModelDownloadError("SHA256 or size mismatch: " + spec["path"])
    os.replace(part, target)
    return metadata


def download(model_id, progress=None, should_stop=None):
    """Explicit, serialized, resumable download. Return the resulting status."""
    # Probe entitlement: download
    import probe_licensing as _pl
    _pl.require('semantic')
    with _DOWNLOAD_LOCK:
        entry = _entry(model_id)
        if entry["source"] != "hf":
            if _status(entry)["installed"]:
                return _status(entry)
            raise ModelDownloadError("Bundled/custom model files are missing; no download source")
        directory = _directory(entry)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "model.json"
        manifest = _read_json(manifest_path, {})
        remote = manifest.setdefault("remote", {})
        # Zero means unknown total, not an invented estimate.
        sizes = [spec.get("size") for spec in entry["files"]]
        total = sum(sizes) if all(isinstance(n, int) and n > 0 for n in sizes) else 0
        completed = 0
        try:
            for spec in entry["files"]:
                _stop(should_stop)
                target = _child(directory, spec["path"])
                if not _file_ok(target, spec):
                    remote[spec["path"]] = _transfer(
                        spec, target, progress, should_stop, completed, total,
                        remote.get(spec["path"], {}))
                    manifest.update(id=model_id, downloaded_at=datetime.now(timezone.utc).isoformat())
                    _write_json(manifest_path, manifest)
                completed += target.stat().st_size
                if progress:
                    progress({"file": spec["path"], "done_bytes": completed, "total_bytes": total})
        except ModelDownloadError:
            raise
        except HTTPError as exc:
            code = exc.code
            exc.close()
            detail = "File not found (404)" if code == 404 else f"HTTP {code}"
            raise ModelDownloadError(detail + "; model: " + model_id) from exc
        except (URLError, OSError, http.client.HTTPException, TimeoutError) as exc:
            raise ModelDownloadError("Network or file error downloading " + model_id + ": " + str(exc)) from exc
        return _status(entry)


def add_custom(kind, src_path_or_dir, title, device_hint="cpu"):
    """Copy a local ONNX set; unknown user licensing prevents activation.

    Returns the new model id. If available, ONNX Runtime opens every graph with
    CPU only to record IO; no inference is run. External-data ONNX is rejected
    unless its dependencies are among the selected files (session validation).
    """
    # Probe entitlement: add_custom
    import probe_licensing as _pl
    _pl.require('semantic')
    if kind not in KINDS:
        raise ValueError("Invalid model kind")
    devices = [device_hint] if isinstance(device_hint, str) else list(device_hint)
    if not devices or any(d not in ("cpu", "gpu") for d in devices):
        raise ValueError("device_hint must contain cpu and/or gpu")
    source = Path(src_path_or_dir).expanduser().resolve(strict=True)
    if source.is_dir():
        selected = sorted(p for p in source.iterdir() if p.is_file() and
                          (p.suffix.lower() == ".onnx" or
                           (kind == "ocr_rec" and p.suffix.lower() == ".txt") or
                           (kind == "image_embed" and p.name == "tokenizer.json")))
    elif source.suffix.lower() == ".onnx":
        selected = [source]
        if kind == "ocr_rec":
            sidecars = sorted(source.parent.glob("*.txt"))
            if source.with_suffix(".txt").is_file():
                sidecars = [source.with_suffix(".txt")]
            elif (source.parent / "charset.txt").is_file():
                sidecars = [source.parent / "charset.txt"]
            if len(sidecars) != 1:
                raise ValueError("Select a directory or provide one unambiguous OCR dictionary")
            selected += sidecars
    else:
        raise ValueError("Select an ONNX file or model directory")
    graphs = [p for p in selected if p.suffix.lower() == ".onnx"]
    if not graphs or (kind == "ocr_rec" and not any(p.suffix.lower() == ".txt" for p in selected)):
        raise ValueError("ONNX model required; OCR recognition also needs a .txt dictionary")
    if any(p.stat().st_size == 0 for p in selected):
        raise ValueError("Model files must not be empty")
    label = title if isinstance(title, str) else title.get("en", "custom")
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:48] or "model"
    model_id = "custom-" + slug + "-" + uuid4().hex[:12]
    directory = _child(models_dir(), "custom/" + model_id)
    staging = _child(models_dir(), "custom/.staging-" + uuid4().hex)
    staging.mkdir(parents=True)
    try:
        files = []
        for path in selected:
            destination = _child(staging, path.name)
            shutil.copy2(path, destination)
            files.append({"path": path.name, "url": "", "size": destination.stat().st_size,
                          "sha256": _sha256(destination)})
        try:
            ort = importlib.import_module("onnxruntime")
        except ModuleNotFoundError as exc:
            if exc.name != "onnxruntime":
                raise
            ort = None
        io_info = {}
        if ort is not None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            for graph in graphs:
                session = ort.InferenceSession(str(staging / graph.name), sess_options=options,
                                              providers=["CPUExecutionProvider"])
                describe = lambda nodes: [{"name": n.name, "shape": n.shape, "type": n.type} for n in nodes]
                io_info[graph.name] = {"inputs": describe(session.get_inputs()),
                                       "outputs": describe(session.get_outputs())}
                del session
        titles = {lang: title for lang in ("ru", "en", "pl")} if isinstance(title, str) else dict(title)
        entry = {"id": model_id, "kind": kind, "title": titles,
                 "description": {lang: "User-supplied local model" for lang in ("ru", "en", "pl")},
                 "device": devices, "source": "custom", "license": "user-supplied",
                 "commercial_ok": "unknown", "files": files, "default_for": [],
                 "size_mb": sum(f["size"] for f in files) / 1_000_000,
                 "repo": "", "revision": "", "added_at": datetime.now(timezone.utc).isoformat(),
                 "compatibility": {"check": "onnxruntime" if ort else "extension-only", "graphs": io_info}}
        _write_json(staging / "model.json", entry)
        with _STATE_LOCK:
            os.replace(staging, directory)
    except Exception:
        # Staging is never registered; preserve it for diagnosis instead of deleting user data.
        raise
    return model_id


def remove_custom(model_id):
    """Delete only this installed custom copy, never the original source files."""
    with _STATE_LOCK:
        entry = _entry(model_id)
        if entry["source"] != "custom":
            raise ValueError("Only custom models may be removed")
        directory = _directory(entry)
        custom_root = models_dir() / "custom"
        literal = custom_root / model_id
        if (custom_root.is_symlink() or custom_root.is_junction()
                or literal.is_symlink() or literal.is_junction()
                or directory.parent != custom_root.resolve()):
            raise ValueError("Refusing to remove a linked or escaped custom directory")
        for path in directory.rglob("*"):
            if path.is_symlink() or path.is_junction():
                raise ValueError("Refusing to remove a custom directory containing links")
        active = {k: v for k, v in _active_map().items() if v != model_id}
        _write_json(models_dir() / "active.json", active)
        shutil.rmtree(directory)


def check_updates(model_id):
    entry = _entry(model_id)
    if entry["source"] != "hf" or not entry["files"]:
        return {"update": None}
    spec = entry["files"][0]
    try:
        request = Request(_url(spec), method="HEAD", headers={"User-Agent": "Linda-Probe-models/1"})
        with urlopen(request, timeout=5) as response:
            current = _headers(response)
        manifest = _read_json(_directory(entry) / "model.json", {})
        saved = manifest.get("remote", {}).get(spec["path"], {})
        key = next((k for k in ("etag", "last_modified") if current.get(k) and saved.get(k)), None)
        return {"update": current[key] != saved[key] if key else None,
                "remote_etag": current.get("etag") or current.get("last_modified")}
    except (URLError, OSError, http.client.HTTPException, ValueError, ModelDownloadError):
        return {"update": None}


def download_status():
    with _JOB_LOCK:
        return dict(_JOB)


def start_download(model_id):
    """Start the singleton daemon worker; False if another download is running."""
    # Probe entitlement: start_download
    import probe_licensing as _pl
    _pl.require('semantic')
    _entry(model_id)
    with _JOB_LOCK:
        if _JOB["phase"] == "downloading":
            return False
        _JOB.update(phase="downloading", model_id=model_id, file=None,
                    done_bytes=0, total_bytes=0, error=None)

    def report(value):
        with _JOB_LOCK:
            _JOB.update(value)

    def worker():
        try:
            download(model_id, progress=report)
        except Exception as exc:
            with _JOB_LOCK:
                _JOB.update(phase="error", error=str(exc))
        else:
            with _JOB_LOCK:
                _JOB.update(phase="done")

    try:
        threading.Thread(target=worker, name="localdocs-model-download", daemon=True).start()
    except Exception as exc:
        with _JOB_LOCK:
            _JOB.update(phase="error", error=str(exc))
        return False
    return True
