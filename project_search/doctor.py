"""Environment diagnostics for LocalDocs (no heavy imports)."""
import importlib.util
import os
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8798
DISK_WARN_GB = 5
PKGS = ["faiss", "numpy", "sentence_transformers", "torch",
        "pypdf", "pypdfium2", "docx", "cryptography"]
TESS_LANGS = ("rus", "eng", "pol")
TESS_BIN_CANDIDATES = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path("/usr/bin/tesseract"),
    Path("/usr/local/bin/tesseract"),
]


def _check(name, ok, detail, hint=""):
    return {"name": name, "ok": ok, "detail": detail, "hint": hint}


def _model_name():
    try:
        from project_search.backend import MODEL_NAME
        return MODEL_NAME
    except Exception:
        return os.environ.get("LOCALDOCS_MODEL", "intfloat/multilingual-e5-small")


def _hf_hub_dir():
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            p = Path(v)
            return p
    if os.environ.get('HF_HOME'):
        return Path(os.environ['HF_HOME']) / 'hub'
    return Path.home() / ".cache" / "huggingface" / "hub"


def _tesseract_exe():
    found = shutil.which("tesseract")
    if found:
        return found
    for cand in TESS_BIN_CANDIDATES:
        if cand.exists():
            return str(cand)
    return None


def _tessdata_dirs(exe):
    dirs = []
    v = os.environ.get("TESSDATA_PREFIX")
    if v:
        p = Path(v)
        dirs.append(p if p.name == "tessdata" else p / "tessdata")
    if exe:
        dirs.append(Path(exe).parent / "tessdata")
    dirs.append(ROOT / "tessdata")
    return dirs


def _writable(d):
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".doctor_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, f"writable: {d}"
    except OSError as e:
        return False, f"not writable: {d} ({e})"


def run_checks():
    checks = []

    vi = sys.version_info
    checks.append(_check(
        "python_version", True if vi >= (3, 10) else False,
        f"Python {sys.version.split()[0]}",
        "" if vi >= (3, 10) else "Use Python 3.10+"))

    for pkg in PKGS:
        found = importlib.util.find_spec(pkg) is not None
        checks.append(_check(
            f"pkg:{pkg}", True if found else False,
            "installed" if found else "not found (find_spec)",
            "" if found else f"pip install {pkg}"))

    exe = _tesseract_exe()
    checks.append(_check(
        "tesseract", True if exe else "warn",
        f"tesseract at {exe}" if exe else "tesseract binary not found",
        "" if exe else "Install Tesseract OCR and add it to PATH"))

    dirs = _tessdata_dirs(exe)
    have = {lang for lang in TESS_LANGS
            if any((d / f"{lang}.traineddata").exists() for d in dirs)}
    missing = [lang for lang in TESS_LANGS if lang not in have]
    if not exe:
        ok, detail = "warn", "no tesseract; optional OCR unavailable, text documents remain supported"
    elif not missing:
        ok, detail = True, f"langs rus/eng/pol found in {next((str(d) for d in dirs if (d / 'rus.traineddata').exists()), '?')}"
    else:
        ok, detail = "warn", f"missing langs: {', '.join(missing)} (searched: {'; '.join(map(str, dirs))})"
    checks.append(_check(
        "tesseract_langs", ok, detail,
        "" if ok is True else "Install rus/eng/pol traineddata into TESSDATA_PREFIX or ./tessdata"))

    model = _model_name()
    mdir = _hf_hub_dir() / ("models--" + model.replace("/", "--"))
    snaps = mdir / "snapshots"
    try:
        cached = snaps.is_dir() and any(snaps.iterdir())
    except OSError:
        cached = False
    local = ROOT / 'models' / model.replace('/', '__')
    if (local / 'config.json').is_file():
        cached = True
        mdir = local
    checks.append(_check(
        f"hf_cache:{model}", True if cached else "warn",
        f"cached at {mdir}" if cached else f"not cached (looked in {mdir})",
        "" if cached else "Run one index/search to download, or set HF_HUB_CACHE"))

    idx = ROOT / "indexes"
    try:
        idx.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(idx).free / (1024 ** 3)
        checks.append(_check(
            "disk:indexes", True if free_gb >= DISK_WARN_GB else "warn",
            f"{free_gb:.1f} GB free on {idx}",
            "" if free_gb >= DISK_WARN_GB else "Free up disk space (<5 GB left)"))
    except OSError as e:
        checks.append(_check("disk:indexes", False, f"disk_usage failed: {e}",
                             "Check that indexes/ exists and is accessible"))

    ok, detail = _writable(ROOT / "indexes")
    checks.append(_check("writable:indexes", ok, detail,
                         "" if ok else "Fix permissions on indexes/"))
    ok, detail = _writable(ROOT / "logs")
    checks.append(_check("writable:logs", ok, detail,
                         "" if ok else "Fix permissions on logs/"))

    key = ROOT / "localdocs_key.pem"
    if key.exists():
        try:
            checks.append(_check("localdocs_key.pem", True,
                                 f"present ({key.stat().st_size} bytes)",
                                 "Restrict file permissions to owner only (chmod 600)"))
        except OSError as e:
            checks.append(_check("localdocs_key.pem", "warn", f"present but stat failed: {e}",
                                 "Check file permissions"))
    else:
        checks.append(_check("localdocs_key.pem", "warn", f"missing: {key}",
                             "Optional HTTPS key; generate with make_cert.py only when enabling HTTPS"))

    s = socket.socket()
    try:
        s.bind(("127.0.0.1", PORT))
        checks.append(_check(f"port:{PORT}", True, f"port {PORT} is free", ""))
    except OSError:
        checks.append(_check(f"port:{PORT}", "warn", f"port {PORT} is busy",
                             "Stop the other instance or change the port"))
    finally:
        s.close()

    return checks
