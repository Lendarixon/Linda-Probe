"""Build the optional .NET Framework worker without opening a console."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading

_LOCK = threading.Lock()
_LDENG_LOCK = threading.Lock()

LAST_ERROR = ""


def compiler():
    for framework in ('Framework64', 'Framework'):
        path = os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Microsoft.NET',
                            framework, 'v4.0.30319', 'csc.exe')
        if os.path.isfile(path):
            return path
    return None


def build(force=False):
    """Return an up-to-date executable or None; never raise."""
    try:
        with _LOCK:
            folder = os.path.dirname(os.path.abspath(__file__))
            source, target = [os.path.join(folder, 'ldname.' + ext) for ext in ('cs', 'exe')]
            if not force and os.path.isfile(target) and os.path.getmtime(target) >= os.path.getmtime(source):
                return target
            csc = compiler()
            if not csc:
                return None
            result = subprocess.run([csc, '-nologo', '-unsafe', '-optimize+', '-platform:x64',
                                     '-out:' + target, source], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return target if result.returncode == 0 and os.path.isfile(target) else None
    except Exception:
        return None


def _default_ldeng_dir():
    try:
        local = os.environ.get('LOCALAPPDATA')
        if local:
            return os.path.join(local, 'LocalDocs', 'native')
    except Exception:
        pass
    return os.path.join(tempfile.gettempdir(), 'localdocs-native')


def _csc_identity(csc):
    try:
        st = os.stat(csc)
        return '%s|%d|%d' % (csc, st.st_size, int(st.st_mtime))
    except Exception:
        return str(csc)


def build_ldeng(dest_dir=None, force=False, source=None):
    """Compile native/ldeng.cs into a per-user cache dir; never raise.

    Returns the versioned exe path (ldeng-<hash8>.exe) or None. The hash
    covers the SHA-256 of the source plus the csc path/version, recorded
    in a small json sidecar so rebuilds happen only when the source (or
    compiler) changed. Compiles to a temp name then os.replace for atomic
    replacement; versioned names tolerate a running old exe holding its file.
    Failures return None with the reason in module-level LAST_ERROR.
    """
    global LAST_ERROR
    try:
        with _LDENG_LOCK:
            folder = os.path.dirname(os.path.abspath(__file__))
            if source is None:
                source = os.path.join(folder, 'ldeng.cs')
            else:
                try:
                    source = os.fspath(source)
                except Exception:
                    source = str(source)
            if dest_dir is None:
                dest_dir = _default_ldeng_dir()
            else:
                try:
                    dest_dir = os.fspath(dest_dir)
                except Exception:
                    dest_dir = str(dest_dir)
            if not os.path.isfile(source):
                LAST_ERROR = 'source not found: %s' % (source,)
                return None
            try:
                with open(source, 'rb') as stream:
                    src_bytes = stream.read()
            except Exception as error:
                LAST_ERROR = 'cannot read source: %s' % (error,)
                return None
            src_hash = hashlib.sha256(src_bytes).hexdigest()
            csc = compiler()
            if not csc:
                LAST_ERROR = 'csc unavailable'
                return None
            csc_id = _csc_identity(csc)
            combined = hashlib.sha256((src_hash + '|' + csc_id).encode('utf-8')).hexdigest()
            hash8 = combined[:8]
            exe_name = 'ldeng-%s.exe' % (hash8,)
            try:
                os.makedirs(dest_dir, exist_ok=True)
            except Exception as error:
                LAST_ERROR = 'cannot create dest dir: %s' % (error,)
                return None
            target = os.path.join(dest_dir, exe_name)
            sidecar = os.path.join(dest_dir, 'ldeng.json')
            if not force and os.path.isfile(target):
                try:
                    if os.path.isfile(sidecar):
                        with open(sidecar, 'r', encoding='utf-8') as stream:
                            info = json.load(stream)
                        if (info.get('hash8') == hash8 and info.get('exe') == exe_name
                                and info.get('source_sha256') == src_hash):
                            LAST_ERROR = ''
                            return target
                        # Name already encodes the current hash, so an existing
                        # target is valid even if the sidecar lags; refresh it.
                        try:
                            with open(sidecar, 'w', encoding='utf-8') as stream:
                                json.dump({'source_sha256': src_hash, 'csc': csc,
                                           'csc_id': csc_id, 'hash': combined,
                                           'hash8': hash8, 'exe': exe_name}, stream)
                        except Exception:
                            pass
                        LAST_ERROR = ''
                        return target
                    LAST_ERROR = ''
                    return target
                except Exception:
                    LAST_ERROR = ''
                    return target
            tmp = os.path.join(dest_dir, 'ldeng-%s.tmp-%d-%d.exe'
                               % (hash8, os.getpid(), threading.get_ident() & 0xFFFFFFFF))
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            try:
                result = subprocess.run(
                    [csc, '-nologo', '-unsafe', '-optimize+', '-platform:x64',
                     '-out:' + tmp, source],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except subprocess.TimeoutExpired:
                LAST_ERROR = 'csc timed out after 120 s'
                return None
            except Exception as error:
                LAST_ERROR = 'csc launch failed: %s' % (error,)
                return None
            if result.returncode != 0 or not os.path.isfile(tmp):
                try:
                    detail = result.stderr.decode(errors='replace') or result.stdout.decode(errors='replace')
                except Exception:
                    detail = ''
                detail = (detail or '').strip()[:500]
                LAST_ERROR = 'csc failed: %s' % (detail or ('exit %d' % result.returncode),)
                try:
                    if os.path.isfile(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                return None
            try:
                os.replace(tmp, target)
            except OSError as error:
                # A running old exe may hold the file; the versioned name
                # normally avoids this, but tolerate it on forced rebuilds.
                if os.path.isfile(target):
                    try:
                        if os.path.isfile(tmp):
                            os.remove(tmp)
                    except Exception:
                        pass
                    LAST_ERROR = ''
                    return target
                LAST_ERROR = 'replace failed: %s' % (error,)
                try:
                    if os.path.isfile(tmp):
                        return tmp
                except Exception:
                    pass
                return None
            try:
                with open(sidecar, 'w', encoding='utf-8') as stream:
                    json.dump({'source_sha256': src_hash, 'csc': csc,
                               'csc_id': csc_id, 'hash': combined,
                               'hash8': hash8, 'exe': exe_name}, stream)
            except Exception:
                pass
            LAST_ERROR = ''
            return target
    except Exception as error:
        try:
            LAST_ERROR = 'build_ldeng failed: %s' % (error,)
        except Exception:
            try:
                LAST_ERROR = 'build_ldeng failed'
            except Exception:
                pass
        return None


def ldeng_status(dest_dir=None, source=None):
    """Return dict(csc=path|None, built=path|None, error=str); never raise."""
    try:
        try:
            csc = compiler()
        except Exception:
            csc = None
        built = None
        try:
            folder = os.path.dirname(os.path.abspath(__file__))
            src = os.path.join(folder, 'ldeng.cs') if source is None else str(source)
            out = _default_ldeng_dir() if dest_dir is None else str(dest_dir)
            if csc and os.path.isfile(src):
                with open(src, 'rb') as stream:
                    src_hash = hashlib.sha256(stream.read()).hexdigest()
                combined = hashlib.sha256((src_hash + '|' + _csc_identity(csc)).encode('utf-8')).hexdigest()
                candidate = os.path.join(out, 'ldeng-%s.exe' % (combined[:8],))
                if os.path.isfile(candidate):
                    built = candidate
            if built is None:
                # Fall back to the sidecar so a previous build is visible
                # even when csc is currently missing.
                sidecar = os.path.join(out, 'ldeng.json')
                if os.path.isfile(sidecar):
                    with open(sidecar, 'r', encoding='utf-8') as stream:
                        info = json.load(stream)
                    candidate = os.path.join(out, str(info.get('exe', '')))
                    if os.path.isfile(candidate):
                        built = candidate
        except Exception:
            pass
        try:
            error = LAST_ERROR if isinstance(LAST_ERROR, str) else str(LAST_ERROR)
        except Exception:
            error = ''
        return {'csc': csc, 'built': built, 'error': error}
    except Exception as error:
        try:
            return {'csc': None, 'built': None, 'error': str(error)}
        except Exception:
            return {'csc': None, 'built': None, 'error': 'unknown'}


if __name__ == '__main__':
    if '--ldeng' in sys.argv[1:]:
        print(json.dumps(ldeng_status()))
    else:
        print(build())
