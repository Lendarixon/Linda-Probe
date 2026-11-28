"""Linda Probe entitlements. No merchant token; only Polar public client APIs.

Paid cache is bound to this Windows user/device using DPAPI. Signed gifts are
offline, perpetual Personal licences (no online device-count enforcement).
"""
from __future__ import annotations
import base64
import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
import urllib.error
import uuid

ROOT = Path(__file__).resolve().parent
LOCK = threading.RLock()
FEATURES = ('semantic', 'ocr', 'photos')
TRIAL_SECONDS = 7 * 86400

class LicenseError(RuntimeError):
    def __init__(self, code):
        super().__init__(code); self.code = code

def config():
    return json.loads((ROOT / 'probe_licensing_config.json').read_text(encoding='utf-8-sig'))

def data_dir():
    override = os.environ.get('LOCALDOCS_DATA_DIR') or os.environ.get('LOCALDOCS_WRITE_DIR')
    base = Path(override) if override else Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'LocalDocs'
    base.mkdir(parents=True, exist_ok=True)
    return base

def machine_id():
    guid = ''
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Cryptography', 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            guid = str(winreg.QueryValueEx(key, 'MachineGuid')[0])
    else:
        guid = str(uuid.getnode())
    return hashlib.sha256(('linda-probe-v1|' + guid).encode()).hexdigest()

def _protect(raw, decrypt=False):
    if os.name != 'nt':
        raise LicenseError('windows_required')
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_byte))]
    buf = ctypes.create_string_buffer(raw)
    inp = Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte))); out = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]; kernel.LocalFree.restype = ctypes.c_void_p
    if not fn(ctypes.byref(inp), None, None, None, None, 1, ctypes.byref(out)):
        raise LicenseError('cache_invalid')
    try: return ctypes.string_at(out.data, out.size)
    finally: kernel.LocalFree(out.data)

def _cache_valid(st):
    if not isinstance(st, dict): return False
    strings = ('key', 'kind', 'status', 'tier', 'machine', 'benefit_id', 'activation_id', 'gift_id')
    if any(k in st and not isinstance(st[k], str) for k in strings): return False
    if 'expires_at' in st and st['expires_at'] is not None and not isinstance(st['expires_at'], str): return False
    for k in ('last_validated', 'last_seen', 'started_at'):
        if k in st and (type(st[k]) not in (int, float) or not 0 <= st[k] <= 253402300799 or not math.isfinite(st[k])): return False
    if not st: return True
    if any(not st.get(k) for k in ('status', 'tier', 'machine')): return False
    if st.get('kind') == 'trial':
        start = st.get('started_at')
        return (st['tier'] == 'personal' and not st.get('key') and start is not None
            and _expiry(st.get('expires_at')) == start + TRIAL_SECONDS)
    if not st.get('key'): return False
    if st.get('kind') == 'promo': return st['tier'] == 'personal' and bool(st.get('expires_at'))
    if st.get('kind') == 'gift': return st['tier'] == 'personal'
    return (st.get('kind') == 'purchase' and st['tier'] in ('personal', 'organization')
        and bool(st.get('benefit_id')) and bool(st.get('activation_id')))

def _rollback(st):
    # Unix timestamps are UTC; tolerate small clock corrections.
    timed = st.get('tier') == 'organization' or st.get('kind') in ('trial', 'promo')
    return timed and (time.time() < st.get('last_seen', 0) - 300
        or (st.get('kind') == 'trial' and time.time() < st.get('started_at', 0)))

def _load():
    try:
        raw = (data_dir() / 'probe_license.dat').read_bytes()
        st = json.loads(_protect(raw, True))
        return st if _cache_valid(st) else {}
    except (OSError, ValueError, TypeError, OverflowError, RecursionError, LicenseError): return {}

def _save(state):
    seen = max(time.time(), state.get('last_seen', 0)) if state.get('kind') in ('trial', 'promo') else time.time()
    state = dict(state, last_seen=seen)
    if not _cache_valid(state): raise LicenseError('cache_invalid')
    target = data_dir() / 'probe_license.dat'
    tmp = target.with_suffix('.tmp')
    tmp.write_bytes(_protect(json.dumps(state).encode()))
    os.replace(tmp, target)

def _trial_marker(raw=None):
    import winreg
    name = 'TrialV1_' + machine_id()
    if raw is None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\LindaProbe', 0, winreg.KEY_QUERY_VALUE) as key:
                return winreg.QueryValueEx(key, name)[0]
        except FileNotFoundError: return None
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, r'Software\LindaProbe', 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_BINARY, raw)

def _trial_used():
    try: return (data_dir() / 'probe_trial.dat').exists() or _trial_marker() is not None
    except (OSError, ImportError): return True

def start_trial():
    with LOCK:
        old = _load()
        if old.get('kind') == 'trial' or _trial_used(): raise LicenseError('trial_used')
        if _entitled(old): raise LicenseError('already_licensed')
        now = int(time.time())
        st = {'kind': 'trial', 'tier': 'personal', 'status': 'granted', 'machine': machine_id(),
            'started_at': now, 'expires_at': datetime.fromtimestamp(now + TRIAL_SECONDS, timezone.utc).isoformat()}
        try:
            raw = _protect(json.dumps(st).encode())
            # Record use before granting Pro; a partial write cannot grant a fresh trial.
            _trial_marker(raw)
            (data_dir() / 'probe_trial.dat').write_bytes(raw)
            _save(st)
        except (OSError, ImportError, LicenseError): raise LicenseError('cache_invalid') from None
        return public_state()

def _response(action, value):
    if not isinstance(value, dict): raise LicenseError('license_server')
    if action == 'validate':
        value = value.get('license_key', value)
        if not isinstance(value, dict): raise LicenseError('license_server')
        if any(not isinstance(value.get(k), str) or not value[k] for k in ('status', 'benefit_id')):
            raise LicenseError('license_server')
        if value.get('expires_at') is not None and not isinstance(value['expires_at'], str):
            raise LicenseError('license_server')
    elif action == 'activate':
        if not isinstance(value.get('id'), str) or not value['id']: raise LicenseError('license_server')
    return value

def _post(action, body):
    req = urllib.request.Request('https://api.polar.sh/v1/customer-portal/license-keys/' + action,
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0 LindaProbe/1.4.1'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status == 204: return _response(action, {})
            result = json.load(r)
            _response(action, result)
            return result
    except urllib.error.HTTPError as e:
        if action == 'deactivate' and e.code == 404: return {}
        # Never put a raw API response (may contain licence/customer data) in logs/UI.
        raise LicenseError('activation_limit' if e.code == 403 else 'invalid_key' if e.code in (400, 404, 422) else 'license_server') from None
    except (OSError, urllib.error.URLError): raise LicenseError('license_offline') from None
    except (ValueError, TypeError, OverflowError, RecursionError): raise LicenseError('license_server') from None

def verify_gift(key):
    match = re.fullmatch(r'PROBE-GIFT-P-(\d{4})-([A-Z2-7]{103})', key)
    if not match: return None
    number, signature = match.groups()
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        public = base64.b64decode(config()['gift_public_key'])
        decoded = base64.b32decode(signature + '=' * (-len(signature) % 8))
        if base64.b32encode(decoded).decode().rstrip('=') != signature: return None
        Ed25519PublicKey.from_public_bytes(public).verify(decoded, ('linda-probe-gift|P|' + number).encode())
    except Exception: return None
    return {'kind': 'gift', 'tier': 'personal', 'gift_id': number, 'status': 'granted'}

def verify_promo(key):
    match = re.fullmatch(r'PROBE-PROMO-P-([0-9]{4})-([0-9]{8})-([A-Z2-7]{103})', key)
    if not match: return None
    number, date, signature = match.groups()
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        end = datetime.strptime(date, '%Y%m%d').replace(tzinfo=timezone.utc) + timedelta(days=1)
        decoded = base64.b32decode(signature + '=' * (-len(signature) % 8))
        if base64.b32encode(decoded).decode().rstrip('=') != signature: return None
        public = base64.b64decode(config()['gift_public_key'])
        Ed25519PublicKey.from_public_bytes(public).verify(decoded, f'linda-probe-promo|P|{number}|{date}'.encode())
    except Exception: return None
    return {'kind': 'promo', 'tier': 'personal', 'status': 'granted', 'expires_at': end.isoformat()}

def _offline_key(key):
    grant = verify_gift(key) or verify_promo(key)
    if grant and grant['kind'] == 'promo':
        old = _load()
        if _expiry(grant['expires_at']) <= time.time() or (old.get('key') == key
                and (_rollback(old) or old.get('status') == 'expired')): raise LicenseError('expired_key')
    return grant

def _expiry(value):
    if value is None: return None
    if not isinstance(value, str) or not value: return 0
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return dt.replace(tzinfo=timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()
    except (ValueError, OverflowError, OSError): return 0

def _entitled(st):
    if not _cache_valid(st): return False
    if st.get('status') != 'granted' or st.get('machine') != machine_id(): return False
    if st.get('kind') == 'gift': return verify_gift(st.get('key', '')) is not None
    if st.get('kind') in ('trial', 'promo'):
        if _rollback(st): return False
        if st['kind'] == 'promo':
            grant = verify_promo(st['key'])
            if not grant or grant['expires_at'] != st.get('expires_at'): return False
        return (_expiry(st.get('expires_at')) or 0) > time.time()
    if st.get('benefit_id') not in config()['benefits'].values() or not st.get('activation_id'): return False
    if config()['benefits'].get(st.get('tier')) != st.get('benefit_id') or _rollback(st): return False
    expiry = _expiry(st.get('expires_at'))
    if st.get('tier') == 'organization' and not expiry: return False
    return expiry is None or expiry > time.time()

def public_state():
    with LOCK:
        st = _load(); active = _entitled(st)
        timed = st.get('kind') in ('trial', 'promo')
        if timed and st.get('status') == 'granted' and (_rollback(st) or (_expiry(st.get('expires_at')) or 0) <= time.time()):
            st['status'] = 'expired'
            try: _save(st)
            except (OSError, LicenseError): pass
        days = max(0, math.ceil(((_expiry(st.get('expires_at')) or 0) - time.time()) / 86400)) if active and timed else 0
        return {'licensed': active, 'edition': 'Pro' if active else 'Free', 'tier': st.get('tier') if active else None,
            'kind': st.get('kind') if active or timed else None, 'status': st.get('status', 'free') if active else 'free',
            'key_hint': (st.get('key', '')[:6] + '...' + st.get('key', '')[-4:]) if st.get('key') else '',
            'expires_at': st.get('expires_at'), 'days_left': days,
            'trial_available': not active and st.get('kind') != 'trial' and not _trial_used(),
            'features': {f: active for f in FEATURES}, 'checkouts': config()['checkouts']}

def allowed(feature):
    return feature not in FEATURES or public_state()['licensed']

def check_key(key):
    """Read-only installer preflight: validate without using an activation slot."""
    if not isinstance(key, str) or len(key) > 512: raise LicenseError('invalid_key')
    key = re.sub(r'\s+', '', key).upper()
    if not key: raise LicenseError('empty_key')
    grant = _offline_key(key)
    if grant:
        result = {'ok': True, 'tier': grant['tier'], 'kind': grant['kind']}
        if grant['kind'] == 'promo': result['expires_at'] = grant['expires_at']
        return result
    if not key.startswith('PROBE-') or key.startswith(('PROBE-GIFT', 'PROBE-PROMO')): raise LicenseError('wrong_product')
    cfg = config()
    lk = _post('validate', {'key': key, 'organization_id': cfg['organization_id']})
    lk = _response('validate', lk)
    tier = next((tier for tier, bid in cfg['benefits'].items() if bid == lk.get('benefit_id')), None)
    if not tier: raise LicenseError('wrong_product')
    if lk.get('status') != 'granted': raise LicenseError('invalid_key')
    expiry = _expiry(lk.get('expires_at'))
    if expiry is not None and expiry <= time.time(): raise LicenseError('expired_key')
    if tier == 'organization' and expiry is None: raise LicenseError('expired_key')
    return {'ok': True, 'tier': tier, 'kind': 'purchase'}

def require(feature):
    if not allowed(feature): raise LicenseError('license_required')

def activate(key):
    if not isinstance(key, str) or len(key) > 512: raise LicenseError('invalid_key')
    key = re.sub(r'\s+', '', key).upper()
    if not key: raise LicenseError('empty_key')
    with LOCK:
        created_activation = None
        old = _load()
        if old.get('key') == key and _entitled(old): return public_state()
        if old.get('key') and old.get('key') != key and _entitled(old): raise LicenseError('remove_key_first')
        grant = _offline_key(key)
        if grant:
            st = dict(grant, key=key, machine=machine_id(), last_validated=time.time())
        else:
            if not key.startswith('PROBE-') or key.startswith(('PROBE-GIFT', 'PROBE-PROMO')): raise LicenseError('wrong_product')
            cfg = config(); org = cfg['organization_id']
            validated = _post('validate', {'key': key, 'organization_id': org})
            lk = _response('validate', validated)
            bid = lk.get('benefit_id')
            tier = next((t for t, b in cfg['benefits'].items() if bid == b), None)
            if not tier: raise LicenseError('wrong_product')
            if lk.get('status') != 'granted': raise LicenseError('invalid_key')
            expiry = _expiry(lk.get('expires_at'))
            if expiry is not None and expiry <= time.time(): raise LicenseError('expired_key')
            if tier == 'organization' and expiry is None: raise LicenseError('expired_key')
            if old.get('key') and old.get('key') != key and _entitled(old): raise LicenseError('remove_key_first')
            # Reuse an existing machine-bound activation after a same-user reinstall.
            prior = old.get('activation_id') if old.get('key') == key else None
            if prior:
                checked = _response('validate', _post('validate', {'key': key, 'organization_id': org, 'activation_id': prior, 'conditions': {'machine': machine_id()}, 'benefit_id': bid}))
                if checked.get('status') != 'granted' or checked.get('benefit_id') != bid: raise LicenseError('invalid_key')
                expiry = _expiry(checked.get('expires_at'))
                if (expiry is not None and expiry <= time.time()) or (tier == 'organization' and expiry is None): raise LicenseError('expired_key')
                lk = checked
                act_id = prior
            else:
                act = _post('activate', {'key': key, 'organization_id': org, 'label': 'Linda Probe Windows', 'conditions': {'machine': machine_id()}})
                act_id = _response('activate', act)['id']
                created_activation = act_id
            st = {'key': key, 'kind': 'purchase', 'status': 'granted', 'tier': tier, 'benefit_id': bid,
                'activation_id': act_id, 'machine': machine_id(), 'expires_at': lk.get('expires_at'), 'last_validated': time.time()}
        try:
            _save(st)
        except Exception:
            if created_activation:
                try:
                    _post('deactivate', {'key': key, 'organization_id': org, 'activation_id': created_activation})
                except Exception:
                    pass  # Preserve the cache error even when cleanup is offline.
            raise LicenseError('cache_invalid') from None
        return public_state()

def deactivate():
    with LOCK:
        st = _load()
        if st.get('kind') == 'purchase' and st.get('activation_id'):
            _post('deactivate', {'key': st['key'], 'organization_id': config()['organization_id'], 'activation_id': st['activation_id']})
        (data_dir() / 'probe_license.dat').unlink(missing_ok=True)
        return public_state()

def revalidate():
    with LOCK:
        st = _load()
        if not st: return public_state()
        if st.get('kind') in ('trial', 'promo'):
            if _rollback(st) or (_expiry(st.get('expires_at')) or 0) <= time.time(): st['status'] = 'expired'
            _save(st)
            return public_state()
        if st.get('kind') != 'purchase' or (not _rollback(st) and time.time() - st.get('last_validated', 0) < 86400):
            _save(st)
            return public_state()
        try:
            lk = _post('validate', {'key': st['key'], 'organization_id': config()['organization_id'],
                'activation_id': st.get('activation_id'), 'benefit_id': st.get('benefit_id'), 'conditions': {'machine': machine_id()}})
            lk = _response('validate', lk)
            if lk['benefit_id'] != st.get('benefit_id'): raise LicenseError('wrong_product')
            st['status'] = lk.get('status', 'invalid')
            st['expires_at'] = lk.get('expires_at')
            st['last_validated'] = time.time(); _save(st)
        except LicenseError as e:
            if e.code not in ('license_offline', 'license_server'):
                st['status'] = 'revoked'; _save(st)
        return public_state()

def startup():
    """Background validation only; imports never activate, download, or call HTTP."""
    revalidate()

def route_feature(path, body):
    if path.startswith('/ocr-'): return 'ocr'
    if path in ('/img-search', '/img-index'): return 'photos'
    if path == '/img-tags' and body.get('action', 'list') in ('auto', 'autotag', 'classify', 'suggest', 'scan', 'run', 'run_paths'): return 'photos'
    if path in ('/search', '/search_all') and body.get('semantic'): return 'semantic'
    if path == '/models' and body.get('action') in ('download', 'set_active', 'add_custom'): return 'semantic'  # all catalog models are Pro
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--installer', type=Path); parser.add_argument('--output', type=Path); parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); result = {'ok': False}
    try:
        if args.installer:
            data = json.loads(args.installer.read_text(encoding='utf-8-sig'))
            result = check_key(data.get('key', '')) if args.check_only else dict(activate(data.get('key', '')), ok=True)
            if data.get('prepare') and not args.check_only:
                import probe_packs
                try:
                    probe_packs.start(data.get('packs'))
                    while probe_packs.status()['phase'] == 'downloading': time.sleep(.3)
                    result['packs'] = probe_packs.status()
                except Exception:
                    result['packs'] = {'phase': 'error', 'error': 'pack_preparation_failed'}
                if result['packs'].get('phase') != 'done':
                    result.update(ok=False, code='models_incomplete')
        else: result = public_state()
    except LicenseError as e: result = {'ok': False, 'code': e.code}
    finally:
        if args.installer: args.installer.unlink(missing_ok=True)
    if args.output: args.output.write_text(json.dumps(result), encoding='utf-8')
    else: print(json.dumps(result))
    if result.get('code') == 'models_incomplete' and result.get('licensed'): return 2
    return 0 if result.get('ok', result.get('licensed')) else 1

if __name__ == '__main__':
    raise SystemExit(main())
