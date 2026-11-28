"""Linda Probe: pythonw linda_probe_app.py [--port 8798]."""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import winreg


ROOT = Path(__file__).resolve().parent
HOME = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'LocalDocs'
STATE = HOME / 'window.json'
TITLE = 'Linda Probe'
LOGS = Path(os.environ.get('LOCALDOCS_WRITE_DIR') or ROOT) / 'logs'


def message(text, title=TITLE, error=False):
    ctypes.windll.user32.MessageBoxW(None, text, title, 0x10 if error else 0x40)


def acquire_window():
    # HANDLE требует явного типа на 64-битной Windows.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.CreateMutexW(None, False, 'LindaProbeWindow')
    error = ctypes.get_last_error()
    if not handle:
        raise ctypes.WinError(error)
    if error != 183:  # ERROR_ALREADY_EXISTS
        return kernel, handle
    user = ctypes.windll.user32
    user.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user.FindWindowW.restype = wintypes.HWND
    user.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    user.SetForegroundWindow.argtypes = (wintypes.HWND,)
    # Первый экземпляр может ещё ждать сервер.
    for _ in range(160):
        hwnd = user.FindWindowW(None, TITLE)
        if hwnd:
            user.ShowWindow(hwnd, 9)  # SW_RESTORE
            user.SetForegroundWindow(hwnd)
            break
        time.sleep(.25)
    kernel.CloseHandle(handle)
    return kernel, None


def healthy(base):
    try:
        # Локальный запрос не должен проходить через системный прокси.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base + '/ping', timeout=.5) as response:
            return response.status == 200 and response.geturl() == base + '/ping'
    except Exception:
        return False


LOADING_HTML = """<!doctype html><meta charset="utf-8"><title>Linda Probe</title>
<body style="margin:0;background:#0d0f15;color:#e9ecf3;font:16px 'Segoe UI',sans-serif;display:grid;place-items:center;height:100vh">
<div style="text-align:center"><div style="font-size:34px;font-weight:300;letter-spacing:.02em">Linda Probe</div>
<div style="margin-top:14px;color:#8d96a8">Запускаю поиск. Первый запуск после установки может занять до минуты.</div>
<div style="margin:22px auto 0;width:220px;height:3px;background:#222838;overflow:hidden"><div style="width:40%;height:100%;background:#1ba1e2;animation:m 1.2s ease-in-out infinite"></div></div></div>
<style>@keyframes m{0%{margin-left:-40%}100%{margin-left:100%}}</style>"""
START_TIMEOUT = 300


def _splash_file():
    try:
        path = ROOT / 'assets' / 'splash.html'
    except Exception:
        return None
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def _splash_url(url):
    found = _splash_file()
    if found is None:
        return None
    try:
        return found.as_uri() + '?base=' + url.rsplit('/', 1)[0]
    except (OSError, ValueError):
        return None


def wait_healthy(base, proc, timeout=START_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError('Процесс сервера завершился до готовности.')
        if healthy(base):
            return
        time.sleep(.25)
    raise TimeoutError('Сервер не ответил за %d секунд.' % timeout)


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    except OSError:
        pass


def start_server(port):
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / 'app_server.log').open('ab', buffering=0) as log:
        return subprocess.Popen(
            [sys.executable, str(ROOT / 'localdocs_server.py'), str(port)],
            cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=log, stderr=log,
            env=dict(os.environ, PYTHONIOENCODING='utf-8'),
        )


def read_geometry():
    try:
        data = json.loads(STATE.read_text(encoding='utf-8'))
        result = {}
        for key, low, high in (('width', 760, 16384), ('height', 520, 16384),
                               ('x', -32768, 32767), ('y', -32768, 32767)):
            value = data.get(key)
            if type(value) is int and low <= value <= high:
                result[key] = value
        # Не восстанавливаем окно за пределами подключённых мониторов.
        user = ctypes.windll.user32
        left, top, width, height = (user.GetSystemMetrics(i) for i in (76, 77, 78, 79))
        if 'x' not in result or 'y' not in result or not (
            left <= result['x'] <= left + width - 100
            and top <= result['y'] <= top + height - 100
        ):
            result.pop('x', None)
            result.pop('y', None)
        return result
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def browser_path():
    for name, folder in (('msedge.exe', 'Microsoft/Edge/Application'),
                         ('chrome.exe', 'Google/Chrome/Application')):
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                try:
                    with winreg.OpenKey(hive, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion'
                                        '\\App Paths\\' + name, 0, winreg.KEY_READ | view) as key:
                        path = Path(os.path.expandvars(winreg.QueryValue(key, None).strip('"')))
                        if path.is_file():
                            return str(path)
                except OSError:
                    pass
        for root in (os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
                     os.environ.get('ProgramFiles', r'C:\Program Files'),
                     os.environ.get('LOCALAPPDATA', str(HOME.parent))):
            path = Path(root) / folder / name
            if path.is_file():
                return str(path)
        found = shutil.which(name)
        if found:
            return found
    return None


def browser_window(url):
    browser = browser_path()
    if browser:
        HOME.mkdir(parents=True, exist_ok=True)
        # Сначала сплэш: он сам дождётся готовности сервера и перейдёт к /app.
        target = _splash_url(url) or url
        subprocess.Popen([
            browser, '--app=' + target, '--window-size=1280,820',
            '--user-data-dir=' + str(HOME / 'webview-profile'),
        ]).wait()
    else:
        webbrowser.open(url)
        # У стандартного браузера нет доступного дескриптора окна.
        message('Поиск открыт в браузере. Нажмите ОК после закрытия вкладки поиска.')


def app_window(url, waiter=None):
    import webview

    geometry = {'width': 1280, 'height': 820, **read_geometry()}
    lock = threading.RLock()
    fullscreen = False

    picking = threading.Lock()

    class Keys:
        def toggle_fullscreen(self):
            nonlocal fullscreen
            with lock:
                window.toggle_fullscreen()
                fullscreen = not fullscreen

        def pick(self, kind='folder', multiple=False, images=False):
            # Native dialog owned by this window: instant, modal, never stacked by repeated clicks.
            if not picking.acquire(blocking=False):
                return {'busy': True}
            try:
                dialogs = getattr(webview, 'FileDialog', None)
                if kind == 'folder':
                    mode = dialogs.FOLDER if dialogs else webview.FOLDER_DIALOG
                    paths = window.create_file_dialog(mode)
                else:
                    mode = dialogs.OPEN if dialogs else webview.OPEN_DIALOG
                    types = (('Изображения (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif;*.tif;*.tiff)',)
                             if images else
                             ('PDF и изображения (*.pdf;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.webp)',
                              'Все файлы (*.*)'))
                    paths = window.create_file_dialog(mode, allow_multiple=bool(multiple), file_types=types)
                return {'paths': [str(p) for p in (paths or ())]}
            except Exception as exc:
                return {'error': str(exc)[:200]}
            finally:
                picking.release()

    if waiter:
        splash = _splash_url(url)
        initial = {'url': splash} if splash else {'html': LOADING_HTML}
    else:
        initial = {'url': url}
    window = webview.create_window(
        TITLE, **initial, **geometry, min_size=(760, 520),
        background_color='#0d0f15', text_select=True, js_api=Keys(),
    )

    def save():
        with lock:
            try:
                HOME.mkdir(parents=True, exist_ok=True)
                temporary = STATE.with_suffix('.tmp')
                temporary.write_text(json.dumps(geometry), encoding='utf-8')
                temporary.replace(STATE)
            except OSError:
                pass

    def resized(width, height):
        with lock:
            if not fullscreen and width >= 760 and height >= 520:
                geometry.update(width=int(width), height=int(height))
                save()

    def moved(x, y):
        with lock:
            if not fullscreen:
                geometry.update(x=int(x), y=int(y))
                save()

    def loaded():
        window.evaluate_js("""
            if (!window.__lindaProbeKeys) {
                window.__lindaProbeKeys = true;
                window.addEventListener('keydown', function (event) {
                    if (event.key === 'F11' || event.key === 'F12') {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        if (event.key === 'F11' && !event.repeat && window.pywebview)
                            window.pywebview.api.toggle_fullscreen();
                    }
                }, true);
            }
        """)

    window.events.resized += resized
    window.events.moved += moved
    window.events.closing += save
    window.events.loaded += loaded
    options = {'debug': False, 'gui': 'edgechromium'}
    icon = ROOT / 'assets/localdocs.ico'
    if icon.is_file():
        options['icon'] = str(icon)
    def background():
        try:
            waiter()
        except Exception as exc:
            message(f'{exc}\nПодробности: {LOGS / "app_server.log"}', 'Не удалось запустить поиск', error=True)
            window.destroy()
            return
        window.load_url(url)

    def start():
        extra = {'func': background} if waiter else {}
        try:
            webview.start(**options, **extra)
        except TypeError as exc:
            if 'icon' not in options or 'icon' not in str(exc):
                raise
            options.pop('icon')
            webview.start(**options, **extra)

    try:
        start()
    except Exception:
        # Если Edge недоступен, даём pywebview выбрать другой движок.
        options.pop('gui')
        start()


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument('--port', type=int, default=8798)
    parser.add_argument('--no-server', action='store_true')
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port: допустимы значения 1..65535')
    kernel, handle = acquire_window()
    if handle is None:
        return 0
    proc = None
    try:
        base = f'http://127.0.0.1:{args.port}'
        if not args.no_server and not healthy(base):
            try:
                proc = start_server(args.port)
            except Exception as exc:
                message(f'{exc}\nПодробности: {LOGS / "app_server.log"}',
                        'Не удалось запустить поиск', error=True)
                return 2
        url = base + '/app'
        def waiter():
            if not args.no_server:
                wait_healthy(base, proc)

        if args.browser:
            try:
                waiter()
            except Exception as exc:
                message(f'{exc}\nПодробности: {LOGS / "app_server.log"}', 'Не удалось запустить поиск', error=True)
                return 2
            browser_window(url)
        else:
            try:
                app_window(url, waiter)
            except Exception:
                try:
                    waiter()
                except Exception as exc:
                    message(f'{exc}\nПодробности: {LOGS / "app_server.log"}', 'Не удалось запустить поиск', error=True)
                    return 2
                browser_window(url)
        return 0
    finally:
        try:
            stop_server(proc)
        finally:
            kernel.CloseHandle(handle)


if __name__ == '__main__':
    raise SystemExit(main())
