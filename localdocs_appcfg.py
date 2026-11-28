from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = 'LocalDocs'


def _package_version() -> str:
    # The search package carries the single release number; a stale literal here made the updater report 1.0.0.
    try:
        from project_search import __version__
        return str(__version__)
    except Exception:
        return '0.0.0'


APP_VERSION = os.environ.get('LOCALDOCS_APP_VERSION') or _package_version()
BUNDLED_MODEL_VERSION = APP_VERSION
APP_ID = '{9A446F89-A4BA-4CA0-8980-55C184AAE862}'
HF_REPO = 'Lindarixon/LocalDocs'
GITHUB_REPO = 'Lendarixon/Linda-Probe'
SITE_URL = 'https://lendarixon.github.io/Linda-Probe/'
RELEASES_URL = f'https://github.com/{GITHUB_REPO}/releases'
BASE_URL = os.environ.get('LOCALDOCS_BASE_URL', f'https://raw.githubusercontent.com/{GITHUB_REPO}/main/update').rstrip('/')
MANIFEST_NAME = 'latest.json'
SIGNATURE_NAME = 'latest.json.sig'
SIGNING_PUBKEY_B64 = os.environ.get('LOCALDOCS_PUBKEY_B64', 'b+1Vn6JHbtQHF6XTjoLn7qqKagE5cCsQ5MY87erF3Dw=')
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = ('char_ngrams_model.json',)


def data_root() -> Path:
    """Return the current application data root."""
    override = os.environ.get('LOCALDOCS_DATA_ROOT')
    if override:
        return Path(override)
    if getattr(sys, 'frozen', False):
        return Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'LocalDocs'
    return BASE_DIR


def models_root() -> Path:
    """Return the Trace update storage directory."""
    return data_root() / 'localdocs_update'
