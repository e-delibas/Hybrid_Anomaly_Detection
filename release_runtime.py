"""Paths and recorded-run environments shared by the release notebooks."""
from pathlib import Path
import json
import os
import sys

RELEASE_ROOT = Path(__file__).resolve().parent


def load_json(relative_path):
    return json.loads((RELEASE_ROOT / relative_path).read_text(encoding='utf-8'))


def load_text(relative_path):
    return (RELEASE_ROOT / relative_path).read_text(encoding='utf-8')


def settings():
    path = RELEASE_ROOT / 'settings.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def locate_project(explicit=None):
    configured = explicit or os.environ.get('CMAPSS_PROJECT_DIR') or settings().get('project_dir')
    if configured:
        candidate = Path(configured).expanduser().absolute()
        if not (candidate / 'data').is_dir() or not (candidate / 'revision_outputs').is_dir():
            raise FileNotFoundError('The project must contain data/ and revision_outputs/: ' + str(candidate))
        return candidate
    candidates = [Path.cwd(), *Path.cwd().parents, RELEASE_ROOT, *RELEASE_ROOT.parents]
    for candidate in candidates:
        if (candidate / 'data').is_dir() and (candidate / 'revision_outputs').is_dir():
            return candidate
    raise FileNotFoundError('Set CMAPSS_PROJECT_DIR or project_dir in settings.json to the recorded experiment project.')


def configured_python(kind):
    env_key = 'CMAPSS_' + kind.upper() + '_PYTHON'
    value = os.environ.get(env_key) or settings().get(kind + '_python') or sys.executable
    # Preserve the executable path: resolving a venv symlink can change its environment.
    return Path(value).expanduser().absolute()


def activate_project():
    project = locate_project()
    classical = configured_python('classical')
    torch = configured_python('torch')
    os.chdir(project)
    return project, classical, torch
