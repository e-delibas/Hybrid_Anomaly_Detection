"""Read-only checks of the archived files used by the release candidate."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
from release_runtime import locate_project, settings


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_check(project, item):
    path = project / item['path']
    result = dict(path=item['path'], expected_sha256=item['sha256'], stages=item.get('stages', []))
    if not path.resolve().is_relative_to(project.resolve()):
        return dict(result, status='OUTSIDE_PROJECT')
    if not path.is_file():
        return dict(result, status='MISSING')
    try:
        actual = digest(path)
        return dict(result, actual_sha256=actual, size_bytes=path.stat().st_size,
                    status='MATCH' if actual == item['sha256'] else 'HASH_MISMATCH')
    except OSError as exc:
        return dict(result, status='READ_ERROR', detail=str(exc))


def anchor_check(project, anchor):
    paths = sorted(project.glob(anchor['pattern']))
    found = []
    for p in paths:
        if not p.resolve().is_relative_to(project.resolve()) or not p.is_file():
            continue
        try:
            actual = canonical(json.loads(p.read_text())) if anchor['hash_kind'] == 'canonical' else digest(p)
            if actual == anchor['sha256']:
                found.append(str(p.relative_to(project)))
        except (OSError, ValueError):
            continue
    return dict(anchor, status='MATCH' if found else 'NO_MATCHING_SUMMARY',
                matches=found, candidates=len(paths))


def python_candidates(project):
    cfg = settings()
    candidates = [sys.executable]
    for kind in ['classical', 'torch']:
        value = os.environ.get('CMAPSS_' + kind.upper() + '_PYTHON') or cfg.get(kind + '_python')
        if value:
            candidates.append(value)
    bases = {Path(sys.prefix), Path(sys.base_prefix), Path('/opt/anaconda3'), Path('/opt/miniconda3'),
             Path.home()/'anaconda3', Path.home()/'miniconda3', Path.home()/'.conda'}
    for base in sorted(bases):
        candidates.extend(str(p) for p in sorted((base/'envs').glob('*/bin/python')))
    registry = Path.home()/'.conda/environments.txt'
    if registry.is_file():
        candidates.extend(str(Path(line.strip())/'bin/python') for line in registry.read_text().splitlines() if line.strip())
    for parent in [project, project.parent, Path.home()/'.venvs']:
        candidates.extend(str(p) for p in sorted(parent.glob('*/bin/python')))
        candidates.extend(str(parent/name/'bin/python') for name in ['.venv', 'venv'])
    unique = []
    for value in candidates:
        path = str(Path(value).expanduser().absolute())
        if Path(path).is_file() and path not in unique:
            unique.append(path)
    return unique


def inspect_environments(project, expected):
    packages = sorted(set().union(*(v.keys() for v in expected.values())) - {'python'})
    code = ('import importlib.metadata as m,json,platform\n'
            "v={'python':platform.python_version()}\n"
            'for p in ' + repr(packages) + ':\n'
            ' try:v[p]=m.version(p)\n'
            ' except m.PackageNotFoundError:v[p]=None\n'
            'print(json.dumps(v))\n')
    reports = []
    for executable in python_candidates(project):
        try:
            proc = subprocess.run([executable, '-c', code], capture_output=True, text=True, timeout=20)
            versions = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
            matches = [kind for kind, wanted in expected.items()
                       if all(versions.get(k) == value for k, value in wanted.items())]
            reports.append(dict(executable=executable, versions=versions,
                                metadata_returncode=proc.returncode, matching_environments=matches))
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
            reports.append(dict(executable=executable, status='METADATA_ERROR', detail=str(exc)))
    return reports


def run_preflight(project=None):
    project = locate_project(project)
    manifest = json.loads((RELEASE_ROOT/'assets/replay_dependencies.json').read_text())
    run = project/'revision_outputs/16_code_release_preflight'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    print('Project:', project, flush=True)
    print('Checking', len(manifest['files']), 'recorded input files. No training will run.', flush=True)
    report = dict(stage='16_code_release_preflight', status='CHECKING', project=str(project),
                  scope='Listed recorded dependencies and summary anchors; not a fresh-training reproduction.',
                  model_training_performed=False, existing_experiment_files_modified=False,
                  tensorflow_or_torch_imported=False, dependency_manifest_sha256=digest(RELEASE_ROOT/'assets/replay_dependencies.json'))
    results = []
    try:
        for index, item in enumerate(manifest['files'], 1):
            results.append(file_check(project, item))
            if index % 100 == 0 or index == len(manifest['files']):
                print('Checked', index, '/', len(manifest['files']), flush=True)
        anchors = [anchor_check(project, a) for a in manifest['summary_anchors']]
        environments = inspect_environments(project, manifest['expected_environments'])
        counts = dict(Counter(r['status'] for r in results))
        environment_matches = {k: [e['executable'] for e in environments if k in e.get('matching_environments', [])]
                               for k in manifest['expected_environments']}
        package_manifest = RELEASE_ROOT/'package_manifest.json'
        package_checks = []
        if package_manifest.is_file():
            for item in json.loads(package_manifest.read_text())['files']:
                p=RELEASE_ROOT/item['path']
                package_checks.append(dict(path=item['path'], status='MATCH' if p.is_file() and digest(p)==item['sha256'] else 'MISSING_OR_CHANGED'))
        sources_ok = all(r['status']=='MATCH' for r in results) and all(a['status']=='MATCH' for a in anchors)
        package_ok = bool(package_checks) and all(r['status']=='MATCH' for r in package_checks)
        environments_ok = all(environment_matches.values())
        report.update(status='RECORDED_DEPENDENCIES_VERIFIED' if sources_ok and package_ok and environments_ok else 'REVIEW_REQUIRED',
                      input_counts=counts, input_issues=[r for r in results if r['status']!='MATCH'],
                      summary_anchors=anchors, environments=environments, environment_matches=environment_matches,
                      package_files_checked=len(package_checks), package_issues=[r for r in package_checks if r['status']!='MATCH'],
                      validated_file_count=len(results),
                      limitations=['Metadata compatibility is not an import/runtime test.',
                                   'Only dependencies listed in replay_dependencies.json were checked.',
                                   'Fresh training and model-score reproduction were not run.'])
        (run/'checked_dependencies.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    except Exception as exc:
        report.update(status='PREFLIGHT_ERROR', error=type(exc).__name__+': '+str(exc))
    path=run/'code_release_preflight_summary.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('STATUS:', report['status'], flush=True)
    print('SEND THIS FILE:', path, flush=True)
    return report, path


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path)
    run_preflight(parser.parse_args().project)
