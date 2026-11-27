#!/usr/bin/env python3
"""Read-only Git publication checks: changed index entries (--staged) or all index entries (--tracked)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRS = {'envs', 'models', 'cache', 'runs', 'checkpoints'}
MEDIA = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx', '.wav', '.mp3', '.flac', '.mp4', '.m4a', '.ogg', '.webm'}
MACHINE_PATH = re.compile(r'/(?:mnt|Users|home)/[^\s\"\'`<>]+|/data/[^\s\"\'`<>]+/(?:LightMusic|music-cover-baseline)')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def check_blob(name, data):
    path = PurePosixPath(name)
    issues = []
    if path.parts[0] in LOCAL_DIRS or (path.parts[0] == 'data' and name != 'data/demo_sources.json'):
        issues.append('runtime data or generated artifact')
    # This existing generated Seed-VC config uses project-relative model paths
    # and participates in runtime fingerprints. Other local configs stay private.
    if name.startswith('configs/local/') or ('.local.' in path.name and name != 'configs/seed_svc.local.yml'):
        issues.append('machine-local configuration')
    if path.suffix.lower() in MEDIA or len(data) > 10 * 1024 * 1024:
        issues.append('model/media file or file larger than 10 MiB')
    if path.name.startswith('.env') and path.name != '.env.example' or path.suffix in {'.key', '.pem'}:
        issues.append('credential/configuration file')
    if path.name.startswith('._') or path.name == '.DS_Store' or '__pycache__' in path.parts:
        issues.append('OS or Python cache metadata')
    if path.parts[0] in {'src', 'scripts', 'configs'} and path.suffix in {'.py', '.sh', '.json', '.yaml', '.yml', '.txt'}:
        text = data.decode('utf-8', errors='replace')
        snapshot = name.startswith('configs/') and path.name.endswith('.installed.txt')
        if name != 'scripts/check_release.py' and not snapshot and MACHINE_PATH.search(text):
            issues.append('machine-specific absolute path; use project-relative configuration')
        if path.suffix == '.txt' and not snapshot and re.search(r'@ file:|^-e /', text, re.MULTILINE):
            issues.append('environment snapshot includes a local installation path')
    return [f'{name}: {reason}' for reason in issues]


def check(root=ROOT, staged=True):
    root = Path(root)
    mode = 'staged' if staged else 'tracked'
    if staged:
        names = git(root, 'diff', '--cached', '--name-only', '--diff-filter=ACMRU', '-z').decode().split('\0')
    else:
        names = git(root, 'ls-files', '-z').decode().split('\0')
    index = {}
    for row in git(root, 'ls-files', '--stage', '-z').decode().split('\0'):
        if row:
            info, name = row.split('\t', 1); index[name] = info.split()
    errors, checked, submodules, warnings = [], [], [], []
    for name in sorted(set(filter(None, names))):
        entry = index.get(name)
        if not entry: continue
        permissions, oid, stage = entry
        if stage != '0':
            errors.append(f'{name}: unresolved index conflict'); continue
        if permissions == '160000':
            submodules.append({'path': name, 'commit': oid}); continue
        if permissions == '120000':
            errors.append(f'{name}: symlink in source publication; keep ordinary source files'); continue
        data = git(root, 'cat-file', 'blob', oid)
        errors += check_blob(name, data)
        if name.startswith('configs/') and name.endswith('.installed.txt') and re.search(rb'@ file:|^-e /', data, re.MULTILINE):
            warnings.append(f'{name}: historical environment fingerprint contains local wheel provenance; preserved for run identity, not an install requirements file. Review before publishing.')
        checked.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    return {'status': 'failed' if errors else 'passed', 'mode': mode, 'files': checked,
            'submodules': submodules, 'errors': errors, 'warnings': warnings,
            'note': 'Index bytes are checked, not unstaged edits. This is a publication hygiene check, not a complete secret scanner or GPU acceptance test.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--staged', action='store_true')
    modes.add_argument('--tracked', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = check(staged=args.staged)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2))
    print(f'Checked {len(result["files"])} files from the Git index.')
    raise SystemExit(bool(result['errors']))


if __name__ == '__main__': main()
