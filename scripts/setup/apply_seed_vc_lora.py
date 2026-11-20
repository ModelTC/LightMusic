#!/usr/bin/env python3
"""Apply the versioned LoRA patch to the pinned submodule; reject unrelated edits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='Verify installation without modifying files')
    a = p.parse_args()
    folder = ROOT / 'patches/seed-vc-lora-v1'
    manifest = json.loads((folder / 'manifest.json').read_text())
    patch = folder / 'implementation.patch'
    if digest(patch) != manifest['patch_sha256']: raise ValueError('LoRA patch checksum mismatch')
    repo = ROOT / 'third_party/seed-vc'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != manifest['base_commit']: raise ValueError('Seed-VC submodule commit differs from the supported base')
    current = {name: digest(repo / name) for name in manifest['files']}
    if all(current[name] == row['after'] for name, row in manifest['files'].items()):
        print('Seed-VC LoRA patch verified'); return
    if a.check: raise ValueError('Seed-VC LoRA patch is not installed or has changed')
    if any(current[name] != row['before'] for name, row in manifest['files'].items()):
        raise ValueError('Seed-VC files contain unexpected/partial changes; inspect before applying')
    subprocess.run(['git', '-C', str(repo), 'apply', '--check', str(patch)], check=True)
    subprocess.run(['git', '-C', str(repo), 'apply', str(patch)], check=True)
    if not all(digest(repo / name) == row['after'] for name, row in manifest['files'].items()):
        raise ValueError('Applied LoRA files failed checksum verification')
    print('Seed-VC LoRA patch applied and verified')


if __name__ == '__main__': main()
