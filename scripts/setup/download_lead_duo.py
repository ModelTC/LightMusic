"""Install immutable anvuew/Frazer inference assets and verify every file."""
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def matches(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == item['sha256']


def main():
    for item in json.loads((ROOT / 'configs/lead_duo.lock.json').read_text())['assets']:
        destination = ROOT / item['path']
        if matches(destination, item):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(item['url'], timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            if not matches(temporary, item):
                raise ValueError(f'Download checksum mismatch: {destination}')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
