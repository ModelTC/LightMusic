import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('release_checks', Path(__file__).resolve().parents[1] / 'scripts/check_release.py')
checks = importlib.util.module_from_spec(spec); spec.loader.exec_module(checks)


class ReleaseChecks(unittest.TestCase):
    def test_rejects_runtime_artifacts_and_local_paths(self):
        for name, payload in [('runs/a/state.json', b'{}'), ('data/dialogue_conversion/manifest.json', b'{}'),
                              ('models/adapter.pth', b'x'), ('configs/local/voice.yaml', b'x'),
                              ('configs/voice.yaml', b'reference: /mnt/private/project/ref.wav'),
                              ('configs/voice.yaml', b'reference: /data/user/LightMusic/data/ref.wav'),
                              ('configs/requirements.txt', b'package @ file:///home/user/wheel.whl'),
                              ('.env', b'SECRET=x')]:
            with self.subTest(name=name): self.assertTrue(checks.check_blob(name, payload))
        self.assertFalse(checks.check_blob('configs/examples/voice.yaml', b'reference: data/raw/ref.wav'))
        self.assertFalse(checks.check_blob('data/demo_sources.json', b'{}'))
        self.assertFalse(checks.check_blob('configs/seed_svc.local.yml', b'name: models/BigVGAN'))
        self.assertTrue(checks.check_blob('configs/seed_svc.local.yml', b'name: /mnt/private/BigVGAN'))

    def test_checks_index_bytes_not_unstaged_worktree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args): return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
            git('init', '-q')
            file = root / 'configs/task.yaml'; file.parent.mkdir()
            file.write_text('reference: /mnt/private/ref.wav\n'); git('add', 'configs/task.yaml')
            file.write_text('reference: data/raw/ref.wav\n')
            self.assertEqual(checks.check(root)['status'], 'failed')
            git('add', 'configs/task.yaml')
            self.assertEqual(checks.check(root)['status'], 'passed')

    def test_preserves_but_reports_historical_environment_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(['git', '-C', str(root), 'init', '-q'], check=True)
            file = root / 'configs/base.installed.txt'; file.parent.mkdir()
            file.write_text('flash_attn @ file:///mnt/local/wheel.whl\n')
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            report = checks.check(root)
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(len(report['warnings']), 1)


if __name__ == '__main__': unittest.main()
