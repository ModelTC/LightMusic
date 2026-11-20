import subprocess
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from music_cover.modules import style_generation
from music_cover.engine import Pipeline
from music_cover.audio import sha256


class H100RuntimeTests(unittest.TestCase):
    def test_review_verifier_resolves_run_relative_checksums(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/verify_review.py'
        spec = importlib.util.spec_from_file_location('h100_verify_review', script)
        verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            attempt = run / 'stages/00_prepare/attempt_001'; attempt.mkdir(parents=True)
            payload = attempt / 'audio.wav'; payload.write_bytes(b'checked payload')
            marker = attempt / 'outputs.json'; marker.write_text('{}')
            state = {'schema_version': 2, 'path_base': 'run', 'status': 'completed',
                     'request': {'review': False}, 'active_stages': ['00_prepare'],
                     'stages': {'00_prepare': {'status': 'succeeded',
                         'attempt': str(attempt.relative_to(run)), 'outputs_manifest_sha256': sha256(marker),
                         'outputs': {'audio': {'path': str(payload.relative_to(run)), 'sha256': sha256(payload)}}}}}
            (run / 'manifest.json').write_text(json.dumps(state))
            with patch.object(verifier, 'verify_delivery', return_value={}):
                self.assertEqual(verifier.verify(run)['checked_stages'], 1)
                payload.write_bytes(b'changed payload')
                with self.assertRaisesRegex(ValueError, 'Artifact checksum mismatch'):
                    verifier.verify(run)

    def test_delivery_copy_or_publication_error_keeps_previous_delivery(self):
        for failure in ['copy', 'replace']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                old = run / 'deliverables'; old.mkdir()
                (old / 'final.wav').write_bytes(b'old validated audio')
                attempt = run / 'attempt'; (attempt / 'artifacts').mkdir(parents=True)
                (attempt / 'artifacts/final.wav').write_bytes(b'new audio')
                pipeline = object.__new__(Pipeline)
                pipeline.run = run
                pipeline.state = {'stages': {'11_delivery': {'attempt': str(attempt)}}}
                target = 'music_cover.engine.shutil.copytree' if failure == 'copy' else 'pathlib.Path.replace'
                with patch(target, side_effect=OSError('simulated storage error')):
                    with self.assertRaises(OSError):
                        pipeline.publish()
                self.assertFalse(old.is_symlink())
                self.assertEqual((old / 'final.wav').read_bytes(), b'old validated audio')
                self.assertFalse(list(run.glob('.delivery_*')))

    def test_all_launchers_initialize_the_project_environment(self):
        root = Path(__file__).resolve().parents[1]
        for script in (root / 'scripts').rglob('*.sh'):
            content = script.read_text()
            if 'PROJECT_ROOT=' in content and script.name != 'environment.sh':
                self.assertIn('source "${PROJECT_ROOT}/scripts/environment.sh"', content, str(script))
                self.assertNotIn('/opt/conda', content, str(script))

    def test_environment_can_be_loaded_outside_the_project(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(['bash', '-c', 'source "$1"; printf "%s\\n" "$MUSIC_COVER_BASE_PREFIX" "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR"',
                                 'bash', str(root / 'scripts/environment.sh')],
                                cwd='/', capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), [str(root / p) for p in
            ['envs/base', 'cache/tmp', 'cache/huggingface', 'cache/triton']])

    def test_memory_budget_is_audited_and_forwarded_without_trimming_score(self):
        request = {'style': 'pop rock', 'seed': 42}
        low = style_generation.params(request, {'memory_budget_gib': 64})
        high = style_generation.params(request, {'memory_budget_gib': 72})
        self.assertNotEqual(low, high)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            lyrics = out / 'lyrics.txt'; lyrics.write_text('test lyrics')
            score = out / 'score.abc'; score.write_text('complete score')
            ctx = Mock()
            style_generation.run(ctx, out, {'lyrics': lyrics, 'score': score}, low)
            self.assertEqual(ctx.execute.call_args.args[-2:], ('--memory-budget-gib', 64.0))
            self.assertEqual(score.read_text(), 'complete score')

    def test_invalid_memory_budget_fails_before_generation(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                style_generation.params({'seed': 42}, {'memory_budget_gib': value})
