"""LoRA workflow regressions: bad input must not corrupt completed training state."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml

from music_cover.audio import sha256
from music_cover.common.io import write_json
from music_cover import lora_training as lora
from music_cover import lora_training_config as config_tool
from music_cover import voice_training as cli


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'runs/voice_training/test'; self.run.mkdir(parents=True)
        self.adapter = self.run / 'checkpoints/adapter.pth'; self.adapter.parent.mkdir()
        self.adapter.write_bytes(b'final adapter')
        self.trained = {'adapter': str(self.adapter), 'adapter_sha256': sha256(self.adapter), 'training_identity': 'training-A'}
        train_out = self.run / 'stages/01_train/attempt_001/artifacts'; train_out.mkdir(parents=True)
        write_json(train_out / 'training.json', self.trained)
        self.state = {'schema_version': 1, 'stages': {'01_train': {'status': 'succeeded', 'output': str(train_out.relative_to(self.run))}}}
        write_json(self.run / 'state.json', self.state)
        self.config = self.root / 'config.yaml'; self.config.write_text('adaptation_type: lora\n')
        for module in (lora, cli):
            p = patch.object(module, 'ROOT', self.root); p.start(); self.addCleanup(p.stop)

    def args(self, split='test', checkpoint=None):
        return SimpleNamespace(evaluation_split=split, checkpoint=checkpoint)

    def report(self, number=1, selected=None, identity='training-A', sealed=True, failed=False):
        adapter = selected or self.adapter
        attempt = self.run / f'stages/02_validate/attempt_{number:03d}'
        file = attempt / 'artifacts/validation.json'
        write_json(file, {'evaluation_split': 'validation', 'training_identity': identity,
            'selected_adapter': str(adapter), 'selected_adapter_sha256': sha256(adapter)})
        if sealed:
            write_json(attempt / 'outputs.json', {'status': 'succeeded', 'artifacts': {
                'validation.json': {'sha256': sha256(file), 'bytes': file.stat().st_size}}})
        if failed: write_json(attempt / 'failure.json', {'error': 'failure after report generation'})
        return file

    def test_default_validation_uses_final_adapter(self):
        self.assertEqual(lora.validation_candidates(self.run, self.args('validation')), [self.adapter])

    def test_latest_validation_is_ordered_numerically(self):
        earlier = self.run / 'checkpoints/earlier.pth'; earlier.write_bytes(b'earlier')
        self.report(number=999, selected=earlier)
        self.report(number=1000, selected=self.adapter)
        self.assertEqual(lora.validation_candidates(self.run, self.args()), [self.adapter])

    def test_import_and_resume_are_rejected_before_loading_config(self):
        with patch.object(cli, 'load_config') as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main(['--stage', '01_train', '--config', str(self.config), '--run', str(self.run),
                          '--resume', '--import-run', 'runs/old'])
            self.assertEqual(raised.exception.code, 2)
            load.assert_not_called()
        self.assertEqual(json.loads((self.run / 'state.json').read_text()), self.state)

    def test_test_requires_successful_validation_without_changing_state(self):
        before = (self.run / 'state.json').read_bytes()
        argv = ['--stage', '02_validate', '--config', str(self.config), '--run', str(self.run), '--evaluation-split', 'test']
        with patch.object(cli, 'load_config', return_value={'adaptation_type': 'lora'}), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised: cli.main(argv)
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual((self.run / 'state.json').read_bytes(), before)
        self.assertFalse((self.run / 'stages/02_validate').exists())

    def test_empty_variables_are_rejected_before_loading_config(self):
        for flag in ['--config', '--run', '--checkpoint']:
            argv = ['--stage', '02_validate', '--config', str(self.config), '--run', str(self.run)]
            if flag == '--checkpoint': argv += [flag, '']
            else: argv[argv.index(flag) + 1] = '   '
            with self.subTest(flag=flag), patch.object(cli, 'load_config') as load, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised: cli.main(argv)
                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()
        self.assertEqual(json.loads((self.run / 'state.json').read_text()), self.state)

    def test_missing_checkpoint_does_not_create_attempt(self):
        with patch.object(cli, 'load_config') as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): cli.main(['--stage', '02_validate', '--config', str(self.config), '--run', str(self.run), '--checkpoint', str(self.root / 'missing.pth')])
            load.assert_not_called()
        self.assertFalse((self.run / 'stages/02_validate').exists())

    def test_default_test_uses_latest_selection_not_unvalidated_final_adapter(self):
        first = self.run / 'checkpoints/step_250.pth'; first.write_bytes(b'250')
        second = self.run / 'checkpoints/step_500.pth'; second.write_bytes(b'500')
        self.report(1, first); self.report(2, second)
        self.assertEqual(lora.validation_candidates(self.run, self.args()), [second])
        self.assertEqual(lora.validation_candidates(self.run, self.args(checkpoint=[str(first)])), [first])
        with self.assertRaisesRegex(ValueError, 'differs'):
            lora.validation_candidates(self.run, self.args(checkpoint=[str(self.adapter)]))

    def test_failed_incomplete_or_other_training_reports_do_not_authorize_test(self):
        self.report(1, sealed=False); self.report(2, failed=True); self.report(3, identity='training-B')
        with self.assertRaisesRegex(ValueError, 'No successful validation'):
            lora.validation_candidates(self.run, self.args())

    def test_tampered_report_or_selected_weight_is_rejected(self):
        report = self.report()
        original = report.read_bytes()
        report.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            lora.validation_candidates(self.run, self.args())
        report.write_bytes(original)
        self.adapter.write_bytes(b'changed after validation')
        with self.assertRaisesRegex(ValueError, 'differs'):
            lora.validation_candidates(self.run, self.args())

    def test_explicit_empty_or_multiple_test_checkpoints_are_rejected(self):
        self.report()
        with self.assertRaisesRegex(ValueError, 'empty'):
            lora.validation_candidates(self.run, self.args(checkpoint=['']))
        other = self.run / 'other.pth'; other.write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            lora.validation_candidates(self.run, self.args(checkpoint=[str(self.adapter), str(other)]))

    def test_source_hash_and_time_windows_are_checked(self):
        source = self.root / 'source.wav'
        sf.write(source, np.sin(np.arange(44100 * 2) * .03) * .1, 44100)
        config = {'validation': {'source_audio': str(source), 'source_sha256': sha256(source)}}
        self.assertEqual(lora.validation_source(config), source)
        good = {'windows': [{'start': 1, 'duration': 1}], 'steps': 30, 'seed': 42}
        self.assertEqual(lora.validate_windows(good, 2), good['windows'])
        bad = [{'windows': []}, {'windows': [{'start': 1, 'duration': 2}]},
               {'windows': [{'start': float('nan'), 'duration': 1}]}, {'windows': [{'start': 0, 'duration': 0}]},
               {**good, 'steps': 0}, {**good, 'seed': True}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError): lora.validate_windows(value, 2)
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): lora.validation_source(config)

    def test_plan_does_not_create_config_or_directories(self):
        output = self.root / 'fresh/config.yaml'
        argv = ['--voice-id', 'voice', '--version', 'r16_v001', '--reference', 'ref.wav',
                '--reference-group', 'voice:prompt', '--dataset-dir', 'dataset', '--source-run', 'song',
                '--output', str(output), '--plan']
        with patch.object(config_tool, 'build_config', return_value=({'adaptation_type': 'lora'}, output)), contextlib.redirect_stdout(io.StringIO()):
            config_tool.main(argv)
        self.assertFalse(output.parent.exists())

    def test_config_rejects_bad_window_notation_with_an_actionable_error(self):
        data = self.root / 'data.yaml'; data.write_text('dataset_dir: data/voice_training/voice/new/dataset\nsources: []\n')
        args = config_tool.parser().parse_args(['--voice-id', 'voice', '--version', 'new', '--reference', 'ref.wav',
            '--reference-group', 'voice:prompt', '--data-config', str(data), '--source-run', 'song',
            '--output', str(self.root / 'new.yaml'), '--validation-windows', '36-12'])
        with self.assertRaisesRegex(ValueError, 'start:duration'):
            config_tool.build_config(args)
        self.assertFalse((self.root / 'new.yaml').exists())


if __name__ == '__main__': unittest.main()
