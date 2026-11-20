"""Profile, routing and export contracts; no training or optimizer steps."""
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
from music_cover.lora import check_reference, pcm_identity, resolve_lora
from music_cover.voice import resolve_voice, merge_voice_config
from music_cover.modules import voice_conversion as node
from music_cover import lora_training as training


class LoRAIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'base.pth'; self.base.write_bytes(b'base')
        self.folder = self.root / 'voice'; self.folder.mkdir()
        sf.write(self.folder / 'reference.wav', np.sin(np.arange(44100) * .1).astype('float32') * .1, 44100, subtype='FLOAT')
        (self.folder / 'adapter.pth').write_bytes(b'adapter')
        (self.folder / 'config.yml').write_text(yaml.safe_dump({'preprocess_params': {'sr': 44100}, 'model_params': {'DiT': {'f0_condition': True}}}))
        for name in ('validation.json', 'training.json'): write_json(self.folder / name, {})
        def rec(name): return {'path': name, 'sha256': sha256(self.folder / name)}
        self.profile = {'schema_version': 2, 'backend': 'seed_vc', 'adaptation_type': 'lora', 'adaptation_scope': 'speaker',
            'voice_id': 'test', 'version': 'v1', 'dataset_identity': 'dataset', 'training_steps': 1000,
            'adapter': rec('adapter.pth'), 'config': rec('config.yml'), 'reference': rec('reference.wav'),
            'validation': rec('validation.json'), 'training': rec('training.json'),
            'base_checkpoint': {'path': 'base.pth', 'sha256': sha256(self.base)},
            'adapter_metadata': {'base_sha256': sha256(self.base)}, 'reference_pcm': pcm_identity(self.folder / 'reference.wav'),
            'validation_status': 'candidate_requires_listening'}
        self.file = self.folder / 'voice.json'; self.save()

    def save(self): write_json(self.file, self.profile)

    def test_lora_profile_and_full_profile_are_distinct(self):
        result = resolve_voice('lora', self.file, self.root)
        self.assertEqual(result['checkpoint'], self.base)
        self.assertEqual(result['adapter'], self.folder / 'adapter.pth')
        with self.assertRaises(ValueError): resolve_voice('finetuned', self.file, self.root)

    def test_wrong_base_or_adapter_fails(self):
        self.base.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'base checkpoint'): resolve_lora(self.file, self.root)
        self.base.write_bytes(b'base'); (self.folder / 'adapter.pth').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'checksum'): resolve_lora(self.file, self.root)

    def test_profile_cannot_redirect_adapter_outside_package(self):
        self.profile['adapter'] = {'path': '../base.pth', 'sha256': sha256(self.base)}; self.save()
        with self.assertRaises(ValueError): resolve_lora(self.file, self.root)

    def test_reference_compares_pcm_and_rejects_other_audio(self):
        data, rate = sf.read(self.folder / 'reference.wav', dtype='float32')
        other = self.root / 'other.wav'; sf.write(other, data, rate, subtype='FLOAT')
        check_reference(other, self.folder / 'reference.wav')
        sf.write(other, data * .5, rate, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'reference differs'): check_reference(other, self.folder / 'reference.wav')

    def test_node_mode_and_explicit_reference_policy(self):
        request = {'seed': 42, 'steps': 30}
        good = {'backend': 'seed_vc', 'mode': 'lora', 'voice_profile': str(self.file), 'reference_policy': 'profile', 'candidate_count': 1}
        self.assertEqual(node.params(request, good)['candidate_seeds'], [42])
        for changes in ({'backend': 'rvc'}, {'mode': 'zero_shot'}, {'voice_profile': None}):
            with self.assertRaises(ValueError): node.params(request, {**good, **changes})

    def test_adapter_is_a_fingerprinted_input_and_reference_override_is_explicit(self):
        task_ref = self.root / 'task.wav'; sf.write(task_ref, np.zeros(44100), 44100)
        ctx = SimpleNamespace(root=self.root, config={node.ID: {'mode': 'lora', 'voice_profile': str(self.file), 'reference_policy': 'profile'}},
            output=lambda stage, role: task_ref if role == 'reference' else self.root / 'vocals.wav')
        values = node.inputs(ctx)
        self.assertEqual(values['adapter'], self.folder / 'adapter.pth')
        self.assertEqual(values['reference'], self.folder / 'reference.wav')
        self.assertEqual(values['task_reference'], task_ref)
        ctx.config[node.ID]['reference_policy'] = 'task'
        with self.assertRaises(ValueError): node.inputs(ctx)

    def test_switching_mode_clears_profile_reference_policy(self):
        value = merge_voice_config({'mode': 'lora', 'reference_policy': 'profile'}, {'mode': 'zero_shot'})
        self.assertEqual(value['reference_policy'], 'task')

    def test_import_routes_without_training(self):
        with patch.object(training, 'import_existing') as imported:
            training.train({}, self.root, self.folder, SimpleNamespace(import_run='old', resume=False))
            imported.assert_called_once_with({}, self.root, self.folder, 'old')
        with self.assertRaises(ValueError):
            training.train({}, self.root, self.folder, SimpleNamespace(import_run='old', resume=True))

    def test_resume_rejects_changed_identity_checkpoint_and_exhausted_budget(self):
        latest = self.folder / 'lora_latest.pth'; latest.write_bytes(b'checkpoint')
        ledger = {'training_identity': 'identity', 'files': {latest.name: sha256(latest)}}
        write_json(self.folder / 'identity.json', ledger)
        with patch.object(training, 'checked_adapter', return_value={'training': {'iters': 250}}) as read:
            training.resume_checkpoint(self.folder, 'identity', 'base', {}, 1000)
            self.assertEqual(read.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'exceed'): training.resume_checkpoint(self.folder, 'identity', 'base', {}, 250)
            with self.assertRaisesRegex(ValueError, 'mismatch'): training.resume_checkpoint(self.folder, 'other_dataset', 'base', {}, 1000)
            latest.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'mismatch'): training.resume_checkpoint(self.folder, 'identity', 'base', {}, 1000)

    def test_resume_requires_optimizer_state(self):
        latest = self.folder / 'lora_latest.pth'; latest.write_bytes(b'adapter_only')
        write_json(self.folder / 'identity.json', {'training_identity': 'identity', 'files': {latest.name: sha256(latest)}})
        with patch.object(training, 'checked_adapter', return_value={}):
            with self.assertRaisesRegex(ValueError, 'optimizer'): training.resume_checkpoint(self.folder, 'identity', 'base', {}, 1000)

    def test_candidate_controller_preserves_lora_mode(self):
        from music_cover.backends import voice_candidates
        request = {'mode': 'lora', 'voice_profile': str(self.file), 'source': str(self.folder / 'reference.wav'),
            'reference': str(self.folder / 'reference.wav'), 'seed': 42, 'steps': 30, 'candidate_count': 1}
        out = self.root / 'candidates'; out.mkdir()
        def fail(command, **kwargs):
            self.assertEqual(command[command.index('--voice-mode') + 1], 'lora')
            raise OSError('Synthetic inference failure')
        with patch.object(voice_candidates, 'resolve_voice', return_value=resolve_lora(self.file, self.root)), \
                patch.object(voice_candidates.subprocess, 'run', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'No readable'): voice_candidates.generate(request, out)
        report = json.loads((out / 'candidates.json').read_text())
        self.assertEqual(report['mode'], 'lora')
        self.assertFalse(report['training_performed'])


if __name__ == '__main__': unittest.main()
