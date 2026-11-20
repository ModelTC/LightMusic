"""CPU-only contract tests. No datasets, downloads, backward calls or optimizer steps."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sys
ROOT = Path(__file__).resolve().parents[1] / 'third_party/seed-vc'
sys.path.insert(0, str(ROOT))
import torch
import yaml

from modules.commons import build_model, recursive_munch
from modules.lora import (LoRALinear, adapter_parameters, architecture_signature,
                          attach_adapter, inject_lora, load_adapter_weights,
                          load_base_weights, make_metadata, read_adapter,
                          save_adapter, validate_config)
from optimizers import build_optimizer


ROOT = Path(__file__).resolve().parents[1] / 'third_party/seed-vc'


def config(small=True):
    cfg = yaml.safe_load((ROOT / 'configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml').read_text())
    if small:
        cfg['model_params']['DiT'].update(hidden_dim=32, num_heads=4, depth=3, content_dim=32)
        cfg['model_params']['length_regulator'].update(channels=32, in_channels=32)
    return cfg


def build(cfg):
    return build_model(recursive_munch(cfg['model_params']), stage='DiT')


class LoRAContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.cfg = config()
        self.model = build(self.cfg)
        self.base = self.path / 'synthetic_base.pth'
        torch.save({'net': {k: m.state_dict() for k, m in self.model.items()}}, self.base)
        self.base_hash = load_base_weights(self.model, self.base)
        self.signature = architecture_signature(self.model, self.cfg)
        self.settings = validate_config()

    def inject(self):
        targets = inject_lora(self.model, **self.settings)
        return make_metadata(self.model, self.base_hash, self.signature, self.settings, targets)

    @staticmethod
    def forward(model, inputs):
        net = model.cfm.estimator
        net.setup_caches(1, 16)
        net.eval()
        with torch.no_grad():
            return net(*inputs)

    def inputs(self):
        return (torch.randn(1, 128, 8), torch.randn(1, 128, 8),
                torch.tensor([8]), torch.tensor([0.5]), torch.randn(1, 192), torch.randn(1, 8, 32))

    def test_zero_adapter_preserves_real_dit_forward(self):
        inputs = self.inputs()
        before = self.forward(self.model, inputs)
        self.inject()
        after = self.forward(self.model, inputs)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        trainable = {id(p) for net in self.model.values() for p in net.parameters() if p.requires_grad}
        self.assertEqual(trainable, {id(p) for p in adapter_parameters(self.model).values()})
        self.assertTrue(all(not p.requires_grad for p in self.model.length_regulator.parameters()))

    def test_nonzero_adapter_roundtrip_and_inference_freeze(self):
        inputs = self.inputs()
        baseline = self.forward(self.model, inputs)
        metadata = self.inject()
        # Synthetic assignment tests serialization; this is not a training update.
        with torch.no_grad():
            for name, p in adapter_parameters(self.model).items():
                if name.endswith('lora_B'):
                    p.fill_(0.01)
        expected = self.forward(self.model, inputs)
        self.assertFalse(torch.allclose(expected, baseline))
        path = self.path / 'adapter.pth'
        save_adapter(self.model, path, metadata)
        payload = read_adapter(path)
        self.assertEqual(set(payload), {'metadata', 'adapter'})
        self.assertTrue(all(k.endswith(('lora_A', 'lora_B')) for k in payload['adapter']))
        fresh = build(self.cfg)
        base_hash = load_base_weights(fresh, self.base)
        self.assertEqual(attach_adapter(fresh, self.cfg, base_hash, path), metadata)
        torch.testing.assert_close(self.forward(fresh, inputs), expected, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for net in fresh.values() for p in net.parameters()))

    def test_optimizer_contains_only_adapters(self):
        self.inject()
        opt = build_optimizer(self.model, lr=1e-4, trainable_only=True)
        self.assertEqual(opt.keys, ['cfm'])
        actual = {id(p) for group in opt.param_groups for p in group['params']}
        self.assertEqual(actual, {id(p) for p in adapter_parameters(self.model).values()})
        self.assertFalse(opt.optimizers['cfm'].state)  # no step was executed

    def test_wrong_base_and_computation_settings_rejected(self):
        metadata = self.inject()
        path = self.path / 'adapter.pth'
        save_adapter(self.model, path, metadata)
        for bad_hash, cfg in [('0' * 64, self.cfg), (self.base_hash, copy.deepcopy(self.cfg))]:
            if bad_hash == self.base_hash:
                cfg['model_params']['DiT']['class_dropout_prob'] = 0.2
            fresh = build(cfg)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                attach_adapter(fresh, cfg, bad_hash, path)
            self.assertFalse(any(isinstance(m, LoRALinear) for m in fresh.cfm.modules()))

    def test_bad_tensors_are_rejected_before_any_copy(self):
        metadata = self.inject()
        path = self.path / 'adapter.pth'
        save_adapter(self.model, path, metadata)
        original = read_adapter(path)
        names = list(original['adapter'])
        before = {k: p.detach().clone() for k, p in adapter_parameters(self.model).items()}
        for corruption in ('missing', 'extra', 'shape', 'nan', 'target', 'rank'):
            payload = copy.deepcopy(original)
            payload['adapter'][names[0]].fill_(1)
            if corruption == 'missing': del payload['adapter'][names[-1]]
            elif corruption == 'extra': payload['adapter']['unknown.lora_A'] = torch.zeros(1)
            elif corruption == 'shape': payload['adapter'][names[-1]] = torch.zeros(1)
            elif corruption == 'nan': payload['adapter'][names[-1]].fill_(float('nan'))
            elif corruption == 'target': payload['metadata']['targets'][0]['out_features'] += 1
            elif corruption == 'rank': payload['metadata']['lora']['rank'] = 8
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                load_adapter_weights(self.model, payload, metadata)
            for k, p in adapter_parameters(self.model).items():
                torch.testing.assert_close(p, before[k], rtol=0, atol=0)

    def test_base_requires_complete_parameter_coverage(self):
        state = torch.load(self.base, weights_only=True)
        name = next(iter(dict(self.model.length_regulator.named_parameters())))
        del state['net']['length_regulator'][name]
        path = self.path / 'bad_base.pth'
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'Incomplete base'):
            load_base_weights(self.model, path)

    def test_legacy_unused_base_keys_are_allowed(self):
        state = torch.load(self.base, weights_only=True)
        state['net']['cfm']['legacy_unused.weight'] = torch.ones(1)
        path = self.path / 'extra_base.pth'
        torch.save(state, path)
        load_base_weights(self.model, path)

    def test_official_ddp_prefix_and_collision_detection(self):
        original = torch.load(self.base, weights_only=True)
        state = {'net': {key: {'module.' + name: value for name, value in values.items()}
                         for key, values in original['net'].items()}}
        path = self.path / 'ddp_base.pth'
        torch.save(state, path)
        fresh = build(self.cfg)
        load_base_weights(fresh, path)
        for key, net in self.model.items():
            for name, p in net.named_parameters():
                torch.testing.assert_close(p, dict(fresh[key].named_parameters())[name], rtol=0, atol=0)
        first = next(iter(original['net']['cfm']))
        state['net']['cfm'][first] = original['net']['cfm'][first]
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            load_base_weights(fresh, path)

    def test_duplicate_injection_and_bad_settings_rejected(self):
        self.inject()
        with self.assertRaises(ValueError): inject_lora(self.model)
        for args in ({'rank': 0}, {'rank': True}, {'alpha': float('nan')}, {'dropout': 1.0}):
            with self.assertRaises(ValueError): validate_config(**args)

    def test_full_svc_meta_parameter_count(self):
        with torch.device('meta'):
            model = build(config(small=False))
            targets = inject_lora(model)
        self.assertEqual(len(targets), 85)
        self.assertEqual(sum(p.numel() for p in adapter_parameters(model).values()), 3551232)

    def test_training_checkpoint_save_restore_without_training(self):
        from train import Trainer
        trainer = Trainer.__new__(Trainer)  # never call the dataset/model-building trainer constructor
        trainer.model = self.model
        trainer.lora_metadata = self.inject()
        trainer.optimizer = build_optimizer(self.model, lr=1e-4, trainable_only=True)
        trainer.log_dir = str(self.path)
        trainer.epoch, trainer.iters = 0, 0
        trainer.save_lora_checkpoint()
        state = read_adapter(self.path / 'lora_latest.pth')
        self.assertIn('training', state)
        self.assertNotIn('net', state)
        with torch.no_grad():
            for p in adapter_parameters(self.model).values(): p.fill_(0.3)
        trainer.restore_lora_checkpoint(self.path / 'lora_latest.pth')
        for name, p in adapter_parameters(self.model).items():
            torch.testing.assert_close(p, state['adapter'][name], rtol=0, atol=0)
        self.assertEqual(trainer.iters, 0)
        self.assertFalse(trainer.optimizer.optimizers['cfm'].state)
        save_adapter(self.model, self.path / 'adapter.pth', trainer.lora_metadata)
        with self.assertRaisesRegex(ValueError, 'training checkpoint'):
            trainer.restore_lora_checkpoint(self.path / 'adapter.pth')

    def test_inference_rejects_incomplete_options_before_loading_models(self):
        from argparse import Namespace
        from inference import load_models
        args = Namespace(fp16=True, lora_adapter='unused.pth', f0_condition=False,
                         checkpoint=None, config=None)
        with self.assertRaisesRegex(ValueError, 'f0-condition'):
            load_models(args)
        args.f0_condition = True
        with self.assertRaisesRegex(ValueError, 'explicit'):
            load_models(args)

    def test_trainer_initialization_lora_and_legacy_without_training(self):
        from train import Trainer
        cfg = copy.deepcopy(self.cfg)
        cfg['log_dir'] = str(self.path / 'runs')
        config_file = self.path / 'config.yml'
        config_file.write_text(yaml.safe_dump(cfg))
        # Mock only dataset/helpers; exercise actual model/base/adapter/optimizer wiring.
        replacements = {
            'build_sv_model': lambda obj, *a: setattr(obj, 'campplus_model', torch.nn.Linear(1, 1)),
            'build_semantic_fn': lambda obj, *a: setattr(obj, 'whisper_model', torch.nn.Linear(1, 1)),
            'build_f0_fn': lambda obj, *a: setattr(obj, 'rmvpe', SimpleNamespace(model=torch.nn.Linear(1, 1))),
            'build_converter': lambda obj, *a: setattr(obj, 'tone_color_converter', SimpleNamespace(model=torch.nn.Linear(1, 1))),
            'build_vocoder': lambda obj, *a: setattr(obj, 'bigvgan_model', torch.nn.Linear(1, 1)),
        }
        with ExitStack() as stack:
            stack.enter_context(patch('train.build_ft_dataloader', return_value=SimpleNamespace(dataset=[])))
            for name, fn in replacements.items(): stack.enter_context(patch.object(Trainer, name, fn))
            for enabled in (False, True):
                trainer = Trainer(str(config_file), str(self.base), 'unused', str(enabled),
                                  batch_size=1, steps=2, device='cpu', lora=enabled)
                self.assertEqual(trainer.iters, 0)
                if enabled:
                    self.assertEqual(trainer.optimizer.keys, ['cfm'])
                    self.assertEqual(len(trainer.lora_metadata['targets']), 15)
                    for name in ('campplus_model', 'whisper_model', 'bigvgan_model'):
                        self.assertTrue(all(not p.requires_grad for p in getattr(trainer, name).parameters()))
                else:
                    self.assertEqual(set(trainer.optimizer.keys), {'cfm', 'length_regulator'})
                    self.assertFalse(adapter_parameters(trainer.model))
                self.assertTrue(all(not opt.state for opt in trainer.optimizer.optimizers.values()))


if __name__ == '__main__':
    unittest.main()
