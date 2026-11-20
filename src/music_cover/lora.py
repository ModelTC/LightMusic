"""Seed-VC LoRA profile and artifact checks; no model allocation on resolution."""
import hashlib
import importlib.util
import json
from pathlib import Path

from .audio import sha256
from .common.paths import ROOT
from .common.voice_profiles import verified_files


def native_lora(root=ROOT):
    file = Path(root) / 'third_party/seed-vc/modules/lora.py'
    spec = importlib.util.spec_from_file_location('music_cover_native_lora', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_adapter(file, base_hash, settings=None, root=ROOT):
    """Validate all serialized tensor keys/shapes before publication (CPU only)."""
    import torch
    native = native_lora(root)
    payload = native.read_adapter(file)
    meta = payload['metadata']
    if meta['base_sha256'] != base_hash:
        raise ValueError('LoRA base checkpoint SHA256 mismatch')
    if settings is not None and meta['lora'] != settings:
        raise ValueError('LoRA settings do not match the requested configuration')
    expected = {}
    rank = meta['lora']['rank']
    for target in meta['targets']:
        name = target['name']
        if not native.TARGET.fullmatch(name) or name + '.lora_A' in expected:
            raise ValueError('Invalid/duplicate LoRA target')
        expected[name + '.lora_A'] = (rank, target['in_features'])
        expected[name + '.lora_B'] = (target['out_features'], rank)
    weights = payload['adapter']
    if not expected or set(weights) != set(expected):
        raise ValueError('LoRA tensor keys mismatch')
    for name, shape in expected.items():
        value = weights[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point() or not torch.isfinite(value).all().item()):
            raise ValueError('Invalid LoRA tensor: ' + name)
    if sum(v.numel() for v in weights.values()) != meta['adapter_parameters']:
        raise ValueError('LoRA parameter count mismatch')
    return payload


def pcm_identity(file):
    import soundfile as sf
    import numpy as np
    data, rate = sf.read(file, dtype='float32', always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError('Empty or non-finite reference')
    return {'sample_rate': rate, 'channels': data.shape[1], 'frames': len(data),
            'sha256': hashlib.sha256(data.astype('<f4').tobytes()).hexdigest()}


def check_reference(actual, bound):
    if pcm_identity(actual) != pcm_identity(bound):
        raise ValueError('LoRA reference differs from the exported voice. Use the bound reference '
                         'in stage 00, or explicitly set --voice-reference-policy profile '
                         'to reuse an existing arrangement with this voice reference.')


def resolve_lora(profile, root=ROOT):
    if not profile:
        raise ValueError('lora requires --voice-profile; training is never implicit')
    root = Path(root)
    file = (root / Path(profile).expanduser()).resolve()
    data = json.loads(file.read_text())
    if (data.get('schema_version') != 2 or data.get('backend') != 'seed_vc'
            or data.get('adaptation_type') != 'lora' or data.get('adaptation_scope') != 'speaker'):
        raise ValueError('Expected a schema v2 Seed-VC LoRA voice profile')
    files = verified_files(file, data, ('adapter', 'config', 'reference', 'validation', 'training'), contained=True)
    base = (root / data['base_checkpoint']['path']).resolve()
    if not base.is_file() or sha256(base) != data['base_checkpoint']['sha256']:
        raise ValueError('LoRA base checkpoint missing or checksum mismatch')
    if data['adapter_metadata']['base_sha256'] != data['base_checkpoint']['sha256']:
        raise ValueError('LoRA profile base identity mismatch')
    if pcm_identity(files['reference']) != data['reference_pcm']:
        raise ValueError('LoRA reference PCM mismatch')
    return {'mode': 'lora', 'backend': 'seed_vc', 'profile': file, 'checkpoint': base, **files,
            'voice_id': data['voice_id'], 'version': data['version'],
            'adapter_metadata': data['adapter_metadata'], 'dataset_identity': data['dataset_identity'],
            'validation_status': data['validation_status'], 'training_steps': data['training_steps']}
