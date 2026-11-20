"""LoRA actions for the common four-stage voice-training controller."""
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

import yaml

from .audio import sha256, inspect_audio, decode
from .common.io import write_json, identity
from .common.paths import ROOT, read_run_state
from .common.voice_paths import portable_path, runtime_model_config, training_output
from .lora import checked_adapter, native_lora, pcm_identity
from .voice import BASE_CHECKPOINT, BASE_CONFIG


def path(value):
    if not str(value).strip():
        raise ValueError('Path must not be empty; check the config or shell variable')
    return (ROOT / Path(value).expanduser()).resolve()


def settings(c):
    return native_lora().validate_config(**c.get('lora', {}))


def check_config(c):
    if not isinstance(c, dict): raise ValueError('LoRA config must be a mapping')
    for key in ('voice_id', 'version', 'dataset_dir', 'reference', 'reference_group', 'reference_sha256', 'training', 'validation'):
        if key not in c: raise ValueError('Missing LoRA config key: ' + key)
    if c.get('adaptation_type') != 'lora': raise ValueError('Expected adaptation_type: lora')
    for key in ('voice_id', 'version'):
        if not isinstance(c[key], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', c[key]):
            raise ValueError(f'{key} must be a nonempty safe path component')
    if not isinstance(c['reference_group'], str) or not c['reference_group'].strip():
        raise ValueError('reference_group must identify the held-out reference recording')
    if not isinstance(c.get('smoke_test', False), bool): raise ValueError('smoke_test must be boolean')
    settings(c)
    t = c['training']
    if not isinstance(t, dict): raise ValueError('training must be a mapping')
    for key in ('max_steps', 'save_every', 'batch_size'):
        if isinstance(t.get(key), bool) or not isinstance(t.get(key), int) or t[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    lr = float(t.get('learning_rate', 1e-4))
    if not math.isfinite(lr) or not 0 < lr <= 1e-3: raise ValueError('Invalid LoRA learning_rate')
    t['learning_rate'] = lr
    seed = t.get('seed', 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Invalid training seed')
    if not re.fullmatch(r'\d+', str(c.get('gpu', '0'))):
        raise ValueError('gpu must be one nonnegative GPU index')
    if sha256(path(c['reference'])) != c['reference_sha256']:
        raise ValueError('Bound training reference changed')
    info = inspect_audio(path(c['reference']))
    if info['sample_rate'] != 44100 or info['channels'] != 1 or not 1 <= info['duration_s'] <= 25:
        raise ValueError('LoRA reference must be a prepared mono 44.1 kHz WAV, 1..25 seconds')
    native_lora().validate_svc_config(yaml.safe_load(path(c.get('base_config', BASE_CONFIG)).read_text()))
    if not path(c.get('base_checkpoint', BASE_CHECKPOINT)).is_file():
        raise ValueError('Base checkpoint file is missing; prepare the public Seed-VC weights first')
    source = validation_source(c)
    import soundfile as sf
    validate_windows(c['validation'], sf.info(source).duration)
    for row in c.get('sources', []):
        if row.get('split') != 'prompt' and row.get('group') == c['reference_group']:
            raise ValueError('Reference recording must be held out; remove its group from train/validation/test')
    return c


def validate_windows(validation, duration=None):
    """Check time windows before training, instead of failing after 1,000 steps."""
    if not isinstance(validation, dict): raise ValueError('validation must be a mapping')
    windows = validation.get('windows', [{'start': 0, 'duration': 12}])
    if not isinstance(windows, list) or not windows:
        raise ValueError('validation.windows must contain at least one start/duration window')
    for index, window in enumerate(windows, 1):
        if not isinstance(window, dict): raise ValueError(f'Validation window {index} must be a mapping')
        start, length = window.get('start'), window.get('duration')
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, length))
                or start < 0 or length <= 0):
            raise ValueError(f'Invalid validation window {index}: start >= 0 and duration > 0 are required')
        if duration is not None and start + length > duration + 1e-6:
            raise ValueError(f'Validation window {index} ends at {start + length:g}s but source audio is only {duration:g}s')
    steps, seed = validation.get('steps', 30), validation.get('seed', 42)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('validation.steps must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('validation.seed must be an integer in [0, 2**32)')
    return windows


def validation_source(c):
    v = c['validation']
    if not isinstance(v, dict): raise ValueError('validation must be a mapping')
    if bool(v.get('source_run')) == bool(v.get('source_audio')):
        raise ValueError('Set exactly one validation.source_run or validation.source_audio')
    if v.get('source_run'):
        run = path(v['source_run'])
        state = read_run_state(run)
        row = state.get('stages', {}).get('04_vocal_separation', {})
        if row.get('status') != 'succeeded':
            raise ValueError(f'Validation source requires a successful song stage 04: {run}')
        artifact = row['outputs']['vocals']
        source, expected = Path(artifact['path']), artifact['sha256']
    else:
        source, expected = path(v['source_audio']), v.get('source_sha256')
    if not source.is_file(): raise ValueError(f'Validation source audio is missing: {source}')
    if not expected or sha256(source) != expected:
        raise ValueError(f'Validation source changed: {source}; verify its provenance before rebuilding the config')
    return source


def validation_reports(run, training_identity):
    """Only sealed successful attempts can authorize a held-out test."""
    reports = []
    attempts = [p for p in (run / 'stages/02_validate').glob('attempt_*') if p.name[8:].isdigit()]
    for attempt in sorted(attempts, key=lambda p: int(p.name[8:])):
        file, marker = attempt / 'artifacts/validation.json', attempt / 'outputs.json'
        if not file.is_file() or not marker.is_file() or (attempt / 'failure.json').exists(): continue
        sealed = json.loads(marker.read_text())
        if sealed.get('status') != 'succeeded': continue
        item = sealed.get('artifacts', {}).get('validation.json', {})
        if item.get('sha256') != sha256(file) or item.get('bytes') != file.stat().st_size:
            raise ValueError(f'Validation report checksum mismatch: {file}; restore the sealed report from a verified backup, do not rewrite its checksum')
        report = json.loads(file.read_text())
        if report.get('evaluation_split') == 'validation' and report.get('training_identity') == training_identity:
            reports.append((file, report))
    return reports


def validation_candidates(run, args, trained=None):
    """Resolve explicit/default checkpoints without loading GPU models or writing state."""
    if trained is None:
        trained = json.loads((training_output(run, '01_train') / 'training.json').read_text())
    raw = args.checkpoint
    if raw is not None and (not raw or any(not str(value).strip() for value in raw)):
        raise ValueError('--checkpoint is empty. Set VC_CKPT="$VC_TRAIN_RUN/checkpoints/adapter.pth" or omit --checkpoint.')
    is_test = args.evaluation_split == 'test'
    reports = validation_reports(run, trained['training_identity']) if is_test else []
    if is_test and not reports:
        raise ValueError('No successful validation selection exists for this training run. '
                         'Run 02_validate with --evaluation-split validation first, then retry test with the same checkpoint. '
                         'No retraining is needed.')
    default = reports[-1][1]['selected_adapter'] if is_test else trained['adapter']
    candidates = [path(value) for value in raw] if raw else [path(default)]
    if len(set(candidates)) != len(candidates): raise ValueError('Duplicate validation checkpoints')
    if is_test and len(candidates) != 1:
        raise ValueError('Test accepts exactly one checkpoint; compare candidates on validation first')
    for candidate in candidates:
        if not candidate.is_file(): raise ValueError(f'Checkpoint file does not exist: {candidate}')
    if is_test and not any(sha256(candidates[0]) == r['selected_adapter_sha256'] for _, r in reports):
        raise ValueError('Test checkpoint differs from the validation selection. '
                         f'Latest selected adapter: {reports[-1][1]["selected_adapter"]}. '
                         'Use that file (or omit --checkpoint), or validate the intended checkpoint first.')
    return candidates


def dataset_manifest(c):
    import numpy as np
    import soundfile as sf
    folder = path(c['dataset_dir'])
    file = folder / 'manifest.json'
    m = json.loads(file.read_text())
    if m['voice_id'] != c['voice_id'] or bool(m.get('smoke_test')) != bool(c.get('smoke_test')):
        raise ValueError('Dataset voice/smoke status differs')
    # Existing splits retain their identity, even when an explicitly bound prompt changes.
    if c.get('dataset_manifest_sha256') and sha256(file) != c['dataset_manifest_sha256']:
        raise ValueError('Existing dataset manifest changed')
    if sha256(path(c['reference'])) != c['reference_sha256']:
        raise ValueError('Bound training reference changed')
    groups, hashes = {}, {}
    for row in m['recordings']:
        split = row['split']
        if split not in {'train', 'validation', 'test', 'prompt'}:
            raise ValueError('Invalid recording split')
        if groups.setdefault(row['group'], split) != split or hashes.setdefault(row['sha256'], split) != split:
            raise ValueError('Recording/group leakage between dataset splits')
        if split != 'prompt' and (row['group'] == c['reference_group'] or row['sha256'] == c['reference_sha256']):
            raise ValueError('Reference recording must be held out')
    seen, clip_hashes = set(), {}
    for row in m['clips']:
        raw_file = folder / row['path']
        file = raw_file.resolve()
        if not file.is_relative_to(folder) or row['path'] in seen or raw_file.is_symlink():
            raise ValueError('Invalid/duplicate dataset clip path')
        seen.add(row['path'])
        if groups.get(row['group']) != row['split'] or Path(row['path']).parts[0] != row['split']:
            raise ValueError('Clip split/group differs from its recording')
        if clip_hashes.setdefault(row['sha256'], row['split']) != row['split']:
            raise ValueError('Identical clip leaked across splits')
        if sha256(file) != row['sha256']: raise ValueError('Dataset clip changed: ' + str(file))
        audio, rate = sf.read(file, dtype='float32', always_2d=True)
        if (rate != 44100 or audio.shape[1] != 1 or not 1 <= len(audio) / rate <= 30
                or not np.isfinite(audio).all() or np.sqrt(np.mean(audio.astype('float64') ** 2)) <= 1e-4):
            raise ValueError('Invalid/silent training clip: ' + str(file))
    required = {'train'} if c.get('smoke_test') else {'train', 'validation', 'test'}
    if not required <= {r['split'] for r in m['clips']}: raise ValueError('Missing required dataset split')
    expected = {(folder / r['path']).resolve() for r in m['clips'] if r['split'] == 'train'}
    actual = {p.resolve() for p in (folder / 'train').rglob('*') if p.is_file()}
    if actual != expected: raise ValueError('Train directory differs from the manifest (extra/missing files)')
    return m


def prepare(c, run, out, args):
    if not (path(c['dataset_dir']) / 'manifest.json').is_file():
        if not c.get('sources'): raise ValueError('New datasets require sources with explicit group/split')
        from .voice_training import prepare as prepare_full
        prepare_full(c, run, out, args)
    m = dataset_manifest(c)
    write_json(out / 'manifest.json', m)
    write_json(out / 'preparation.json', {'dataset_identity': m['identity'],
        'manifest_sha256': sha256(path(c['dataset_dir']) / 'manifest.json'),
        'reference_sha256': c['reference_sha256'], 'reference_group': c['reference_group'],
        'original_manifest_reference_sha256': m['reference_sha256'],
        'split_counts': dict(Counter(r['split'] for r in m['clips'])), 'training_performed': False})


def code_identity():
    repo = ROOT / 'third_party/seed-vc'
    return {name: sha256(repo / name) for name in ('train.py', 'inference.py', 'optimizers.py', 'modules/lora.py')}


def prepared_dataset(c, run):
    m = dataset_manifest(c)
    prepared = json.loads((training_output(run, '00_prepare') / 'preparation.json').read_text())
    if (prepared['manifest_sha256'] != sha256(path(c['dataset_dir']) / 'manifest.json')
            or prepared['dataset_identity'] != m['identity'] or prepared['reference_sha256'] != c['reference_sha256']
            or prepared['reference_group'] != c['reference_group']):
        raise ValueError('Dataset/reference changed since 00_prepare; prepare again')
    return m


def resume_checkpoint(checkpoints, key, base_hash, lora, max_steps):
    ledger = json.loads((checkpoints / 'identity.json').read_text())
    latest = checkpoints / 'lora_latest.pth'
    if ledger['training_identity'] != key or ledger['files'].get(latest.name) != sha256(latest):
        raise ValueError('Resume dataset/base/config/code/checkpoint mismatch')
    restored = checked_adapter(latest, base_hash, lora)
    if 'training' not in restored: raise ValueError('Resume requires optimizer/RNG state')
    if max_steps <= restored['training']['iters']: raise ValueError('max_steps must exceed restored step')
    return restored


def import_existing(c, run, out, source):
    """Import an audited standalone run without executing any training step."""
    source = path(source)
    record = json.loads((source / 'training.json').read_text())
    audit = json.loads((source / 'training_audit.json').read_text())
    inputs = json.loads((source / 'inputs.json').read_text())
    m = prepared_dataset(c, run)
    base_hash = sha256(path(c.get('base_checkpoint', BASE_CHECKPOINT)))
    file = source / 'adapter.pth'
    payload = checked_adapter(file, base_hash, settings(c))
    if (record.get('status') != 'succeeded' or record['dataset_identity'] != m['identity']
            or record['dataset_manifest_sha256'] != sha256(path(c['dataset_dir']) / 'manifest.json')
            or record['base_sha256'] != base_hash or record['adapter_sha256'] != sha256(file)
            or inputs['inputs']['reference']['sha256'] != c['reference_sha256']
            or audit.get('status') != 'passed' or not audit.get('only_lora_optimized')
            or not audit.get('frozen_core_unchanged') or audit['iters'] != record['training_step']):
        raise ValueError('Standalone training evidence does not match this voice/dataset/adapter')
    config_file = path(record['config'])
    if sha256(config_file) != record['config_sha256']: raise ValueError('Imported model config changed')
    # Architecture compatibility is checked again by actual inference before export.
    destination = run / 'checkpoints'
    destination.mkdir(exist_ok=False)
    shutil.copy2(file, destination / 'adapter.pth')
    shutil.copy2(config_file, run / 'model_config.yml')
    evidence = out / 'imported_evidence'; evidence.mkdir()
    for name in ('training.json', 'training_audit.json', 'inputs.json'):
        shutil.copy2(source / name, evidence / name)
    key = identity({'adapter': sha256(file), 'dataset': m['identity'], 'base': base_hash, 'source': str(source)})
    write_json(out / 'training.json', {'training_identity': key, 'dataset_identity': m['identity'],
        'mode': 'imported_lora', 'training_performed': False, 'source_run': str(source),
        'end_step': record['training_step'], 'adapter': portable_path(destination / 'adapter.pth'),
        'adapter_sha256': sha256(file), 'base_sha256': base_hash,
        'reference_sha256': c['reference_sha256'], 'adapter_metadata': payload['metadata'],
        'model_config_sha256': sha256(run / 'model_config.yml'), 'source_training': record,
        'training_audit': audit, 'code_sha256': code_identity()})


def frozen_digest(models):
    digest = hashlib.sha256()
    for group, net in models.items():
        for name, value in net.named_parameters():
            if not value.requires_grad:
                digest.update((group + '.' + name).encode())
                digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def train(c, run, out, args):
    if getattr(args, 'import_run', None):
        if args.resume: raise ValueError('Import and resume are mutually exclusive')
        return import_existing(c, run, out, args.import_run)
    import numpy as np
    import torch
    from .backends.voice_train import verify_training_assets, local_asset
    from .backends.seed_vc_runtime import configure_bigvgan_loader
    verify_training_assets()
    m, t, lora = prepared_dataset(c, run), c['training'], settings(c)
    base = path(c.get('base_checkpoint', BASE_CHECKPOINT))
    source_config = path(c.get('base_config', BASE_CONFIG))
    stable = {'dataset_identity': m['identity'], 'manifest_sha256': sha256(path(c['dataset_dir']) / 'manifest.json'),
        'base_sha256': sha256(base), 'config_sha256': sha256(source_config), 'reference_sha256': c['reference_sha256'],
        'lora': lora, 'training': {k: v for k, v in t.items() if k not in ('max_steps', 'save_every')},
        'code': code_identity(), 'controller_sha256': sha256(__file__)}
    key = identity(stable)
    checkpoints = run / 'checkpoints'
    ledger_file = checkpoints / 'identity.json'
    latest = checkpoints / 'lora_latest.pth'
    if args.resume:
        resume_checkpoint(checkpoints, key, stable['base_sha256'], lora, t['max_steps'])
    elif checkpoints.exists():
        raise ValueError('Training run already has checkpoints; use --resume or a new run')
    else:
        checkpoints.mkdir()
    cfg = yaml.safe_load(source_config.read_text())
    cfg['log_dir'] = str(run); cfg['pretrained_model'] = ''
    cfg['model_params']['vocoder']['name'] = 'models/BigVGAN'
    cfg['model_params']['speech_tokenizer']['name'] = 'models/Whisper-small'
    (run / 'model_config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    runtime = runtime_model_config(run / 'model_config.yml', out / 'runtime_model_config.yml')
    seed = int(t.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    repo = ROOT / 'third_party/seed-vc'
    os.chdir(repo); sys.path.insert(0, str(repo))
    configure_bigvgan_loader()
    import train as native
    native.load_custom_model_from_hf = local_asset
    trainer = native.Trainer(str(runtime), str(base), str(path(c['dataset_dir']) / 'train'), 'checkpoints',
        batch_size=t['batch_size'], num_workers=0, steps=t['max_steps'], save_interval=t['save_every'], device='cuda:0',
        lora=True, lora_rank=lora['rank'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
        lora_learning_rate=t['learning_rate'], lora_resume=str(latest) if args.resume else None)
    from modules.lora import adapter_parameters
    parameters = adapter_parameters(trainer.model)
    optimized = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
    if optimized != {id(p) for p in parameters.values()}: raise ValueError('Optimizer includes non-LoRA parameters')
    expected = {str(path(c['dataset_dir']) / r['path']) for r in m['clips'] if r['split'] == 'train'}
    if set(trainer.train_dataloader.dataset.data) != expected: raise ValueError('Trainer dataset differs from manifest')
    before = frozen_digest(trainer.model)
    started, start = trainer.iters, time.monotonic()
    original_step, original_save = trainer.train_one_step, trainer.save_lora_checkpoint
    def step(batch):
        loss = original_step(batch)
        if not math.isfinite(loss): raise ValueError('Non-finite LoRA training loss')
        row = {'step': trainer.iters + 1, 'loss': float(loss), 'elapsed_seconds': time.monotonic() - start}
        with (run / 'loss.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        write_json(out / 'progress.json', row)
        return loss
    def save():
        original_save()
        files = {p.name: sha256(p) for p in checkpoints.glob('lora_*.pth')}
        write_json(ledger_file, {'training_identity': key, 'inputs': stable, 'files': files})
    trainer.train_one_step, trainer.save_lora_checkpoint = step, save
    trainer.train()
    after = frozen_digest(trainer.model)
    if before != after or sha256(base) != stable['base_sha256']:
        raise ValueError('Frozen base changed during LoRA training')
    if trainer.iters != t['max_steps']: raise ValueError('Trainer did not reach requested step count')
    adapter = checkpoints / 'adapter.pth'
    payload = checked_adapter(adapter, stable['base_sha256'], lora)
    write_json(out / 'training.json', {'training_identity': key, 'inputs': stable, 'dataset_identity': m['identity'],
        'mode': 'lora', 'training_performed': True, 'start_step': started, 'end_step': trainer.iters,
        'adapter': portable_path(adapter), 'adapter_sha256': sha256(adapter), 'base_sha256': stable['base_sha256'],
        'reference_sha256': c['reference_sha256'], 'adapter_metadata': payload['metadata'],
        'model_config_sha256': sha256(run / 'model_config.yml'),
        'trainable_parameters': sum(p.numel() for p in parameters.values()),
        'frozen_core_before': before, 'frozen_core_after': after, 'only_lora_optimized': True,
        'resumed': args.resume, 'resume_sample_order': 'saved epoch restarts; not bitwise mid-epoch resume',
        'elapsed_seconds': time.monotonic() - start})


def training_record(c, run):
    m = prepared_dataset(c, run)
    trained = json.loads((training_output(run, '01_train') / 'training.json').read_text())
    if (trained['dataset_identity'] != m['identity'] or trained['reference_sha256'] != c['reference_sha256']
            or trained['base_sha256'] != sha256(path(c.get('base_checkpoint', BASE_CHECKPOINT)))
            or trained['model_config_sha256'] != sha256(run / 'model_config.yml')):
        raise ValueError('Training identity/config/reference changed')
    return m, trained


def validate(c, run, out, args):
    m, trained = training_record(c, run)
    candidates = validation_candidates(run, args, trained)
    base, config = path(c.get('base_checkpoint', BASE_CHECKPOINT)), run / 'model_config.yml'
    split = args.evaluation_split
    ledger = json.loads((run / 'checkpoints/identity.json').read_text()) if (run / 'checkpoints/identity.json').exists() else None
    for candidate in candidates:
        digest = sha256(candidate)
        if digest != trained['adapter_sha256'] and not (ledger and ledger['training_identity'] == trained['training_identity']
                and candidate.parent == run / 'checkpoints' and ledger['files'].get(candidate.name) == digest):
            raise ValueError('Validation adapter is not from this training run')
        checked_adapter(candidate, sha256(base), settings(c))
    v = c['validation']
    source = validation_source(c)
    validate_windows(v, inspect_audio(source)['duration_s'])
    if v.get('full_profile'):
        from .voice import resolve_voice
        full_voice = resolve_voice('finetuned', v['full_profile'])
        if full_voice['voice_id'] != c['voice_id']:
            raise ValueError('Full-finetuned comparison profile belongs to another voice')
    refs = [path(c['dataset_dir']) / r['path'] for r in m['clips'] if r['split'] == split]
    if not refs:
        if not c.get('smoke_test'): raise ValueError('No held-out evaluation references')
        refs = [path(c['reference'])]
    from .backends.voice_metrics import VoiceMetrics
    from .voice_candidates import assessment
    metrics = VoiceMetrics()
    def infer(source_file, folder, adapter=None, full_profile=None):
        folder.mkdir(parents=True)
        argv = [sys.executable, '-m', 'music_cover.backends.cli', 'convert', '--input', str(source_file),
            '--reference', str(path(c['reference'])), '--output', str(folder), '--steps', str(v.get('steps', 30)),
            '--seed', str(v.get('seed', 42))]
        if full_profile: argv += ['--voice-mode', 'finetuned', '--voice-profile', str(path(full_profile))]
        else: argv += ['--checkpoint', str(base), '--model-config', str(config)]
        if adapter: argv += ['--lora-adapter', str(adapter)]
        write_json(folder / 'command.json', {'argv': argv})
        with (folder / 'inference.log').open('w') as log:
            subprocess.run(argv, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=v.get('timeout_seconds', 10800))
        return folder / 'converted.wav'
    cases, baseline = [], []
    for index, window in enumerate(v.get('windows', [{'start': 0, 'duration': 12}])):
        folder = out / f'case_{index + 1:03d}'; folder.mkdir()
        audio = folder / 'source.wav'
        decode(source, audio, rate=44100, channels=1, start=window['start'], duration=window['duration'])
        zero = infer(audio, folder / 'zero_shot')
        row = {'case': index + 1, 'source_sha256': sha256(audio), 'zero_shot': metrics.compare(audio, zero, refs, folder / 'zero_shot/metrics')}
        if v.get('full_profile'):
            full = infer(audio, folder / 'finetuned', full_profile=v['full_profile'])
            row['finetuned'] = metrics.compare(audio, full, refs, folder / 'finetuned/metrics')
        baseline.append(row); cases.append((audio, folder))
    results = []
    for index, adapter in enumerate(candidates):
        rows = []
        for audio, folder in cases:
            directory = folder / f'lora_{index + 1:03d}'
            converted = infer(audio, directory, adapter)
            result = metrics.compare(audio, converted, refs, directory / 'metrics')
            result['assessment'] = assessment(result, inspect_audio(audio)['duration_s'])
            result['conversion'] = json.loads((directory / 'conversion.json').read_text())
            rows.append(result)
        finite = [r['speaker_similarity'] for r in rows if r.get('speaker_similarity') is not None and math.isfinite(r['speaker_similarity'])]
        results.append({'adapter': portable_path(adapter), 'adapter_sha256': sha256(adapter), 'cases': rows,
            'eligible': all(r['assessment']['status'] == 'passed' for r in rows),
            'mean_speaker_similarity': sum(finite) / len(finite) if finite else None})
    selected = max(results, key=lambda r: (r['eligible'], r['mean_speaker_similarity'] if r['mean_speaker_similarity'] is not None else -1))
    write_json(out / 'validation.json', {'schema_version': 2, 'adaptation_type': 'lora',
        'training_identity': trained['training_identity'], 'dataset_identity': m['identity'], 'evaluation_split': split,
        'reference_sha256': c['reference_sha256'], 'base_sha256': sha256(base), 'config_sha256': sha256(config),
        'reference_independent': not bool(c.get('smoke_test')), 'baseline': baseline, 'candidates': results,
        'selected_adapter': selected['adapter'], 'selected_adapter_sha256': selected['adapter_sha256'],
        'assessment_status': 'passed' if selected['eligible'] else 'review_required',
        'subjective_quality': 'awaiting_human_review', 'training_performed': False,
        'metric_note': 'Speaker similarity against held-out references; not identity accuracy or listening MOS.'})
    sections = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>LoRA 验证试听</title>',
        '<h1>LoRA 验证试听</h1><p>固定源音频、参考音色与随机种子。以下是干声；主观音色和音质仍需试听。</p>']
    for index, (_, folder) in enumerate(cases, 1):
        sections.append(f'<h2>片段 {index}</h2>')
        for variant in ['zero_shot'] + (['finetuned'] if v.get('full_profile') else []) + [f'lora_{i + 1:03d}' for i in range(len(candidates))]:
            audio = (folder / variant / 'converted.wav').relative_to(out).as_posix()
            sections.append(f'<p>{html.escape(variant)}</p><audio controls src="{html.escape(audio)}"></audio>')
    sections.append('<p><a href="validation.json">定量指标与完整运行记录</a></p></html>')
    (out / 'validation.html').write_text('\n'.join(sections))


def export(c, run, out, args):
    m, trained = training_record(c, run)
    report_file = training_output(run, '02_validate') / 'validation.json'
    report = json.loads(report_file.read_text())
    adapter = path(args.checkpoint[0]) if args.checkpoint else path(report['selected_adapter'])
    digest = sha256(adapter)
    if len(args.checkpoint or []) > 1: raise ValueError('Export accepts one adapter')
    row = next((r for r in report['candidates'] if r['adapter_sha256'] == digest), None)
    if (row is None or report['training_identity'] != trained['training_identity']
            or report['dataset_identity'] != m['identity'] or report['reference_sha256'] != c['reference_sha256']
            or report['base_sha256'] != trained['base_sha256'] or report['config_sha256'] != sha256(run / 'model_config.yml')):
        raise ValueError('Adapter has not been validated with these inputs')
    payload = checked_adapter(adapter, trained['base_sha256'], settings(c))
    version = path(c.get('export_dir', f"models/voices/{c['voice_id']}/seed_vc_lora/{c['version']}"))
    if version.exists(): raise ValueError('Exported voice versions are immutable; choose a new version')
    temporary = version.with_name(version.name + '.exporting'); temporary.mkdir(parents=True, exist_ok=False)
    # Strip optimizer/RNG state when exporting an intermediate training checkpoint.
    if 'training' in payload:
        native_lora().atomic_save({'metadata': payload['metadata'], 'adapter': payload['adapter']}, temporary / 'adapter.pth')
    else: shutil.copy2(adapter, temporary / 'adapter.pth')
    cfg = yaml.safe_load((run / 'model_config.yml').read_text())
    cfg['log_dir'] = 'runs'; cfg['pretrained_model'] = ''
    cfg['model_params']['vocoder']['name'] = 'models/BigVGAN'
    cfg['model_params']['speech_tokenizer']['name'] = 'models/Whisper-small'
    (temporary / 'config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy2(path(c['reference']), temporary / 'reference.wav')
    shutil.copy2(report_file, temporary / 'validation.json')
    shutil.copy2(training_output(run, '01_train') / 'training.json', temporary / 'training.json')
    def record(name): return {'path': name, 'sha256': sha256(temporary / name)}
    profile = {'schema_version': 2, 'backend': 'seed_vc', 'adaptation_type': 'lora', 'adaptation_scope': 'speaker',
        'voice_id': c['voice_id'], 'version': c['version'], 'dataset_identity': m['identity'],
        'training_steps': payload.get('training', {}).get('iters', trained['end_step']),
        'training_run': portable_path(run), 'adapter_metadata': payload['metadata'],
        'base_checkpoint': {'path': portable_path(path(c.get('base_checkpoint', BASE_CHECKPOINT))), 'sha256': trained['base_sha256']},
        'adapter': record('adapter.pth'), 'config': record('config.yml'), 'reference': record('reference.wav'),
        'validation': record('validation.json'), 'training': record('training.json'),
        'reference_pcm': pcm_identity(temporary / 'reference.wav'), 'reference_required': True,
        'reference_source': {'path': portable_path(path(c['reference'])), 'sha256': c['reference_sha256'], 'group': c['reference_group']},
        'validation_status': 'candidate_requires_listening' if row['eligible'] else 'quality_review_required',
        'subjective_quality': 'awaiting_human_review', 'implementation_sha256': code_identity(),
        'validated_training_adapter_sha256': digest}
    write_json(temporary / 'voice.json', profile)
    temporary.rename(version)
    write_json(out / 'export.json', {'voice_profile': portable_path(version / 'voice.json'), 'training_performed': False, **profile})
