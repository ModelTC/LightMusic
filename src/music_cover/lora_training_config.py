"""Create or preview a LoRA config with checked data, reference and validation inputs."""
import argparse
import re
import yaml
from .audio import sha256
from .common.cli import nonempty
from .common.voice_paths import portable_path
from .lora_training import path, check_config, dataset_manifest
from .voice import BASE_CHECKPOINT, BASE_CONFIG


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('voice-id', 'version', 'reference', 'reference-group', 'output'):
        p.add_argument('--' + name, required=True, type=nonempty)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument('--dataset-dir', type=nonempty, help='Existing manifest dataset; split identity is preserved')
    data.add_argument('--data-config', type=nonempty, help='Preparation YAML with dataset_dir/sources and explicit group/split')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-run', type=nonempty, help='Existing song run with successful stage 04; evaluation only')
    source.add_argument('--validation-source', type=nonempty, help='Fixed source vocal WAV for evaluation')
    p.add_argument('--validation-windows', nargs='+', default=['36:12', '156:12'], help='start:duration in seconds; all windows must fit the source')
    p.add_argument('--validation-steps', type=int, default=30)
    p.add_argument('--validation-seed', type=int, help='Defaults to --seed')
    p.add_argument('--full-profile', type=nonempty, help='Optional same-voice full-finetuned comparison')
    p.add_argument('--base-checkpoint', type=nonempty, default=BASE_CHECKPOINT)
    p.add_argument('--base-config', type=nonempty, default=BASE_CONFIG)
    p.add_argument('--export-dir', type=nonempty)
    p.add_argument('--gpu', default='0', type=nonempty)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=float, default=16)
    p.add_argument('--dropout', type=float, default=0)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--plan', action='store_true', help='Check inputs and print config without creating files')
    return p


def build_config(a):
    for value in (a.voice_id, a.version):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
            raise ValueError('voice-id/version must be safe path components')
    output = path(a.output)
    if output.exists(): raise ValueError('Config already exists; choose a new output')
    raw = yaml.safe_load(path(a.data_config).read_text()) if a.data_config else {'dataset_dir': a.dataset_dir}
    if not isinstance(raw, dict): raise ValueError('data-config must be a mapping')
    c = {k: raw[k] for k in ('dataset_dir', 'sources', 'segment_seconds') if k in raw}
    if not c.get('dataset_dir'): raise ValueError('data-config requires dataset_dir')
    c['dataset_dir'] = portable_path(path(c['dataset_dir']))
    windows = []
    for text in a.validation_windows:
        try:
            start, duration = map(float, text.split(':'))
        except ValueError:
            raise ValueError(f'Invalid window {text!r}; use start:duration, for example 36:12') from None
        windows.append({'start': start, 'duration': duration})
    validation = {'windows': windows, 'steps': a.validation_steps,
                  'seed': a.seed if a.validation_seed is None else a.validation_seed}
    if a.source_run: validation['source_run'] = portable_path(path(a.source_run))
    else:
        validation.update(source_audio=portable_path(path(a.validation_source)), source_sha256=sha256(path(a.validation_source)))
    if a.full_profile:
        from .voice import resolve_voice
        if resolve_voice('finetuned', a.full_profile)['voice_id'] != a.voice_id:
            raise ValueError('Full-finetuned comparison profile belongs to another voice')
        validation['full_profile'] = portable_path(path(a.full_profile))
    c.update(adaptation_type='lora', voice_id=a.voice_id, version=a.version, gpu=a.gpu, smoke_test=a.smoke_test,
        reference=portable_path(path(a.reference)), reference_sha256=sha256(path(a.reference)), reference_group=a.reference_group,
        base_checkpoint=portable_path(path(a.base_checkpoint)), base_config=portable_path(path(a.base_config)),
        lora={'rank': a.rank, 'alpha': a.alpha, 'dropout': a.dropout},
        training={'max_steps': a.max_steps, 'save_every': a.save_every, 'batch_size': a.batch_size, 'seed': a.seed, 'learning_rate': a.learning_rate},
        validation=validation, export_dir=portable_path(path(a.export_dir or f'models/voices/{a.voice_id}/seed_vc_lora/{a.version}')))
    folder, export = path(c['dataset_dir']), path(c['export_dir'])
    if folder == export or folder in export.parents or export in folder.parents or output.is_relative_to(folder) or output.is_relative_to(export):
        raise ValueError('Config, dataset and export locations must not contain one another')
    if export.exists(): raise ValueError('Export version already exists; choose a new version/export-dir')
    manifest = folder / 'manifest.json'
    if manifest.exists():
        c['dataset_manifest_sha256'] = sha256(manifest)
        dataset_manifest(c)
    elif not c.get('sources'): raise ValueError('No dataset manifest and no preparation sources')
    check_config(c)
    return c, output


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        config, output = build_config(args)
        if args.plan:
            print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            print(f'PLAN: checks passed; no files created. Config target: {output}')
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as f: yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        print(f'Created LoRA config: {output}')
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        p.error(str(error))


if __name__ == '__main__': main()
