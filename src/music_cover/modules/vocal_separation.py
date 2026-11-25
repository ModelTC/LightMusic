import json
import shutil
import subprocess

import soundfile as sf

from ..audio import decode
from ..common.io import write_json
from ..common.resources import requirements

ID = '04_vocal_separation'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['audio.py', 'backends/separator.py', 'backends/lead_duo.py']
TIMELINE = {'*': 'generated'}


def backends(r, c):
    return ['separator', 'lead_duo']


def requires(r):
    if r.get('arrangement') == 'original':
        return ['00_prepare', '02_pitch_adaptation']
    return ['03_style_generation']


def inputs(ctx):
    files = {str(path.relative_to(ctx.root)): path for path in requirements(ctx.root, 'lead_duo')[1:]}
    if ctx.request.get('arrangement') == 'original':
        files['cover'] = ctx.output('00_prepare', 'source')
        files['plan'] = ctx.output('02_pitch_adaptation', 'plan')
        if str(ctx.request['transpose']) != '0':
            for role in ('vocals', 'instrumental'):
                files[role] = ctx.output('02_pitch_adaptation', f'file:source_separation/{role}.wav')
    else:
        files['cover'] = ctx.output('03_style_generation', 'cover')
    return files


def params(r, c):
    return {**c, 'arrangement': r.get('arrangement', 'generated')}


def check_transposition_tools():
    if not shutil.which('rubberband'):
        raise ValueError('Original lead transposition requires the Rubber Band R3 CLI on PATH')
    help_text = subprocess.run(['rubberband', '-h'], capture_output=True, text=True, timeout=15)
    options = help_text.stdout + help_text.stderr
    if any(flag not in options for flag in ('--centre-focus', '-3', '--formant')):
        raise ValueError('Rubber Band CLI must support R3, formant preservation and centre focus')
    filters = subprocess.check_output(['ffmpeg', '-hide_banner', '-filters'], text=True, timeout=15)
    if ' rubberband ' not in filters:
        raise ValueError('Original accompaniment transposition requires the FFmpeg rubberband filter')


def shift(source, target, semitones, lead):
    if lead:
        subprocess.run(['rubberband', '-q', '-3', '-F', '--centre-focus', '-p', str(semitones),
                        source, target], check=True)
    else:
        effect = f'rubberband=tempo=1:pitch={2 ** (semitones / 12)}:formant=preserved'
        subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', source,
                        '-af', effect, '-c:a', 'pcm_f32le', target], check=True)


def run(ctx, out, i, p):
    semitones = json.loads(i['plan'].read_text())['applied_semitones'] if 'plan' in i else 0
    if semitones:
        check_transposition_tools()
    if 'vocals' in i:
        total, instruments = i['vocals'], i['instrumental']
    else:
        decode(i['cover'], out / 'separation_input.wav')
        ctx.execute('separator', 'separate', out, '--input', out / 'separation_input.wav')
        total, instruments = out / 'vocals.wav', out / 'instrumental.wav'
    ctx.execute('separator', 'lead', out, '--input', total)
    lead = out / 'lead.wav'
    voices, rate = sf.read(total, dtype='float32', always_2d=True)
    main, main_rate = sf.read(lead, dtype='float32', always_2d=True)
    if rate != main_rate or voices.shape != main.shape:
        raise ValueError('Lead and total vocals must share a sample grid')
    backing = out / 'backing.wav'
    sf.write(backing, voices - main, rate, subtype='FLOAT')
    del voices, main

    tracks = {}
    for role, source in [('vocals', lead), ('backing', backing), ('instrumental', instruments)]:
        if semitones:
            target = out / f'{role}_transposed.wav'
            shift(source, target, semitones, lead=role == 'vocals')
        elif source.parent != out:
            target = out / source.name
            shutil.copyfile(source, target)
        else:
            target = source
        tracks[role] = target
    bed, bed_rate = sf.read(tracks['instrumental'], dtype='float32', always_2d=True)
    for role, name in [('backing', 'mix_bed'), ('vocals', 'cover')]:
        if name == 'cover' and 'plan' not in i:
            break
        wave, wave_rate = sf.read(tracks[role], dtype='float32', always_2d=True)
        if rate != bed_rate or rate != wave_rate or wave.shape != bed.shape:
            raise ValueError(f'Unaligned {role} after transposition')
        bed += wave
        tracks[name] = out / f'{name}.wav'
        sf.write(tracks[name], bed, rate, subtype='FLOAT')
    write_json(out / 'stem_plan.json', {'applied_audio_semitones': semitones, 'converted_stem': 'lead',
                                        'retained_stems': ['backing', 'instrumental']})
    return tracks
