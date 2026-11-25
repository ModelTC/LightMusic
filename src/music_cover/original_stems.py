"""Prepare the original accompaniment and transpose only after lead separation."""
import json
import shutil
import subprocess

import numpy as np
import soundfile as sf

from .common.io import write_json


def restore_consonants(voices, lead, rate):
    """Move gated high-frequency residual from backing to lead before transposition."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    from scipy.signal import butter, sosfiltfilt

    backing = voices - lead
    hop = round(0.01 * rate)
    if len(lead) < hop:
        return lead, backing, 0.0

    def band(signal, low, high):
        coefficients = butter(3, [low, high], btype='bandpass', fs=rate, output='sos')
        return sosfiltfilt(coefficients, signal)

    def envelope(signal):
        blocks = signal[:len(signal) // hop * hop].reshape(-1, hop)
        return np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)

    center = voices.mean(axis=1)
    high = envelope(band(center, 3000, 10000))
    body = envelope(band(center, 100, 2000))
    gate = np.clip((high / (body + .001) - .55) / .85, 0, 1)
    gate *= np.clip((high - .002) / .009, 0, 1)
    gate = gaussian_filter1d(maximum_filter1d(gate, size=3), 1.0)
    mask = np.interp(np.arange(len(center)), (np.arange(len(gate)) + .5) * hop, gate)
    transfer = (.5 * mask * band(backing.mean(axis=1), 3000, 10000))[:, None]
    return (lead + transfer).astype('float32'), (backing - transfer).astype('float32'), float(np.sum(mask > .1) / rate)


def prepare(ctx, out, inputs):
    separation = out / 'original'
    if 'vocals' in inputs:
        vocals, instrumental = inputs['vocals'], inputs['instrumental']
    else:
        ctx.execute('separator', 'separate', separation, '--input', inputs['cover'])
        vocals, instrumental = separation / 'vocals.wav', separation / 'instrumental.wav'

    ctx.execute('separator', 'lead', separation, '--input', vocals)
    lead = separation / 'lead.wav'
    voices, rate = sf.read(vocals, dtype='float32', always_2d=True)
    main, lead_rate = sf.read(lead, dtype='float32', always_2d=True)
    if main.shape != voices.shape or rate != lead_rate:
        raise ValueError('Original lead and vocal stems must share a sample grid')
    backing = separation / 'backing.wav'
    main, residual, active_seconds = restore_consonants(voices, main, rate)
    lead = separation / 'lead_restored.wav'
    sf.write(lead, main, rate, subtype='FLOAT')
    sf.write(backing, residual, rate, subtype='FLOAT')
    del voices, main, residual

    semitones = json.loads(inputs['plan'].read_text())['applied_semitones']
    tracks = {}
    for role, source in [('lead', lead), ('backing', backing), ('accompaniment', instrumental)]:
        target = out / f'{role}.wav'
        if semitones and role == 'lead':
            subprocess.run(['rubberband', '-q', '-3', '-F', '--centre-focus',
                            '-p', str(semitones), source, target], check=True)
        elif semitones:
            effect = f'rubberband=tempo=1:pitch={2 ** (semitones / 12)}:formant=preserved'
            subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', source,
                            '-af', effect, '-ar', str(rate), '-c:a', 'pcm_f32le', target], check=True)
        else:
            shutil.copyfile(source, target)
        tracks[role] = target

    bed, bed_rate = sf.read(tracks['accompaniment'], dtype='float32', always_2d=True)
    for role, name in [('backing', 'instrumental'), ('lead', 'cover')]:
        wave, sample_rate = sf.read(tracks[role], dtype='float32', always_2d=True)
        if sample_rate != bed_rate or wave.shape != bed.shape:
            raise ValueError(f'Unaligned transposed stem: {role}')
        bed += wave
        tracks[name] = out / f'{name}.wav'
        sf.write(tracks[name], bed, bed_rate, subtype='FLOAT')
    tracks['vocals'] = tracks.pop('lead')
    write_json(out / 'stem_plan.json', {'applied_semitones': semitones, 'converted_stem': 'lead',
                                        'consonant_gate_seconds_above_0_1': active_seconds,
                                        'retained_stems': ['backing', 'accompaniment']})
    return tracks
