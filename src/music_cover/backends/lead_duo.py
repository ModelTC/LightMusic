"""Estimate the lead on the native stereo grid using the two audited BS-RoFormers."""
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

from pathlib import Path
from ..common.paths import ROOT
from ..audio import sha256
from ..common.resources import resource_data

MODELS = (
    ('karaoke_bs_roformer_anvuew.ckpt', 'karaoke_bs_roformer_anvuew.yaml'),
    ('bs_roformer_karaoke_frazer_becruily.ckpt', 'config_karaoke_frazer_becruily.yaml'),
)


@torch.inference_mode()
def estimate(network, audio, chunk, overlap):
    """Hann overlap-add; MSST may round the synthesized chunk down to an STFT hop."""
    step = chunk // overlap
    window = np.maximum(np.hanning(chunk), 0.0001).astype('float32')
    total = np.zeros_like(audio)
    weights = np.zeros(len(audio), dtype='float32')
    for start in range(0, len(audio), step):
        signal = audio[start:start + chunk]
        block = np.pad(signal.T, ((0, 0), (0, chunk - len(signal))))
        output = network(torch.from_numpy(block).cuda().unsqueeze(0))[0, 0].cpu().numpy()
        size = min(output.shape[-1], len(signal))
        weight = window[:size]
        total[start:start + size] += output[:, :size].T * weight[:, None]
        weights[start:start + size] += weight
    if np.any(weights <= 0):
        raise RuntimeError('Lead separation left frames uncovered')
    return total / weights[:, None]


def lead(args):
    directory = ROOT / 'models/LeadDuo'
    # The runner launches each backend operation in its own interpreter.
    # Verify the downloaded Python sources even when --preflight was omitted.
    for item in resource_data(ROOT, 'lead_duo')['assets']:
        path = ROOT / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError(f'Lead separation asset checksum mismatch: {path}')
    sys.path.insert(0, str(directory / 'msst'))
    from models.bs_roformer.bs_roformer import BSRoformer

    audio, rate = sf.read(args.input, dtype='float32', always_2d=True)
    if rate != 44100 or audio.shape[1] != 2 or not len(audio):
        raise ValueError('Lead separation requires nonempty 44.1 kHz stereo audio')
    result = np.zeros_like(audio)
    for checkpoint, config in MODELS:
        settings = yaml.load((directory / config).read_text(), Loader=yaml.FullLoader)
        network = BSRoformer(**{**settings['model'], 'use_torch_checkpoint': False})
        network.load_state_dict(torch.load(directory / checkpoint, weights_only=True, map_location='cpu'))
        network.eval().cuda()
        result += estimate(network, audio, settings['audio']['chunk_size'], settings['inference']['num_overlap'])
        del network
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite lead estimate')
    sf.write(Path(args.output) / 'lead.wav', result / len(MODELS), rate, subtype='FLOAT')
