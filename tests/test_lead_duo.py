"""Cover MSST's batch/stem/channel layout and truncated STFT output."""
import unittest
from unittest.mock import patch

import numpy as np


class OverlapAddTests(unittest.TestCase):
    def test_single_stem_axis_and_short_final_window(self):
        import torch
        from music_cover.backends.lead_duo import estimate

        audio = np.linspace(-1, 1, 110, dtype='float32').reshape(-1, 2)

        def network(block):
            # (batch, one model stem, stereo, frames)
            return block.unsqueeze(1)[..., :-1]

        with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            result = estimate(network, audio, chunk=16, overlap=4)
        np.testing.assert_allclose(result, audio, atol=2e-7)

    def test_two_checkpoint_estimates_average_and_reconstruct_total(self):
        import sys
        import tempfile
        from pathlib import Path
        from types import ModuleType

        import soundfile as sf
        import torch
        import yaml
        from music_cover.backends import lead_duo

        class GainModel(torch.nn.Module):
            def __init__(self, use_torch_checkpoint):
                super().__init__()
                self.register_buffer('gain', torch.tensor(0.))

            def forward(self, block):
                return (self.gain * block).unsqueeze(1)

        model_module = ModuleType('models.bs_roformer.bs_roformer')
        model_module.BSRoformer = GainModel
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'models' / 'LeadDuo'
            directory.mkdir(parents=True)
            settings = {'model': {}, 'audio': {'chunk_size': 16}, 'inference': {'num_overlap': 4}}
            for (checkpoint, config), gain in zip(lead_duo.MODELS, [.25, .75], strict=True):
                torch.save({'gain': torch.tensor(gain)}, directory / checkpoint)
                (directory / config).write_text(yaml.safe_dump(settings))
            total = np.linspace(-.5, .5, 110, dtype='float32').reshape(-1, 2)
            source = root / 'total.wav'
            sf.write(source, total, 44100, subtype='FLOAT')
            with (patch.object(lead_duo, 'ROOT', root),
                  patch.object(lead_duo, 'resource_data', return_value={'assets': []}),
                  patch.dict(sys.modules, {model_module.__name__: model_module}),
                  patch.object(sys, 'path', sys.path.copy()),
                  patch.object(torch.Tensor, 'cuda', lambda tensor: tensor),
                  patch.object(GainModel, 'cuda', lambda model: model)):
                lead_duo.separate_lead(source, root)
            lead, rate = sf.read(root / 'lead.wav')
            backing, backing_rate = sf.read(root / 'backing.wav')
            self.assertEqual((rate, backing_rate), (44100, 44100))
            np.testing.assert_allclose(lead, total * .5, atol=2e-7)
            np.testing.assert_allclose(lead + backing, total, atol=2e-7)
