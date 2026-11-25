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
