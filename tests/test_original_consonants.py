import unittest

import numpy as np

from music_cover.original_stems import restore_consonants


class ConsonantTransferTests(unittest.TestCase):
    def test_high_frequency_residual_is_moved_between_stems(self):
        rate = 44100
        t = np.arange(rate) / rate
        vowel = .015 * np.sin(2 * np.pi * 220 * t)
        hiss = .15 * np.sin(2 * np.pi * 6000 * t) * ((t > .3) & (t < .5))
        lead = np.column_stack([vowel, vowel]).astype('float32')
        voices = (lead + hiss[:, None]).astype('float32')
        restored, backing, seconds = restore_consonants(voices, lead, rate)
        self.assertGreater(seconds, .1)
        self.assertLess(seconds, .3)
        self.assertGreater(np.max(np.abs(restored - lead)), .04)
        np.testing.assert_allclose(restored + backing, voices, atol=3e-8, rtol=0)
        np.testing.assert_allclose(restored[:rate // 5], lead[:rate // 5], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
