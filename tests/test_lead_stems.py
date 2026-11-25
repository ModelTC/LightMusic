import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from music_cover.modules import active_ids, vocal_separation


class LeadStems(unittest.TestCase):
    def test_original_skips_only_generation(self):
        generated = active_ids({})
        self.assertEqual(active_ids({'arrangement': 'original'}),
                         [stage for stage in generated if stage != '03_style_generation'])

    def test_generated_and_original_zero_keep_backing_and_instruments_distinct(self):
        for original in (False, True):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                out = root / '04'
                out.mkdir()
                rate = 44100
                total = np.full((1024, 2), .25, np.float32)
                lead = total * .75
                inst = total * .5
                def write(path, audio):
                    sf.write(path, audio, rate, subtype='FLOAT')
                def execute(env, operation, output, *args):
                    if operation == 'separate':
                        write(output / 'vocals.wav', total)
                        write(output / 'instrumental.wav', inst)
                    elif operation == 'lead':
                        write(output / 'lead.wav', lead)
                    else:
                        self.fail(operation)
                ctx = Mock()
                ctx.execute.side_effect = execute
                inputs = {'cover': root / 'input.wav'}
                write(inputs['cover'], total + inst)
                if original:
                    inputs['plan'] = root / 'plan.json'
                    inputs['plan'].write_text(json.dumps({'applied_semitones': 0}))
                with patch.object(vocal_separation, 'shift') as shift:
                    tracks = vocal_separation.run(ctx, out, inputs, {})
                    shift.assert_not_called()
                for role, expected in [('vocals', lead), ('backing', total-lead),
                                       ('instrumental', inst), ('mix_bed', inst+total-lead)]:
                    np.testing.assert_allclose(sf.read(tracks[role])[0], expected)
                self.assertEqual('cover' in tracks, original)
