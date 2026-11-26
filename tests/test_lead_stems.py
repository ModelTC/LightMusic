import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from music_cover.audio import sha256
from music_cover.common.io import write_json
from music_cover.common.paths import RunPaths
from music_cover.modules import active_ids, vocal_separation
from music_cover.pipeline import preflight_original_transposition


class LeadStems(unittest.TestCase):
    def test_original_skips_only_generation(self):
        generated = active_ids({})
        self.assertEqual(active_ids({'arrangement': 'original'}),
                         [stage for stage in generated if stage != '03_style_generation'])

    def test_auto_preflight_checks_verified_plan_and_rejects_corruption(self):
        stage = '02_pitch_adaptation'
        params = {'transpose': 'auto'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, marker = root / 'plan.json', root / 'outputs.json'
            prepare = root / 'prepare'
            prepare.mkdir()
            recording = root / 'song.wav'
            recording.write_bytes(b'recorded song')
            reference = root / 'reference.wav'
            reference.write_bytes(b'reference voice')
            request = dict(params, song=recording, reference=reference, reference_start=0, reference_duration=5)
            inputs = {key: {'path': str(path), 'sha256': sha256(path)}
                      for key, path in [('song', recording), ('reference', reference)]}
            tracks = {'source': inputs['song'], 'reference': inputs['reference']}
            write_json(prepare / 'inputs.json', {'artifacts': inputs})
            write_json(root / 'inputs.json', {'artifacts': tracks})
            prepared = {'params': {'start': 0, 'duration': 5}, 'attempt': str(prepare),
                        'outputs': tracks, 'status': 'succeeded'}
            for semitones in (0, 1, -2):
                write_json(plan, {'applied_semitones': semitones})
                role = {'path': plan.name, 'bytes': plan.stat().st_size, 'sha256': sha256(plan)}
                common = {'stage': stage, 'status': 'succeeded', 'fingerprint': 'example'}
                write_json(marker, {**common, 'artifacts': {'plan': role}})
                record = {**common, 'params': params, 'attempt': str(root),
                          'outputs_manifest_sha256': sha256(marker),
                          'outputs': {'plan': {**role, 'path': str(plan)}}}
                paths = RunPaths(root, root, {'stages': {stage: record, '00_prepare': prepared}})
                with patch.object(vocal_separation, 'check_transposition_tools') as tools:
                    preflight_original_transposition(request, paths, ['04_vocal_separation'], params, False)
                    self.assertEqual(tools.call_count, int(semitones != 0))
            for overrides in ({'song': reference}, {'reference': recording}, {'reference_start': 1}):
                with patch('builtins.print') as report, patch.object(vocal_separation, 'check_transposition_tools') as tools:
                    preflight_original_transposition({**request, **overrides}, paths, ['04_vocal_separation'], params, False)
                    self.assertIn('not yet verified', report.call_args.args[0])
                    tools.assert_not_called()
            write_json(plan, {'applied_semitones': 0})
            with self.assertRaisesRegex(ValueError, 'artifact missing or changed'):
                preflight_original_transposition(request, paths, ['04_vocal_separation'], params, False)

    def test_generated_and_original_preserve_backing_and_route_pitch_changes(self):
        for original, semitones in ((False, 0), (True, 0), (True, 1)):
            with self.subTest(original=original, semitones=semitones), tempfile.TemporaryDirectory() as folder:
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
                        write(output / 'backing.wav', total - lead)
                    else:
                        self.fail(operation)
                ctx = Mock()
                ctx.execute.side_effect = execute
                inputs = {'cover': root / 'input.wav'}
                write(inputs['cover'], total + inst)
                if original:
                    inputs['plan'] = root / 'plan.json'
                    inputs['plan'].write_text(json.dumps({'applied_semitones': semitones}))
                commands = []
                def run_shift(command, check):
                    commands.append(command)
                    source = command[-2] if command[0] == 'rubberband' else command[command.index('-i') + 1]
                    shutil.copyfile(source, command[-1])
                with patch.object(vocal_separation, 'decode', side_effect=shutil.copyfile), \
                        patch.object(vocal_separation, 'check_transposition_tools') as check_tools, \
                        patch.object(vocal_separation.subprocess, 'run', side_effect=run_shift):
                    tracks = vocal_separation.run(ctx, out, inputs, {})
                if semitones:
                    check_tools.assert_called_once()
                    self.assertEqual([cmd[0] for cmd in commands], ['rubberband', 'ffmpeg', 'ffmpeg'])
                    self.assertEqual(commands[0][2:7], ['-3', '-F', '--centre-focus', '-p', '1'])
                    self.assertEqual(commands[0][-2], out / 'lead.wav')
                    for command in commands[1:]:
                        self.assertIn(f'pitch={2 ** (1 / 12)}:', command[command.index('-af') + 1])
                else:
                    check_tools.assert_not_called()
                    self.assertEqual(commands, [])
                for role, expected in [('vocals', lead), ('backing', total-lead),
                                       ('instrumental', inst), ('mix_bed', inst+total-lead)]:
                    np.testing.assert_allclose(sf.read(tracks[role])[0], expected)
                self.assertEqual('cover' in tracks, original)
