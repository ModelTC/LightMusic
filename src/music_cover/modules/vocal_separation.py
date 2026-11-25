from ..audio import decode
ID = '04_vocal_separation'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['audio.py', 'backends/separator.py', 'original_stems.py', 'voice_data_separation.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['separator']
def requires(r):
    if r.get('arrangement') == 'original':
        return ['00_prepare', '02_pitch_adaptation']
    return ['03_style_generation']
def inputs(ctx):
    if ctx.request.get('arrangement') == 'original':
        from ..voice_data_separation import LEAD_MODEL
        models = ctx.root / 'models/separation'
        result = {'cover': ctx.output('00_prepare', 'source'), 'plan': ctx.output('02_pitch_adaptation', 'plan'),
                  'lead_model': models / LEAD_MODEL, 'lead_config': models / LEAD_MODEL.replace('.ckpt', '_config.yaml')}
        if str(ctx.request['transpose']) != '0':
            for role in ('vocals', 'instrumental'):
                result[role] = ctx.output('02_pitch_adaptation', f'file:source_separation/{role}.wav')
        return result
    return {'cover': ctx.output('03_style_generation', 'cover')}
def params(r, c): return c


def run(ctx, out, i, p):
    if 'plan' in i:
        from ..original_stems import prepare
        return prepare(ctx, out, i)
    decode(i['cover'], out / 'separation_input.wav')
    ctx.execute('separator', 'separate', out, '--input', out / 'separation_input.wav')
    return {'vocals': out / 'vocals.wav', 'instrumental': out / 'instrumental.wav'}
