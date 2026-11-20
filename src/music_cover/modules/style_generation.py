from ..common.io import write_json
ID = '03_style_generation'
DEFAULTS = {'timeout_seconds': 10800, 'memory_budget_gib': 64}
CODE = ['backends/yue.py', 'pitch.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['yue']
def requires(r): return ['00_prepare', '02_pitch_adaptation']
def inputs(ctx): return {'score': ctx.output('02_pitch_adaptation', 'score'), 'lyrics': ctx.output('00_prepare', 'lyrics')}
def params(r, c):
    memory = float(c.get('memory_budget_gib', 64))
    if not 0 < memory <= 1024:
        raise ValueError('memory_budget_gib must be positive and finite, at most 1024')
    return {**c, 'memory_budget_gib': memory, 'style': r.get('style', ''), 'seed': r['seed']}


def run(ctx, out, i, p):
    write_json(out / 'generation_request.json', {'style': p['style'], 'lyrics': i['lyrics'].read_text(), 'seed': p['seed'], 'cot': 'melody'})
    ctx.execute('yue', 'cover', out, '--request', out / 'generation_request.json', '--score', i['score'],
                '--memory-budget-gib', p['memory_budget_gib'])
    return {'cover': out / 'generated/audio.flac'}
