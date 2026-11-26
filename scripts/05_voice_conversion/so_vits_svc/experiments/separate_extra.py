import json,os,logging,time
from pathlib import Path
import torch
from audio_separator.separator import Separator
from music_cover.common.paths import ROOT
from music_cover.audio import sha256
torch.set_num_threads(4)
out=ROOT/'data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028/extra_kongcheng';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'data/raw/songs/kong_cheng_yang_kun.mp3';original=source
from music_cover.backends.lead_duo import separate_lead
if not (out/'vocals.wav').exists():
    separator = Separator(model_file_dir=str(ROOT/'models/separation'), output_dir=str(out),
        output_format='WAV', normalization_threshold=1., amplification_threshold=0.,
        sample_rate=44100, use_soundfile=True, log_level=logging.WARNING,
        mdxc_params={'segment_size':256,'override_model_segment_size':False,'batch_size':1,'overlap':4,'pitch_shift':0})
    separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    os.chdir(out)
    separator.separate(str(source), {'Vocals':'vocals','Instrumental':'instrumental'})
    del separator
    torch.cuda.empty_cache()
separate_lead(out/'vocals.wav', out)
(out/'source.json').write_text(json.dumps({'path':str(original),'sha256':sha256(original),'metadata':str(original.with_suffix('.source.json')),'song_group':'yang_kun:空城','split':'train','note':'Additional full recording from existing source library; not validation/test'},ensure_ascii=False,indent=2))
