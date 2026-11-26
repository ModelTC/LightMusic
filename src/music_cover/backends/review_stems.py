from __future__ import annotations
import csv
import gc
import html
import json
import math
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf
from ..audio import fit_length, inspect_audio, read_at, sha256
from ..common.paths import ROOT
from ..common.resources import resource_data
from ..common.io import write_json as save_json, artifact_paths

from ..common.review_audio import RATE, LEAF_ROLES, PITCHED_ROLES, LABELS, stereo_aligned, mix_components, write_stem

def separate_review(request, output):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    output = Path(output).resolve()
    stems = output / "audio"
    stems.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(int(request.get("seed", 42)))
    torch.manual_seed(int(request.get("seed", 42)))
    metrics = json.loads(Path(request["mix_metrics"]).read_text())
    master, voices, accompaniment, residual = mix_components(
        request["final"], request["vocal"], request["instrumental"], metrics)
    frames = len(master)
    if 'backing_post_gain' in request:
        backing = stereo_aligned(request['backing_post_gain'], frames)
        write_stem(stems, 'lead_vocals', voices)
        write_stem(stems, 'backing_vocals', backing)
        write_stem(stems, 'vocals', voices + backing)
        write_stem(stems, 'instrumental', accompaniment)
        write_stem(stems, 'mix_residual', residual - backing)
        del backing, residual
    else:
        for role, data in [("vocals", voices), ("instrumental", accompaniment), ("mix_residual", residual)]:
            write_stem(stems, role, data)
        del residual

        # Legacy requests carry converted vocals without a retained backing stem.
        from tempfile import TemporaryDirectory
        from ..audio import decode
        from ..common.resources import python_for
        with TemporaryDirectory(dir=output) as temporary:
            raw = Path(temporary)
            source = raw / "total.wav"
            decode(stems / "vocals.wav", source, rate=44100)
            subprocess.run([python_for(ROOT, "lead_duo"), "-m", "music_cover.backends.cli", "lead",
                            "--input", source, "--output", raw], check=True)
            lead = stereo_aligned(raw / "lead.wav", frames)
            write_stem(stems, "lead_vocals", lead)
            write_stem(stems, "backing_vocals", voices - lead)

    print("[review] Separating final accompaniment using Demucs 6s", flush=True)
    model = get_model("htdemucs_6s", repo=ROOT / "models/review/demucs").eval().to("cuda")
    wave = torch.from_numpy(read_at(stems / "instrumental.wav", model.samplerate).T.copy())
    ref = wave.mean(0)
    mean, std = ref.mean(), ref.std().clamp_min(1e-8)
    with torch.inference_mode():
        estimates = apply_model(model, ((wave - mean) / std)[None], device="cuda", shifts=1,
                                split=True, overlap=0.25, progress=True, num_workers=0)[0].cpu()
    estimates = estimates * std + mean
    raw = output / "_demucs"
    raw.mkdir()
    reconstructed = np.zeros_like(accompaniment)
    for index, source in enumerate(model.sources):
        role = "instrumental_vocal_leak" if source == "vocals" else source
        temporary = raw / f"{role}.wav"
        sf.write(temporary, estimates[index].T.numpy(), model.samplerate, subtype="FLOAT")
        data = stereo_aligned(temporary, frames)
        write_stem(stems, role, data)
        reconstructed += data
    write_stem(stems, "instrumental_residual", accompaniment - reconstructed)
    shutil.rmtree(raw)
    del model, estimates, wave, reconstructed
    gc.collect()
    torch.cuda.empty_cache()

    total = np.zeros_like(master, dtype=np.float64)
    entries = {}
    for role in ("vocals", "instrumental", *LEAF_ROLES):
        path = stems / f"{role}.wav"
        info = inspect_audio(path)
        if info["frames"] != frames or info["sample_rate"] != RATE or info["channels"] != 2:
            raise RuntimeError(f"Unaligned review stem: {role}")
        info.update(path=f"audio/{role}.wav", label=LABELS[role],
                    sum_with_leaf_tracks=role in LEAF_ROLES,
                    status="near_silent" if info["rms"] < 1e-5 else "audio_present_not_quality_verified")
        entries[role] = info
        if role in LEAF_ROLES:
            total += sf.read(path, dtype="float64", always_2d=True)[0]
    error = total - master
    maximum = float(np.max(np.abs(error)))
    if maximum > 2e-6:
        raise RuntimeError(f"Leaf stems do not reconstruct final.wav: max error {maximum}")
    sf.write(output / "stems_recombined.wav", total, RATE, subtype="FLOAT")
    models = resource_data(ROOT, "review_stems")
    vocal_source = "04 lead separation via 06 post-gain stems"
    if "backing_post_gain" not in request:
        models = {"assets": models["assets"] + resource_data(ROOT, "lead_duo")["assets"]}
        vocal_source = "07 lead separation of converted vocals"

    save_json(output / "stems.json", {
        "schema_version": 1, "input_final_sha256": sha256(request["final"]),
        "sample_rate": RATE, "frames": frames, "duration_s": frames / RATE,
        "time_origin": "sample 0 of final.wav; leading silence preserved", "stems": entries,
        "leaf_sum_roles": list(LEAF_ROLES),
        "reconstruction": {"max_abs_error": maximum, "rms_error": float(np.sqrt(np.mean(error ** 2))),
                           "includes_residuals": True, "proves_separation_quality": False},
        "mix_gains": {"master_gain": metrics["final"]["master_gain"], "vocal_gain": metrics["vocal_gain"]},
        "models": models, "vocal_source": vocal_source,
        "limitations": ["Model estimates, not original recording stems; leakage is possible.",
                        "Backing vocal estimate can contain lead reverb or artifacts, not proof of harmony singers.",
                        "Other can contain synths/strings; a named stem does not prove that instrument is present.",
                        "Parent vocals/instrumental must not be summed together with their child tracks.",
                        "Residuals preserve reconstruction; they are not musical instruments."]})
