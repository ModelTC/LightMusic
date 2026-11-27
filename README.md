# LightMusic

**English** | [简体中文](README_CN.md)

LightMusic is a modular pipeline for music restyling and voice conversion. Starting with a song, lyrics, and a reference vocal, it generates a new arrangement, converts the lead voice, and produces aligned WAV/MP3 mixes. It supports Seed-VC V1 zero-shot conversion, full fine-tuning, and LoRA, plus trained so-vits-svc, RVC, and Fish Diffusion voices.

The default development branch is **main**. The current project version is **v2.0**, which includes Seed-VC V1 LoRA training, validation, export, and inference, project-local environments, CLI validation, and GitHub CI. The project version is independent of the Seed-VC model version: **LightMusic continues to use Seed-VC V1**. See [validation results](VALIDATION.md) and the [changelog](CHANGELOG.md).

## Get the code

```bash
git clone --branch main https://github.com/ModelTC/LightMusic.git
cd LightMusic
git submodule update --init third_party/YuE third_party/seed-vc third_party/so-vits-svc third_party/rvc
python3 scripts/setup/apply_seed_vc_lora.py
```

Next, provision the environments and models using the guides below. Model weights, recordings, training datasets, and generated audio are not included in Git.

## Documentation

Most detailed guides are currently in Chinese.

| Task | Guide |
|---|---|
| Deploy and run the pipeline | [Deployment skill](skill/SKILL.md), [usage](USAGE.md), [environment layout](docs/03-environment.md) |
| Train LoRA and run voice conversion and mixing | [13-step LoRA guide](docs/25_seedvc_lora_manual.md), [CLI reference](scripts/05_voice_conversion/lora/README.md) |
| Use other trained-voice routes | [Seed-VC full fine-tuning](docs/13_voice_adaptation.md), [SVC](docs/18_so_vits_svc.md), [RVC](docs/20_rvc.md), [Fish Diffusion](docs/24_fish_diffusion_manual.md) |
| Download model assets | [Pinned model resources](skill/references/weights.md) |
| Contribute and verify changes | [Contributing](CONTRIBUTING.md), [repository and data boundaries](docs/08-repository.md) |

## Pipeline

```text
00 Prepare → 01 Transcribe → 02 Analyze range / transpose → 03 Generate with YuE2
→ 04 Separate vocals → 05 Convert lead voice → 06 Mix → 07 Separate stems
→ 08 Transcribe final audio → 09 Export stem MIDI → 10 Evaluate → 11 Package

Independent voice training:
Separate recordings → 00 Prepare → 01 Train → 02 Validate → 03 Export
                                                            ↓
                                              Main pipeline stage 05
```

Stage 04 runs the anvuew and Frazer Karaoke BS-RoFormer models on the vocal track and averages their lead-vocal estimates. Stage 05 converts the lead voice; stage 06 mixes it with the preserved backing vocals and instruments. Use `--arrangement original` to keep the original arrangement. Pitch shifting of the separated lead vocal uses Rubber Band R3 with formant preservation. See the [arrangement and lead-voice guide](USAGE.md#主唱换声与原编曲).

Stage 05 provides four backends and six routes. LoRA training runs independently and never starts implicitly during song inference. Stop at stage 06 when you only need a finished song. `--no-review` skips stages 07–09 while still running evaluation and delivery at stages 10–11.

## Quick start in a deployed environment

Run from the repository root. Prepare your song, lyrics, and reference vocal, then edit the example configuration to use your files and desired style:

```bash
export VC_ROOT="$PWD"
source "$VC_ROOT/scripts/environment.sh"
mkdir -p configs/local
cp -n configs/examples/cover.yaml configs/local/my_cover.yaml
# Edit my_cover.yaml and verify that all three input files exist.
bash scripts/run_baseline.sh --config configs/local/my_cover.yaml --until 06_mix --preflight
bash scripts/run_baseline.sh --config configs/local/my_cover.yaml --run runs/my_cover_v001 --until 06_mix --gpu 0
```

The first successful mix is written to `runs/my_cover_v001/stages/06_mix/attempt_001/artifacts/final.mp3`. Use `stage_state.json` to locate the active successful attempt after a rerun. The top-level `deliverables/` directory is created by stage 11.

Environments, caches, models, and data live inside the project. A new machine still needs compatible dependencies, pinned upstream code, and model assets. CPU checks do not establish GPU inference readiness. Automated metrics, including ECAPA similarity, support diagnosis and do not replace listening or measure identity accuracy.
