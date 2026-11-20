# 音乐曲风与音色转换 baseline

从原曲、歌词和参考人声生成新编曲，经音色转换与混音输出 WAV/MP3。支持 Seed-VC V1 零样本、全量微调、LoRA，以及 so-vits-svc、RVC、Fish Diffusion。

当前版本为 **v2.0**，包含 **Seed-VC V1 LoRA 训练、验证、导出和 05 推理**，以及项目内环境部署、CLI 参数检查与 GitHub CI。这里的 v2.0 是项目版本，Seed-VC 使用 V1。验证范围见 [VALIDATION](VALIDATION.md)，版本变化见 [CHANGELOG](CHANGELOG.md)。

```bash
git clone --branch v2.0 https://github.com/ModelTC/music-cover-baseline.git
cd music-cover-baseline
git submodule update --init third_party/YuE third_party/seed-vc third_party/so-vits-svc third_party/rvc
python3 scripts/setup/apply_seed_vc_lora.py
```

接着按下方文档部署环境和模型；Git 仓库不包含权重、录音或生成音频。

## 从哪里开始

| 任务 | 文档 |
|---|---|
| 部署与主流程 | [部署 Skill](skill/SKILL.md)、[使用说明](USAGE.md)、[环境布局](docs/03-environment.md) |
| 手动训练 LoRA，再运行 05→06 | [LoRA 13 步手册](docs/25_seedvc_lora_manual.md)、[入口与参数](scripts/05_voice_conversion/lora/README.md) |
| 全量微调 / SVC / RVC / Fish | [Seed-VC](docs/13_voice_adaptation.md)、[SVC](docs/18_so_vits_svc.md)、[RVC](docs/20_rvc.md)、[Fish](docs/24_fish_diffusion_manual.md) |
| 模型资源 | [权重清单](skill/references/weights.md) |
| 贡献代码与 GitHub 检查 | [CONTRIBUTING](CONTRIBUTING.md)、[仓库与数据边界](docs/08-repository.md) |

## 流程

```text
00 准备 → 01 转谱 → 02 分析音区/移调 → 03 YuE2 编曲 → 04 分离
→ 05 换声 → 06 混音 → 07 分轨 → 08 转谱 → 09 MIDI → 10 评估 → 11 交付

独立音色训练：分离录音 → 训练 00_prepare → 01_train → 02_validate
                                              → 03_export → 供主流程 05 使用
```

05 共四种后端、六条路线。LoRA 是 Seed-VC V1 的适配器路线，与 V2 无关；05 不会隐式启动训练。只需歌曲成品时运行到 06；`--no-review` 只跳过 07–09，仍会执行 10–11。

## 已部署环境的最小示例

从仓库根目录开始，准备自己的歌曲、歌词、参考人声，修改示例路径和曲风：

```bash
export VC_ROOT="$PWD"
source "$VC_ROOT/scripts/environment.sh"
mkdir -p configs/local
cp -n configs/examples/cover.yaml configs/local/my_cover.yaml
# 先编辑 my_cover.yaml，确认三个输入文件实际存在。
bash scripts/run_baseline.sh --config configs/local/my_cover.yaml --until 06_mix --preflight
bash scripts/run_baseline.sh --config configs/local/my_cover.yaml --run runs/my_cover_v001 --until 06_mix --gpu 0
```

首次成功混音在 `runs/my_cover_v001/stages/06_mix/attempt_001/artifacts/final.mp3`。有效 attempt 读取 `stage_state.json`，不要按最大编号猜测；只有执行 11 才生成顶层 `deliverables/`。

环境、缓存、模型和数据放在项目内，不要求软链接。新机器仍需部署依赖、固定子模块和权重；源码检查通过不等于真实 GPU 推理通过。ECAPA 等指标只用于诊断，不能代替试听或作为身份准确率。

## English overview

**Music Cover Baseline v2.0** is a modular pipeline for score-guided music restyling and target-voice conversion. It combines transcription, pitch adaptation, YuE2 arrangement generation, vocal separation, voice conversion, and aligned mixing, with optional stems, MIDI, evaluation, and delivery packaging.

Seed-VC V1 supports zero-shot conversion, full fine-tuning, and Attention+FFN LoRA adaptation. LoRA training is independent of song inference and includes dataset provenance checks, validation-based checkpoint selection, held-out test audits, immutable voice exports, and stage-05 loading. Additional trained-voice backends include so-vits-svc, RVC, and Fish Diffusion.

The repository provides project-local runtime configuration, pinned upstream integrations, reproducible LoRA patches, a deployment skill, step-by-step CLI documentation, and CPU CI checks. Model weights, recordings, training datasets, and generated media are provisioned separately. Automated checks do not replace GPU acceptance tests or human listening. See [CONTRIBUTING](CONTRIBUTING.md) and [VALIDATION](VALIDATION.md) for verification scope.

**Project v2.0 does not switch to Seed-VC V2; this release continues to use Seed-VC V1.**
