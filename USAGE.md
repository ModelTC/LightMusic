# 使用说明

命令从仓库根目录执行，适用于当前含 LoRA 的工作树。环境准备见 [环境说明](docs/03-environment.md)，权重见 [部署 Skill](skill/SKILL.md)。历史实验配置依赖作者的私有素材，不是开箱即用的输入。

## 1. 准备输入和环境

```bash
export VC_ROOT="$PWD"
source "$VC_ROOT/scripts/environment.sh"
mkdir -p configs/local
cp -n configs/examples/cover.yaml configs/local/my_cover.yaml
```

编辑 `configs/local/my_cover.yaml`，填入真实源歌、UTF-8 歌词、目标人物参考人声和曲风。参考片段 1–25 秒，时间窗必须落在音频内；歌唱音域分析使用 `reference_kind: singing`。素材分离见 [分离入口](scripts/voice_data/separate/README.md)。

`configs/local/` 不提交 Git；公开模板放 `configs/examples/`。相对路径按项目根目录解析。Shell 入口加载项目内环境和缓存，无须激活系统 Conda。

## 2. 主流程 00–06

```bash
export VC_TASK="$VC_ROOT/configs/local/my_cover.yaml"
export VC_SONG_RUN="$VC_ROOT/runs/my_cover_v001"
bash scripts/run_baseline.sh --config "$VC_TASK" --until 06_mix --preflight
bash scripts/run_baseline.sh --config "$VC_TASK" --run "$VC_SONG_RUN" --until 06_mix --gpu 0
```

分步操作时 00 保存任务配置，后续使用同一 run：

```bash
bash scripts/00_prepare/run.sh --config "$VC_TASK" --run "$VC_SONG_RUN" --gpu 0
bash scripts/01_source_score/run.sh --run "$VC_SONG_RUN" --gpu 0
bash scripts/02_pitch_adaptation/run.sh --run "$VC_SONG_RUN" --gpu 0
bash scripts/03_style_generation/run.sh --run "$VC_SONG_RUN" --gpu 0
bash scripts/04_vocal_separation/run.sh --run "$VC_SONG_RUN" --gpu 0
# 选择下一节的一条 05 命令，再执行 06。
bash scripts/05_voice_conversion/zero_shot/run.sh --run "$VC_SONG_RUN" --gpu 0
bash scripts/06_mix/run.sh --run "$VC_SONG_RUN" --gpu 0
```

仅准备训练评估用的新编曲，统一入口用 `--until 04_vocal_separation`。00–04 不训练音色，也不保证 03 的声线已经是目标人物。

## 3. 05 的六条路线

| 路线 | 入口（相对 scripts/05_voice_conversion） | 额外条件 |
|---|---|---|
| Seed-VC 零样本 | `zero_shot/run.sh` | 公共模型 + 任务参考 |
| Seed-VC 全量微调 | `finetuned/run.sh` | `seed_vc/<版本>/voice.json` |
| Seed-VC V1 LoRA | `lora/run.sh` | `seed_vc_lora/<版本>/voice.json` + 匹配底模/原生补丁 |
| so-vits-svc | `so_vits_svc/run.sh` | 人物导出模型 |
| RVC | `rvc/run.sh` | 人物模型 + 索引 |
| Fish Diffusion | `fish_diffusion/run.sh` | 人物扩散模型 |

LoRA 训练与导出按 [13 步手册](docs/25_seedvc_lora_manual.md) 操作；已有导出包直接推理：

```bash
export VC_PROFILE="$VC_ROOT/models/voices/my_voice/seed_vc_lora/r16_v001/voice.json"
bash scripts/05_voice_conversion/lora/run.sh \
  --run "$VC_SONG_RUN" --voice-profile "$VC_PROFILE" \
  --voice-candidates 1 --voice-candidate-seeds 42 --gpu 0 --preflight
bash scripts/05_voice_conversion/lora/run.sh \
  --run "$VC_SONG_RUN" --voice-profile "$VC_PROFILE" \
  --voice-candidates 1 --voice-candidate-seeds 42 --gpu 0
bash scripts/06_mix/run.sh --run "$VC_SONG_RUN" --gpu 0
```

LoRA 默认要求任务参考和导出包参考解码后的 PCM 一致。若明确借用另一人物的编曲，使用 `--voice-reference-policy profile` 采用导出包绑定的参考，并保留原任务元信息。全量微调用 `finetuned/run.sh` 和对应 profile；SVC/RVC/Fish 参数见各自 README。

多路线比较先 fork：

```bash
bash scripts/05_voice_conversion/fork_run.sh \
  --source-run "$VC_SONG_RUN" --run runs/my_cover_lora_v001
```

fork 引用经校验的 00–04 文件，必须保留源 run。真正独立的新人物任务从新的 00 执行到 04。

## 4. 训练、validation 与 test

`01_train` 更新 LoRA；`02_validate --evaluation-split validation` 选择 checkpoint；`02_validate --evaluation-split test` 仅审核同一训练身份中已被 validation 选中的一个 checkpoint。

test 可省略 `--checkpoint`，自动读取最近一次成功且哈希完整的 validation 选择。不要传空的 `"$VC_CKPT"`；未验证时先运行 validation。继续训练产生新权重后，应重新 validation，不能用旧选择证明新权重已经验证。详细命令见 [手册第 8–10 步](docs/25_seedvc_lora_manual.md)。

## 5. 产物、重跑与故障定位

```bash
bash scripts/run_baseline.sh --run "$VC_SONG_RUN" --status
```

- 主流程状态：`stage_state.json`；训练状态：`state.json`。
- attempt 内有 `command.json`、`process.log`、`outputs.json`，失败时有 `failure.json`。
- 04 输出 `vocals.wav`、`instrumental.wav`；05 输出 `converted.wav` 和候选报告；06 输出 `final.wav`、`final.mp3`、`mix.json`。
- 主流程 `--force` 创建新 attempt，保留旧文件；不要改写已封存产物。
- 复制试听文件时核对源 SHA256、profile、模型模式与目标音色，防止同名文件串版。
- 评分不合格和执行失败不同，候选推荐不代表人工试听通过。

维护检查见 [CONTRIBUTING](CONTRIBUTING.md)。

## 主唱换声与原编曲

先运行 `python3 scripts/setup/download_lead_duo.py` 安装双模型权重、配置和匹配的推理源码。默认生成模式保持 03 生成的调性。04 将总人声进一步拆成主唱与伴唱，05 只读取主唱，06 混回保留的伴唱和乐器。

配置 `request.arrangement: original` 或传 `--arrangement original` 可以沿用原曲编排。02 选调后，04 在原调完成分离，再对主唱、伴唱与乐器使用相同半音数。非零移调需要 PATH 中有支持 R3 的 Rubber Band CLI（验证版本 3.3.0）和带 rubberband 滤镜的 FFmpeg。主唱使用 `rubberband -3 -F --centre-focus`，零移调直接保留音频。该模式跳过 03，其余分析及交付遵循 review 设置。
