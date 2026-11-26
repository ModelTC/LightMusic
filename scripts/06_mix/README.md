# 06_mix · 对齐人声并混音

对齐转换后的人声，匹配增益并与伴奏合成 WAV 和 MP3。

本阶段独立脚本自动定位项目根目录、设置 PATH/PYTHONPATH，并直接调用现有 base 的 `envs/base/bin/python`。不需要手动导出模块路径或安装项目包；本阶段不加载模型权重。

## 使用示例

以下示例统一使用“单依纯参考音色＋绿洲”，运行目录为 `runs/lhd_lv_zhou_shan_yi_chun_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

两种模式都需要同一次运行中有效的 [04 分离](../04_vocal_separation/README.md) 和 [05 换声](../05_voice_conversion/README.md) 产物。生成编曲还会读取 [03](../03_style_generation/README.md) 的整曲作为对照。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
# 在项目根目录执行
bash scripts/06_mix/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_modular_test
```

可在命令末尾加 `--preflight`，只检查依赖，不执行混音。也可从其它工作目录使用脚本的绝对路径启动；脚本按自身位置定位项目。

执行到这里即可在本模块 artifacts 中试听 final.wav / final.mp3。运行目录顶层 deliverables 的发布由 11 完成。

## 读取哪些输入

以下路径相对于 `runs/lhd_lv_zhou_shan_yi_chun_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `03_style_generation` | `03_style_generation/attempt_001/artifacts/generated/audio.flac` | 生成编曲的长度及音频对照 |
| `04_vocal_separation` | `cover.wav` | 原编曲的长度及目标调音频对照 |
| `04_vocal_separation` | 04 的 `vocals` 角色：生成编曲或原编曲零移调取 `lead.wav`，原编曲非零移调取 `vocals_transposed.wav` | 换声前人声，用于对齐与增益参考 |
| `04_vocal_separation` | `instrumental`、`backing` | 目标调纯乐器和伴唱，分别写出增益后分轨 |
| `04_vocal_separation` | `mix_bed.wav` | 含伴唱与乐器的混音底轨，与换声主唱相加 |
| `05_voice_conversion` | `05_voice_conversion/attempt_001/artifacts/converted.wav` | 转换后人声 |

## 本模块输出

首次执行的输出目录：`runs/lhd_lv_zhou_shan_yi_chun_modular_test/stages/06_mix/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `final.wav` | 48 kHz 立体声 24-bit 成品；供 07、08、10、11 使用 |
| `final.mp3` | 默认 192 kbps 的成品 MP3 |
| `converted_aligned.wav` | 对齐后人声；供 07 和可选的 10 音高检查 |
| `mix.json` | 对齐、增益及音频指标；供 07、10、11 使用 |
| `reconstruction.wav` | 换声前人声与伴奏的重混对照 |
| `vocal_post_gain.wav` | 增益后的换声主唱 |
| `instrumental_post_gain.wav` | 增益后的纯乐器伴奏 |
| `backing_post_gain.wav` | 增益后的原音色伴唱 |
| `mix_residual.wav` | 最终混音与上述三项之和的数值残差 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/lhd_lv_zhou_shan_yi_chun_modular_test --from 06_mix
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/mix.py`。默认模块配置：`configs/modules/06_mix.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。

## 微调三个候选

当 05 采用 finetuned 三候选时，本阶段也保留三个版本；根目录 final 为默认输出；有合格者时采用推荐，无合格者时标记待复核。具体例子：

```bash
# 在项目根目录执行
bash scripts/06_mix/run.sh --run runs/lhd_candidates_example
```

需要同一运行的前置模块已完成。读取的候选来自 05 的 candidates.json 及三条 converted.wav。本模块输出在 `runs/lhd_candidates_example/stages/06_mix/attempt_001/artifacts/`，查看 `candidates.html`、`candidates.json` 和 `candidates/candidate_001/final.mp3`（002、003 同理）。实际 attempt 由 stage_state.json 决定。分轨与 MIDI 对应默认输出版本；无合格推荐时仍生成混音，并在报告标记待人工确认。完整顺序见[三候选指南](../../docs/15_voice_candidates.md)。
