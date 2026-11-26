# 04_vocal_separation · 提取主唱、伴唱与纯乐器

先用 BS-RoFormer 拆分总人声和纯乐器，再以 anvuew 与 Frazer 两个 Karaoke 模型的等权平均波形估计主唱。伴唱取总人声减主唱。

分离后端直接调用现有 base `envs/base/bin/python`，与 02 使用相同的分离模型及运行环境。
无需激活项目 separator 环境；依赖与预检见[环境说明](../../docs/03-environment.md)。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

生成编曲依赖 [03_style_generation](../03_style_generation/README.md)。原编曲依赖 [00_prepare](../00_prepare/README.md) 的原曲与 [02_pitch_adaptation](../02_pitch_adaptation/README.md) 的选调计划。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
# 在项目根目录执行
source scripts/environment.sh

bash scripts/04_vocal_separation/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

默认分离 03 生成的歌曲。02 中用于音区分析的原曲人声属于另一条时间轴，仅在原编曲模式下可复用。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `03_style_generation` | `03_style_generation/attempt_001/artifacts/generated/audio.flac` | 生成编曲的分离输入 |
| `00_prepare`、`02_pitch_adaptation` | `source.wav`、`transposition.json` | 原编曲的音频与选定半音数 |
| `02_pitch_adaptation` | `source_separation/vocals.wav`、`source_separation/instrumental.wav` | 原编曲且请求 auto 或显式非零移调时，04 直接读取 02 的一级分离结果 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/04_vocal_separation/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `separation_input.wav`、`vocals.wav` | 04 自行执行一级分离时的输入与总人声。复用 02 分轨时从 02 读取总人声 |
| `lead.wav`、`backing.wav` | 分离所得主唱和伴唱。实际移调为零时直接作为下游的 `vocals`、`backing` |
| `instrumental.wav` | 纯乐器。实际移调为零时作为下游的 `instrumental` |
| `vocals_transposed.wav`、`backing_transposed.wav`、`instrumental_transposed.wav` | 原编曲实际非零移调时，分别作为目标调的 `vocals`、`backing`、`instrumental` |
| `mix_bed.wav` | 同一调性的伴唱加纯乐器，06 将换声主唱混入此轨 |
| `cover.wav` | 原编曲模式的目标调对照，包含换声前主唱、伴唱和乐器 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 04_vocal_separation
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/vocal_separation.py`。默认模块配置：`configs/modules/04_vocal_separation.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
