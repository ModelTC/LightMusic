# Seed-VC V1 LoRA 手动实操

按“输入 → 权重 → 环境 → 数据 → 配置 → 准备 → 训练 → validation → test → 导出 → 独立推理 run → 05 → 06”的顺序操作。示例采用 Attention+FFN、rank 16、1000 步。命令需按顺序在同一终端执行；新终端先重新设置第 3 步变量。示例名称须替换为自己的真实素材，不下载任何人的录音。

## 1. 先明确需要哪些输入

| 输入 | 项目内位置示例 | 用途 |
|---|---|---|
| 项目根目录 | 当前仓库目录，设置为 `VC_ROOT` | 代码、环境、缓存的共同根目录 |
| 已完成的歌曲 00–04 | `$VC_ROOT/runs/my_cover_v001` | 提供实际 04 人声作固定验证源，之后供 05/06 复用 |
| 原始目标音色录音 | `data/raw/references/my_voice/sources/<录音ID>/original.wav` | 多份独立录音，训练前分离主唱 |
| 目标参考 | 本次歌曲 00 的 `reference.wav` | 1–25 秒、44.1 kHz、单声道 WAV；来自独立保留录音 |
| 验证/测试录音 ID | 例 `validation_01`、`test_01` | 按完整录音分组，禁止同一录音跨集合 |
| 新数据集、新训练 run、新模型版本 | 由第 3 步定义 | 避免覆盖旧实验 |

如尚无该音色的独立 00–04，先按 [USAGE](../USAGE.md) 设置该人物参考与曲风，并运行到 `04_vocal_separation`。fork 只引用源 run，并不产生独立 00–04。这里的 04 人声是评估输入，不能当作目标音色训练材料。

## 2. 确认模型及公共权重

需要 V1 44.1 kHz/F0-conditioned 底模、Whisper-small、CAMPPlus、BigVGAN、RMVPE；训练还需 OpenVoiceV2/`se_db.pt`，验证需 ECAPA，原始录音分离需两级分离模型。下载命令见 [权重表](../skill/references/weights.md)，不要用 V2 或其他底模替换。

首次克隆后安装本项目原生 LoRA 补丁；已有环境先检查：

```bash
git submodule update --init third_party/seed-vc
python3 scripts/setup/apply_seed_vc_lora.py
python3 scripts/setup/apply_seed_vc_lora.py --check
```

## 3. 进入 base，并设置本次路径

先进入实际项目根目录。base 位于项目的 `envs/base`，不依赖系统 Conda：

```bash
export VC_ROOT="$PWD"
test -f "$VC_ROOT/scripts/environment.sh"
source "$VC_ROOT/scripts/environment.sh"

export VC_VOICE="my_voice"
export VC_VERSION="lora_r16_v001"
export VC_SOURCE_RUN="$VC_ROOT/runs/my_cover_v001"
export VC_RECORDINGS="$VC_ROOT/data/voice_training/$VC_VOICE/$VC_VERSION/recordings"
export VC_DATASET="$VC_ROOT/data/voice_training/$VC_VOICE/$VC_VERSION/dataset"
export VC_DATA_CFG="$VC_ROOT/configs/local/${VC_VOICE}_${VC_VERSION}_data.yaml"
export VC_CFG="$VC_ROOT/configs/local/${VC_VOICE}_${VC_VERSION}.yaml"
export VC_TRAIN_RUN="$VC_ROOT/runs/voice_training/${VC_VOICE}_${VC_VERSION}"
export VC_EXPORT="$VC_ROOT/models/voices/$VC_VOICE/seed_vc_lora/$VC_VERSION"
export VC_INFER_RUN="$VC_ROOT/runs/${VC_VOICE}_${VC_VERSION}_cover"

# 读取当前成功 00 的参考，不硬编码 attempt_001。
export VC_REFERENCE="$("$VC_ROOT/envs/base/bin/python" - <<'PY'
import os
from pathlib import Path
from music_cover.common.paths import read_run_state
from music_cover.common.voice_config import stage_artifacts
state = read_run_state(Path(os.environ["VC_SOURCE_RUN"]))
print(stage_artifacts(state, "00_prepare", ["reference"])["reference"])
PY
)"
test -f "$VC_REFERENCE"
"$VC_ROOT/envs/base/bin/python" scripts/setup/setup_voice_training.py --scope all --check-only
```

命令均在根目录执行。参考录音 ID 以下用 `reference_01`；请替换为真实保留录音 ID，并确保它未进入任一训练/评估集合。

## 4. 从原始音频重新分离目标人声

```bash
bash "$VC_ROOT/scripts/voice_data/separate/run.sh" \
  --input-dir "$VC_ROOT/data/raw/references/$VC_VOICE/sources" \
  --pattern '*/original.wav' --output-dir "$VC_RECORDINGS" \
  --exclude-ids reference_01 --gpu 0 --plan
```

先检查清单，再去掉 `--plan` 执行实际分离。素材是 M4A 时改为 `'*/original.m4a'`。输出为每份录音的 `lead.wav`、其他分轨与总 `manifest.json`，还没有训练切片。整批成功并试听主唱后继续。

已有经过核验的数据集可以跳过分离和下一步公共数据配置，直接用 LoRA 配置生成器的 `--dataset-dir`；它会核对 manifest、文件哈希、录音分组和划分，不会重新随机切分。

## 5. 生成本次训练配置

### 1. 必须对应已有数据的路径

原始分离目录、参考 WAV、00–04 run、底模必须实际存在。以下 ID 是占位示例，改为已分离的真实编号：

```bash
bash "$VC_ROOT/scripts/voice_data/create_config/run.sh" \
  --voice-id "$VC_VOICE" --version "$VC_VERSION" \
  --recordings-dir "$VC_RECORDINGS" --source-run "$VC_SOURCE_RUN" \
  --validation-ids validation_01 --test-ids test_01 \
  --segment-seconds 8 --dataset-dir "$VC_DATASET" \
  --output "$VC_DATA_CFG"
```

### 2. 由使用者命名的身份与版本

`voice-id` 标识人物；`version` 标识这次模型。配置、dataset、训练 run 和 export 分开存放。已有导出版本不可覆盖，重开实验使用新版本。

### 3. 由你设计，但受实际素材约束的参数

`reference-group` 必须和数据集使用同一命名规则，例如 `my_voice:reference_01`，且该录音不能进入 train/validation/test。`36:12 156:12` 表示两段起点和时长，只适用于源音频至少 168 秒且对应区间有演唱的情况；短音频必须改窗。

### 4. 可以自行调整的训练与验证参数

```bash
bash "$VC_ROOT/scripts/05_voice_conversion/lora/training/create_config/run.sh" \
  --voice-id "$VC_VOICE" --version "$VC_VERSION" \
  --data-config "$VC_DATA_CFG" \
  --reference "$VC_REFERENCE" --reference-group "$VC_VOICE:reference_01" \
  --source-run "$VC_SOURCE_RUN" --validation-windows 36:12 156:12 \
  --rank 16 --alpha 16 --dropout 0 --learning-rate 1e-4 \
  --max-steps 1000 --save-every 250 --batch-size 1 --seed 42 \
  --validation-steps 30 --validation-seed 42 --gpu 0 \
  --export-dir "$VC_EXPORT" --output "$VC_CFG" --plan
```

检查输出后，去掉末尾 `--plan` 生成 YAML。plan 不写文件，也不启动训练。复用已存在切片时，把 `--data-config "$VC_DATA_CFG"` 替换为 `--dataset-dir "$VC_DATASET"`。

`max-steps` 是训练步数，`validation-steps` 是推理扩散步数。可用 `--full-profile <同音色全量微调voice.json>` 增加对比；不要求为了 LoRA 先训练全量模型。配置生成器提前拒绝空参数、缺文件、越界窗口、非法参数及不匹配人物。

## 6. 生成训练、验证和测试切片——训练 00_prepare

```bash
bash "$VC_ROOT/scripts/05_voice_conversion/lora/training/00_prepare/run.sh" \
  --config "$VC_CFG" --run "$VC_TRAIN_RUN"
```

输出 `dataset/manifest.json`、train/validation/test 音频及本次训练准备记录。已有数据集会严格核验并复用划分。

## 7. 微调目标音色模型——训练 01_train

此步真正启动训练：

```bash
bash "$VC_ROOT/scripts/05_voice_conversion/lora/training/01_train/run.sh" \
  --config "$VC_CFG" --run "$VC_TRAIN_RUN"
```

Attention+FFN rank 16 覆盖 85 个线性层、170 个 A/B 张量，共 3,551,232 个可训练参数；底模及辅助模型冻结。产物包括 `checkpoints/adapter.pth`、中间 checkpoint、loss 和冻结参数审计。

需要继续到累计 2000 步时使用同一入口加 `--resume --max-steps 2000`。导出的 adapter 不含优化器，不可代替训练恢复 checkpoint；DataLoader 的 epoch 内逐样本恢复不保证完全复现。

## 8. 验证最终 1000 步模型——训练 02_validate

```bash
export VC_CKPT="$VC_TRAIN_RUN/checkpoints/adapter.pth"
test -f "$VC_CKPT"
bash "$VC_ROOT/scripts/05_voice_conversion/lora/training/02_validate/run.sh" \
  --config "$VC_CFG" --run "$VC_TRAIN_RUN" \
  --evaluation-split validation --checkpoint "$VC_CKPT"
```

这是选择模型的 validation 阶段。可重复传 `--checkpoint` 比较已登记的中间权重；未传时使用最终 adapter。结果写入本次 attempt 的 `validation.json`，选择结果及文件 SHA256 被记录并封存。

## 9. 用独立测试集复核同一个模型

validation 成功后运行：

```bash
bash "$VC_ROOT/scripts/05_voice_conversion/lora/training/02_validate/run.sh" \
  --config "$VC_CFG" --run "$VC_TRAIN_RUN" \
  --evaluation-split test
```

不传 checkpoint 时，自动使用同一训练身份最近一次成功 validation 的选择。也可明确传一个已选中的 `--checkpoint "$VC_CKPT"`。test 不用于选模型；失败记录、其他训练身份、未选中的权重、被修改的报告/权重均不能授权 test。

若提示空参数，重新设置变量或省略可选 checkpoint；若提示没有 validation，回到第 8 步。不要传 `--checkpoint ""`。此类参数和前置条件错误会在创建 attempt 前终止，不把上次成功验证改成失败。

## 10. 导出本次音色模型——训练 03_export

```bash
bash "$VC_ROOT/scripts/05_voice_conversion/lora/training/03_export/run.sh" \
  --config "$VC_CFG" --run "$VC_TRAIN_RUN"
export VC_PROFILE="$VC_EXPORT/voice.json"
test -f "$VC_PROFILE"
```

导出包含 adapter、模型配置、参考、底模绑定信息与验证证据。整目录作为不可变版本保留；裸 adapter 不是完整音色包。仅执行 validation 后也可导出；本手册先做 test 再导出。

## 11. 为微调整曲结果建立独立运行目录

```bash
bash "$VC_ROOT/scripts/05_voice_conversion/fork_run.sh" \
  --source-run "$VC_SOURCE_RUN" --run "$VC_INFER_RUN"
```

独立的是 05/06 的状态与输出；00–04 仍引用源 run。不要删除源 run，也不要把它描述成重新生成了 00–04。

## 12. 用新微调模型转换整曲人声——主流程 05

```bash
bash "$VC_ROOT/scripts/05_voice_conversion/lora/run.sh" \
  --run "$VC_INFER_RUN" --voice-profile "$VC_PROFILE" \
  --voice-candidates 1 --voice-candidate-seeds 42 --gpu 0 --preflight
bash "$VC_ROOT/scripts/05_voice_conversion/lora/run.sh" \
  --run "$VC_INFER_RUN" --voice-profile "$VC_PROFILE" \
  --voice-candidates 1 --voice-candidate-seeds 42 --gpu 0
```

默认要求任务参考与导出参考 PCM 一致。明确借用其他人物编曲时，可加 `--voice-reference-policy profile` 使用导出参考，并核对该人物 ID；本手册使用同人物 00–04，无须此选项。

检查 `conversion.json`、候选报告与实际选择的模型/音色。ECAPA 不是身份概率；需要试听。

## 13. 生成整曲混音——主流程 06

```bash
bash "$VC_ROOT/scripts/06_mix/run.sh" --run "$VC_INFER_RUN" --gpu 0
bash "$VC_ROOT/scripts/run_baseline.sh" --run "$VC_INFER_RUN" --status
```

首次产物在 `$VC_INFER_RUN/stages/06_mix/attempt_001/artifacts/final.wav` 和 `final.mp3`，重跑读取 `stage_state.json` 的有效 attempt。只到 06 不生成顶层 deliverables。

复用同一 00–04 比较零样本/全量/LoRA 时分别 fork，固定 seed 与候选数，使用同人物参考。试听文件复制后核对 SHA256，不依赖同名 MP3 判断版本。

恢复训练要求原训练控制器代码身份一致；代码更新后旧 run 可能因此拒绝 resume，应保留原代码版本继续旧实验，或用新版本新建训练。既有权重的 validation、导出和推理不要求重训；不要改写 identity.json 绕过检查。
