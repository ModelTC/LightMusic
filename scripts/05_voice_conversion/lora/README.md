# Seed-VC V1 LoRA：训练、导出和 05→06

按顺序复制命令请读 [13 步手册](../../../docs/25_seedvc_lora_manual.md)。本入口只支持 V1 44.1 kHz/F0-conditioned 模型，不涉及 V2；05 不隐式训练。环境与缓存通过 scripts/environment.sh 放在项目内。

## 初次部署

```bash
git submodule update --init third_party/seed-vc
python3 scripts/setup/apply_seed_vc_lora.py
python3 scripts/setup/apply_seed_vc_lora.py --check
```

主仓库 patches/seed-vc-lora-v1 固定上游、补丁与文件哈希。安装器可重复执行，但拒绝覆盖不同修改或部分安装；不能只提交子模块工作区变化而漏交补丁。

## 配置和数据

```bash
bash scripts/05_voice_conversion/lora/training/create_config/run.sh \
  --voice-id my_voice --version r16_v001 \
  --dataset-dir data/voice_training/my_voice/v001/dataset \
  --reference data/raw/references/my_voice/reference.wav \
  --reference-group my_voice:held_out_recording \
  --source-run runs/my_cover_v001 --validation-windows 36:12 156:12 \
  --rank 16 --alpha 16 --dropout 0 --learning-rate 1e-4 \
  --max-steps 1000 --save-every 250 --batch-size 1 --seed 42 \
  --validation-steps 30 --validation-seed 42 --gpu 0 \
  --output configs/local/my_voice_lora_r16_v001.yaml --plan
```

核对后去掉 --plan 写入新 YAML。已有 dataset 必须包含 manifest 与 train/validation/test；复用保留原划分、检查全部文件哈希。全新数据使用 --data-config 代替 --dataset-dir，配置来自公共 voice_data/create_config，含 dataset_dir、sources、group、split。

参考为独立录音的 1–25 秒、44.1 kHz 单声道 WAV。reference-group 不得进入数据集。验证源来自成功 04；也可用 --validation-source 指定固定源人声。窗口必须落在音频时长内，本例要求至少 168 秒，且需实际试听确认有声。

可选 --full-profile 加入同人物全量微调比较。配置预览会检查输入、数值范围和窗口，不创建目录、不训练。

## 四阶段与 test

```bash
export VC_CFG=configs/local/my_voice_lora_r16_v001.yaml
export VC_TRAIN_RUN=runs/voice_training/my_voice_lora_r16_v001

bash scripts/05_voice_conversion/lora/training/00_prepare/run.sh --config "$VC_CFG" --run "$VC_TRAIN_RUN"
# 下一条实际启动训练。
bash scripts/05_voice_conversion/lora/training/01_train/run.sh --config "$VC_CFG" --run "$VC_TRAIN_RUN"
bash scripts/05_voice_conversion/lora/training/02_validate/run.sh --config "$VC_CFG" --run "$VC_TRAIN_RUN" --evaluation-split validation
bash scripts/05_voice_conversion/lora/training/02_validate/run.sh --config "$VC_CFG" --run "$VC_TRAIN_RUN" --evaluation-split test
bash scripts/05_voice_conversion/lora/training/03_export/run.sh --config "$VC_CFG" --run "$VC_TRAIN_RUN"
```

validation 默认使用最终 adapter，可重复 --checkpoint 比较登记的中间权重。test 默认读取最近一次成功 validation 的选定权重，且必须匹配当前 training identity；也可显式传一个已选中的 checkpoint。不能跳过 validation 或用 test 挑模型。

--checkpoint "" 属于参数错误；应重新设置 VC_CKPT 或省略可选参数。缺失文件、未完成 validation 等错误在新 attempt 之前返回，保留已有成功状态。失败/篡改/其他训练身份的报告不能授权 test。声纹指标不代表身份准确率。

## 恢复与导入

```bash
# 累计目标 2000 步；只在明确继续训练时执行。
bash scripts/05_voice_conversion/lora/training/01_train/run.sh \
  --config "$VC_CFG" --run "$VC_TRAIN_RUN" --resume --max-steps 2000

# 已审计旧实验导入时，用此条代替正常 01_train。
bash scripts/05_voice_conversion/lora/training/01_train/run.sh \
  --config "$VC_CFG" --run "$VC_TRAIN_RUN" --import-run runs/existing_lora_experiment
```

恢复核对数据、底模、配置、代码和 checkpoint，恢复优化器/调度器/RNG；原生 DataLoader 不保证 epoch 中途逐样本完全复现。导入需要 adapter.pth、training.json、training_audit.json、inputs.json 及模型配置，只复制证据，记录 training_performed=false；随后仍需验证和导出。

rank 16 对 Attention+FFN 的 85 个线性层产生 170 个 A/B 张量、3,551,232 个训练参数。底模/辅助模型冻结，并记录冻结参数审计。

## 导出与单曲推理

导出默认位于 models/voices/<voice_id>/seed_vc_lora/<version>，包含 voice.json、adapter、模型配置、参考和验证证据。底模通过哈希绑定；不能仅搬一个裸 .pth 代替导出包。已存在版本不覆盖。

```bash
bash scripts/05_voice_conversion/fork_run.sh --source-run runs/my_cover_v001 --run runs/my_cover_lora_v001
bash scripts/05_voice_conversion/lora/run.sh \
  --run runs/my_cover_lora_v001 \
  --voice-profile models/voices/my_voice/seed_vc_lora/r16_v001/voice.json \
  --voice-candidates 1 --voice-candidate-seeds 42 --gpu 0
bash scripts/06_mix/run.sh --run runs/my_cover_lora_v001 --gpu 0
```

可先在 05 命令末尾加 --preflight。默认要求任务参考与 profile 参考的解码 PCM 一致；明确借用不同人物编曲时加 --voice-reference-policy profile。fork 引用源 00–04，保留源 run；输出以 stage_state.json 的有效 attempt 为准。

CPU 回归和提交检查见 [CONTRIBUTING](../../../CONTRIBUTING.md)。

恢复训练要求原训练控制器代码身份一致；代码更新后旧 run 可能因此拒绝 resume，应保留原代码版本继续旧实验，或用新版本新建训练。既有权重的 validation、导出和推理不要求重训；不要改写 identity.json 绕过检查。
