# Seed-VC V1 LoRA

本项目 LoRA 接在 V1 44.1 kHz/F0-conditioned DiT 的 Attention wqkv/wo 与 FFN w1/w2/w3。rank 16 / alpha 16 / dropout 0 时为 85 层、170 个张量、3,551,232 参数；辅助模型与底模冻结。不涉及 V2。

完整手动命令见 [13 步手册](../../docs/25_seedvc_lora_manual.md)，入口速查见 [LoRA README](../../scripts/05_voice_conversion/lora/README.md)。

## 按任务使用

- 复用成品：验证完整 voice.json 导出目录，用 lora/run.sh 指向 profile；不运行训练。
- 新训练：原始录音分离 → 按录音分组的数据配置 → create_config（可 --plan）→ 00_prepare → 01_train → validation → test → 03_export。
- 复用数据：--dataset-dir 保留原 manifest、分组和划分；参考所属录音独立留出。
- 导入旧实验：01_train --import-run 仅导入已审计的 adapter 与证据，training_performed=false；不能把导入说成重训。随后验证和导出。
- 恢复训练：同 run 使用 --resume --max-steps <累计目标>，核对底模、数据、配置和 checkpoint。导出 adapter 不带优化器，不能用于恢复。

## 检查和排错

create_config 提前检查参考格式、文件哈希、验证源及窗口。--plan 不创建文件、不启动模型训练。窗口 start:duration 必须落在源音频内。

02_validate 默认 validation；不指定 checkpoint 使用最终 adapter。test 必须在同一 training identity 的成功 validation 之后；不指定 checkpoint 时使用最近一次成功选择。已失败、被篡改或其他训练身份的报告不能授权 test。空参数/前置条件失败应不产生新 attempt、不覆盖已有成功状态。

出现 “empty value” 先检查 VC_CFG、VC_TRAIN_RUN、VC_CKPT；出现 “run validation first” 先运行 validation，不删除 state 绕过约束。test 不接受多个 checkpoint，也不允许使用未被 validation 选中的权重。

## 推理与可复现性

安装器 scripts/setup/apply_seed_vc_lora.py 对固定上游应用 patches/seed-vc-lora-v1，--check 只读。不得仅留下子模块本地修改而漏交主仓库补丁。

05 默认检查任务参考与 profile 参考的 PCM；需要采用导出参考时显式 --voice-reference-policy profile。核对导出 voice_id、adapter/底模哈希、conversion.json，再交给 06 混音。训练 run、数据集和源歌曲 run 不随 Git 发布。
