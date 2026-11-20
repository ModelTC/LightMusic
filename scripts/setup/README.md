# 环境、模型与素材准备

本目录不属于单曲 00–11。命令从项目根目录执行，先 source scripts/environment.sh。环境部署见 [环境说明](../../docs/03-environment.md)，完整下载版本与哈希见 [权重参考](../../skill/references/weights.md)。

| 操作 | 入口 |
|---|---|
| SheetSage2/MERT/YuE2/VAE | envs/control/bin/python scripts/setup/download_models.py |
| BS-RoFormer | envs/control/bin/python scripts/setup/download_separator_model.py |
| Seed-VC/F0 公共模型 | envs/control/bin/python scripts/setup/download_voice_models.py |
| 评审与主唱分离资产 | envs/separator/bin/python scripts/setup/setup_review_assets.py |
| 训练/评估附加资源 | envs/base/bin/python scripts/setup/setup_voice_training.py --scope all --check-only |
| 原生 V1 LoRA 补丁 | python3 scripts/setup/apply_seed_vc_lora.py |
| 补丁只读校验 | python3 scripts/setup/apply_seed_vc_lora.py --check |

下载器会写入模型与相应锁；已有部署先核对，避免覆盖用户配置。LoRA 安装器要求固定上游和哈希，拒绝覆盖未知本地修改；主仓库补丁必须随代码发布。

```bash
source scripts/environment.sh
bash scripts/run_baseline.sh --config configs/local/my_cover.yaml --until 06_mix --preflight
envs/base/bin/python scripts/setup/capture_environment.py \
  --output-dir cache/environment_snapshot --skip-assets
```

环境捕获指定 output-dir 不改 configs 运行锁；省略会更新缓存身份。检查源代码使用 scripts/check_project.py，不需要重新计算全部模型哈希。演示素材下载器 prepare_demo.py 只准备它声明的公开演示素材，不提供历史实验歌手数据。
