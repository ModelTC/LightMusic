# 运行环境与迁移

从 ModelTC/music-cover-baseline 的 v2.0 分支获取本版代码，保留固定子模块。已有工作树先检查改动，不覆盖未知修改。新克隆使用 `git clone --branch v2.0 https://github.com/ModelTC/music-cover-baseline.git`。

```bash
git status --short
git rev-parse HEAD
git submodule status
git submodule update --init third_party/YuE third_party/seed-vc third_party/so-vits-svc third_party/rvc
python3 scripts/setup/apply_seed_vc_lora.py
python3 scripts/setup/apply_seed_vc_lora.py --check
```

不要 update --remote。Fish 源码由自己的下载器按资产锁获取并应用 patches/fish_diffusion，不是第五个 gitlink。

## 环境布局与检查

解释器位于项目 envs，缓存位于 cache，不要求软链接：

| 用途 | 解释器 |
|---|---|
| controller/00 | envs/control/bin/python |
| 01/08 SheetSage2 | envs/sheetsage2/bin/python |
| 02–06 后端、音色训练与评估 | envs/base/bin/python |
| 07/09、评审与打包 | envs/separator/bin/python |

```bash
source scripts/environment.sh
nvidia-smi
envs/base/bin/python -c 'import sys,torch; print(sys.executable, torch.__version__, torch.version.cuda, torch.cuda.is_available())'
ffmpeg -version
ffprobe -version
bash scripts/run_baseline.sh --config configs/local/my_cover.yaml --until 06_mix --preflight
```

FFmpeg 需支持 libmp3lame。环境记录见 configs/environments.lock.json；配置中 memory_budget_gib 可调整，示例为 64，不代表任意显存/硬件都已验证。不要缩短歌曲来掩盖显存不足。

## 安装边界

[环境说明](../../docs/03-environment.md) 给出当前布局和版本依据。requirements-h100-base.txt 与 constraints-h100.txt 是已部署 H100 的依赖依据，仍需匹配 PyTorch/CUDA 和原生扩展，不是经空白机器验证的一键安装器。不要直接安装上游全部 requirements 覆盖可用环境。

configs/*.installed.txt 是历史原始快照，可能含旧 wheel 本地路径，并参与运行身份；不能直接当 requirements 使用。迁移 envs 时在目标前缀重建，避免搬动包含绝对路径的环境。不要修改 H200 环境来满足 H100 部署。

普通只读审计使用：

```bash
envs/base/bin/python scripts/setup/capture_environment.py --output-dir cache/environment_snapshot --skip-assets
```

不传 output-dir 会更新运行锁并影响缓存身份；仅在确实修改环境且需要记录时使用。权重按 [weights](weights.md) 固定版本准备，新部署真实推理单独验收。

## 故障定位

先查看本次 command.json 的解释器、process.log 和 failure.json；区分缺包、模型文件缺失、哈希不匹配与 CUDA OOM。预检或 --help 成功不等于模型加载成功；--model-load 也不等于人物整曲推理通过。

旧文档中系统 Conda 与旧挂载点是历史环境证据，当前入口以本页和源码为准。LoRA 的空 checkpoint、validation/test 顺序见 [LoRA 参考](lora.md)。
