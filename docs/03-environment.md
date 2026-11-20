# 当前运行环境与恢复边界

当前代码统一使用项目内解释器。各 Shell 入口加载 scripts/environment.sh，把环境、模型缓存、临时文件放在项目下；不依赖固定服务器挂载点，不需要软链接。

| 环境 | 主要用途 |
|---|---|
| envs/control/bin/python | 总入口、00 与调度 |
| envs/sheetsage2/bin/python | 01/08 转谱 |
| envs/base/bin/python | 02/04 分离与 F0、03 YuE2、05 各路线、06；音色训练与评估 |
| envs/separator/bin/python | 07 分轨、09 MIDI、评审打包 |

实际调度以 src/music_cover/common/resources.py 与当前 command.json 为准。所需环境不能仅由当前 Shell 激活名称判断。

## 已部署环境检查

```bash
source scripts/environment.sh
envs/base/bin/python -c 'import sys, torch; print(sys.executable, sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
ffmpeg -version
ffprobe -version
bash scripts/run_baseline.sh --config configs/local/my_cover.yaml --until 06_mix --preflight
envs/base/bin/python scripts/setup/apply_seed_vc_lora.py --check
```

FFmpeg 需要 libmp3lame。YuE2 的 memory_budget_gib 通过 03 模块配置传入，H100 示例使用 64 GiB；这不是最低显存实测，不表示已验证所有 GPU。曲长与上下文会影响显存。

## 安装与版本依据

当前 H100 base 核心栈为 Python 3.11、PyTorch 2.8.0/CUDA 12.8、NumPy 2.2.6、SciPy 1.15.3、Transformers 4.57.3。参阅 [依赖清单](../configs/requirements-h100-base.txt)、[约束](../configs/constraints-h100.txt) 和 configs/environments.lock.json；01/评审环境使用各自快照，不要直接合并环境。

这些文件是已部署环境的恢复依据，不是一份经过空白机器验证的完整锁。Torch/CUDA、FlashAttention wheel、YuE2 固定源码及个别 --no-deps 安装仍需按目标机器核对。不能把 envs 原目录直接移动后宣称环境已迁移成功。历史 pip 元数据约束冲突与指定模型路径的运行验证应分别报告。

模型来源和哈希见 [权重参考](../skill/references/weights.md)。仅做 CPU 源码/CLI 回归的轻量环境见 [CONTRIBUTING](../CONTRIBUTING.md)，无需 GPU 或模型。

## 取证与缓存身份

```bash
envs/base/bin/python scripts/setup/capture_environment.py \
  --output-dir cache/environment_snapshot --skip-assets
```

指定 output-dir 生成独立快照，不更新 configs 里的运行身份锁；省略该参数会更新锁文件。configs/*.installed.txt 为原始 pip freeze 取证，可能有本地 wheel URL，不能直接用作安装清单。它们参与缓存校验，不应为了格式整齐随意改写。

旧 H200、系统 Conda 路径和原实验成绩保留在 [历史验证](../VALIDATION.md) 及历史专题文档，不能作为当前解释器路径说明。新增 LoRA 手动流程见 [13 步手册](25_seedvc_lora_manual.md)。
