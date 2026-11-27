# LightMusic 仓库、运行数据与发布边界

源码包括 src、scripts、tests、configs 模板/锁、docs、skill、patches；运行资产放 envs、cache、models、data、runs。均在项目根目录下，不要求软链接。

## 仓库与当前工作目录

仓库为 [ModelTC/LightMusic](https://github.com/ModelTC/LightMusic)，默认开发分支为 `main`，由原 `v2.0` 分支改名而来；项目版本号仍为 `v2.0`。更名前的旧 `main` 保存在 `archive/main-before-v2.0-20261127`。

主入口 [README.md](../README.md) 为英文，[README_CN.md](../README_CN.md) 为中文。本文档中的服务器示例统一使用 `/mnt/devoversea_afs_2/liuhongda/LightMusic`；其他机器以实际克隆位置为准。

项目名称更新不更改 Python 包 `music_cover`、CLI 入口 `scripts/run_baseline.sh` 或既有运行编号。历史运行日志、模型导出包与 installed 环境快照保留原始字节和哈希；不要用全文替换重写已封存的验收证据。迁移后的运行状态通过项目路径解析器读取。

## 哪些文件提交

| 内容 | 策略 |
|---|---|
| 源码、测试、公开模板和使用文档 | 审阅后提交 |
| 原生 LoRA 补丁与 manifest | 随主仓库提交；子模块保持固定 gitlink |
| 新本地配置 | configs/local，已忽略 |
| 权重、录音、切片、训练 checkpoint、音频结果 | 不提交 |
| 既有实验配置/验证文档 | 作为历史示例保留，真实素材不分发 |
| installed 环境快照 | 原始取证及缓存身份；本地 wheel 路径在发布检查中单独警告 |

.gitignore 不会自动移除已跟踪文件。遗留机器专用、尚未跟踪的配置可在本机 .git/info/exclude 精确排除；不要移动用户当前使用的配置而破坏环境变量。发布检查见 [CONTRIBUTING](../CONTRIBUTING.md)。

## 子模块与复现

```bash
git submodule update --init third_party/YuE third_party/seed-vc third_party/so-vits-svc third_party/rvc
python3 scripts/setup/apply_seed_vc_lora.py
python3 scripts/setup/apply_seed_vc_lora.py --check
```

LoRA 修改由 patches/seed-vc-lora-v1 可复现安装，不把本地改过的子模块目录当作已经发布。Fish 使用自己的固定源码下载/补丁流程。不要 update --remote 或通过强制 reset 覆盖工作区。

## 数据保留与迁移

有效主流程产物由 stage_state.json、训练产物由 state.json 定位，并用 outputs.json 的大小/哈希核对。fork 只引用源 00–04；删除源 run 前先分析引用关系，不能只看目录名。

训练 run、导出包和源素材承担不同职责。导出包用于推理，adapter 不含优化器，不能替代训练 checkpoint。同名试听文件复制后核对 SHA256；不要手工替换已封存的历史产物。

跨机器需要迁移数据/模型、核对路径与哈希，并在新前缀重建环境。环境取证使用 capture_environment.py --output-dir cache/...；不要在只读检查时更新参与运行身份的锁。当前部署说明见 [环境文档](03-environment.md)。

## 自动检查的范围

GitHub 轻量工作流只做源码、配置、文档链接、LoRA 工作流单元测试和发布内容检查。服务器还运行完整 CPU 回归。模型加载、真实训练、05→06 音频与试听属于另外的授权验收，不由 CI 成绩代替。
