# 代码迭代与 GitHub 检查

先查看已有改动，不要 reset 掉未提交的部署与 LoRA 实现。新实验配置放 `configs/local/`，公开模板放 `configs/examples/`。环境、缓存、模型、数据和 runs 不进入源码提交。

## 开发前

```bash
git status --short
git diff --stat
git submodule status
source scripts/environment.sh
envs/base/bin/python scripts/setup/apply_seed_vc_lora.py --check
```

LoRA 原生实现通过 [补丁 manifest](patches/seed-vc-lora-v1/manifest.json) 固定到 Seed-VC 上游 commit。首次部署执行 `python3 scripts/setup/apply_seed_vc_lora.py`；安装器拒绝覆盖不匹配的本地修改。改原生 LoRA 时同步补丁、哈希和测试，不能只留下子模块 dirty 标记。

## 源码与 CPU 回归

已部署服务器：

```bash
envs/base/bin/python scripts/check_project.py --output cache/source_check.json
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src envs/base/bin/python -m unittest discover -s tests -v
```

无 GPU/模型的轻量 CI 使用 Python 3.11：

```bash
python -m pip install -r configs/requirements-ci.txt
python scripts/check_project.py
PYTHONPATH=src python -m unittest discover -s tests -p 'test_lora*.py' -v
python -m unittest discover -s tests -p 'test_release_checks.py' -v
```

[source-checks.yml](.github/workflows/source-checks.yml) 在 push/PR 执行上述检查与全索引发布检查。它不训练、不下载模型，不代表歌曲听感验收。原生小型 CPU LoRA 测试需要 Torch，位于 `tests/test_native_seed_vc_lora.py`，随服务器全量测试运行。

回归覆盖空变量、validation/test 顺序、checkpoint/报告哈希、失败 attempt 隔离、窗口越界和只读配置预览。模型行为改动还需在授权的真实任务上验证 05→06、解码及模型/产物哈希。不要为通过测试修改历史数据划分和证据。

## 提交前

逐项审阅后暂存本次准备提交的源码：

```bash
git diff --check
git diff --stat
# git add <已审阅的文件>
git diff --cached --stat
envs/base/bin/python scripts/check_release.py --staged --output cache/release_check.json
```

`--staged` 检查实际暂存字节，`--tracked` 检查整个索引。检查器不会自动暂存、提交或推送。

检查器拒绝运行产物、模型/媒体、私有配置、执行配置中的机器路径、常见密钥文件与超大文件；它不是完整秘密扫描器，仍需审阅 diff。子模块只记录固定 gitlink。

以下现有文件保留：

- `configs/*.installed.txt` 是历史环境取证与运行身份，可能含本地 wheel 来源。检查器明确列为 warning；它们不是可移植安装清单，不能直接 pip install。本次保留字节，避免使已完成的 00–04 缓存身份失效。以后从发布中去掉历史路径时，需要同步迁移运行身份策略。
- `configs/seed_svc.local.yml` 已跟踪且使用项目相对模型路径，是运行配置；其他新本地配置放 `configs/local/`。
- 历史实验文档可保留，但素材不随 Git 提供；新使用入口不要依赖某台服务器或旧 run。

后续迭代先在 CHANGELOG 中记录未发布变化；准备新版本时同步 VERSION、README、skill 和验证范围。本版源码位于 main 分支；创建 tag 或 GitHub Release 是另一个显式发布动作。
