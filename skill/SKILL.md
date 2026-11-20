---
name: music-cover-deploy
description: 在 music-cover-baseline 中部署项目内环境和固定模型，准备歌曲与目标参考，运行 00–06 音乐生成并定位产物和故障。支持 Seed-VC V1 零样本、全量微调、LoRA，以及 so-vits-svc、RVC、Fish Diffusion；训练、验证、导出与单曲推理分开执行。
---

# Music Cover 部署、训练与生成

使用仓库现有 Shell 入口，不另写一套模型流程。先定位同时包含 scripts/run_baseline.sh、src/music_cover 和 configs/pipeline.yaml 的根目录。本 skill 与项目 v2.0 配套，Seed-VC 仍使用 V1；核对实际 commit，旧发布分支未必包含当前接口。

## 先识别任务

- 已有素材生成歌曲：读 [USAGE](../USAGE.md)，先 00 再顺序执行到 06。
- 新机器/缺包：读 [部署参考](references/deployment.md)，再查 [权重参考](references/weights.md)。
- V1 LoRA：读 [LoRA 参考](references/lora.md)；需要可复制的完整步骤时读 [13 步手册](../docs/25_seedvc_lora_manual.md)。
- 全量微调、SVC、RVC、Fish：读 [训练参考](references/training.md)。
- 维护、提交准备：读 [CONTRIBUTING](../CONTRIBUTING.md)，同步代码、CLI 帮助、使用文档和本 skill。

| 05 路线 | 必要输入 |
|---|---|
| Seed-VC zero_shot | 公共权重 + 任务目标参考 |
| Seed-VC finetuned | 全量音色导出包 + 参考 |
| Seed-VC lora | V1 LoRA 导出包 + 匹配底模/原生补丁 + 参考 |
| so-vits-svc | 人物 SVC 导出包 |
| RVC | 人物 RVC 导出包和索引 |
| Fish Diffusion | 人物扩散模型导出包 |

只有 Seed-VC 提供本项目的参考音频零样本路线。SVC/RVC/Fish 的任务参考仍供音区分析与质量评估，不作为这些后端的零样本提示。

## 执行约定

1. 所有相对路径按项目根解析。环境、缓存、模型和数据留在项目内，不创建软链接；不复制旧服务器挂载点。入口加载 scripts/environment.sh。
2. controller 用 envs/control，01/08 用 envs/sheetsage2；02–06 与人物训练、验证用 envs/base；07/09 及评审打包用 envs/separator。以 common/resources.py 和本次 command.json 为准。
3. 新配置放 configs/local，公开例子放 configs/examples。部署先检查已有修改、子模块和模型哈希，不覆盖未知用户改动。
4. 训练与主流程独立。仅在用户要求训练时启动 01_train；已有导出包可直接推理。用户授权复用数据时核对 manifest、分组与哈希，不重新切分，不把复用说成新训练。
5. 需要单曲成品时执行至 06；仅准备编曲时至 04。--no-review 只跳过 07–09，仍运行 10–11。
6. fork 引用源 run 的 00–04，必须保留源 run。需要独立人物任务则新建配置从 00 实际执行，不用复制状态冒充。
7. LoRA validation 负责选 checkpoint，test 只审核已选定的一个；空变量先修正。test 可省略 checkpoint 自动读取同一训练身份已封存的 validation 选择。
8. 参考 PCM 与 LoRA profile 默认严格匹配；用户明确要复用另一音色的编曲时使用 --voice-reference-policy profile，并解释仍保留源任务元信息。

## 验证和交付

用 stage_state.json（主流程）或 state.json（训练）定位有效 attempt，不按目录最大编号猜测。查看 command.json、process.log、outputs.json 与 failure.json。

代码维护先做源码/CPU 回归；没有实际训练或 GPU 推理时如实说明。新部署真实生成验收应检查音频解码、时长、当前 profile/底模/adapter 哈希，并提供 04/05/06 对照。ECAPA、音高指标不是身份准确率或主观试听评分；区分默认 selected 与合格 recommended。

只到 06 时 final.wav/mp3 在该阶段 artifacts；顶层 deliverables 属于 11。复制试听文件前后核对 SHA256，防止同名文件串版。报告实际路线、音色版本、成功 run/attempt、输出路径及验证边界。
