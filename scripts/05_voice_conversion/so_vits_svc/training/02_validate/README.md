# 02_validate

读取 01 检查点、基准 04 源人声，以及独立 validation 参考，输出定量报告。

```bash
# 在项目根目录执行
bash scripts/05_voice_conversion/so_vits_svc/training/02_validate/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028 \
  --checkpoint runs/voice_training/yang_kun_so_vits_svc_v001_20261028/checkpoints/step_00001000.pth
```

本阶段记录：`runs/voice_training/yang_kun_so_vits_svc_v001_20261028/stages/02_validate/attempt_*/`。具体产物、前后依赖和续训示例见 [完整指南](../../../../../docs/18_so_vits_svc.md)。
