# 智能检测算法验证台

竞赛演示：真实 YOLO 剪枝+INT8 代码面板 + 模拟流水线终端 + YOLO / 质减量增对比。

**说明：** 演示端不跑真实权重推理（无 GPU / bch.pt），检测框与指标为 mock；代码区展示用户真实 `prune_quantize.py` / `validate_testset.py`。

## 运行

```bash
cd 智能检测算法验证台
python3 app.py
# 打开 http://127.0.0.1:8765
```

## 真实代码

- `code/prune_quantize.py` — YOLO load → L1 Conv2d prune → ONNX → dynamic INT8
- `code/validate_testset.py` — YOLO val on test split → mAP / P / R
