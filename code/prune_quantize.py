from ultralytics import YOLO
import torch
from torch.nn.utils import prune
import onnxruntime.quantization as quant

model = YOLO("./weights/bch.pt")

def model_prune(model_obj, prune_ratio=0.2):
    for name, module in model_obj.model.named_modules():
        if hasattr(module, "weight") and isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            prune.remove(module, "weight")
    return model_obj

pruned_model = model_prune(model, prune_ratio=0.2)

pruned_model.export(format="onnx", opset=17, simplify=True)

onnx_input_path = "./weights/bch.onnx"
onnx_quant_out = "./weights/bch_int8.onnx"

quant.quantize_dynamic(
    model_input=onnx_input_path,
    model_output=onnx_quant_out,
    weight_type=quant.QuantType.QUInt8
)
print("剪枝+INT8量化完成")