import os
import yaml
from ultralytics import YOLO

def validate_testset():
    # 路径配置（当前脚本所在目录为根目录）
    project_root = "."
    model_path = os.path.join(project_root, "weights", "bch.pt")  # 模型：./weights/bch.pt
    data_config_path = os.path.join(project_root, "data.yaml")    # 数据集配置：./data.yaml（修正路径）
    result_save_path = os.path.join(project_root, "runs", "test_validation")  # 结果保存：./runs/test_validation

    # 创建结果目录
    os.makedirs(result_save_path, exist_ok=True)

    # 加载并校验数据集配置
    with open(data_config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
        # 拼接测试集完整路径（path + test）
        test_dir = os.path.join(data_config.get("path", "."), data_config.get("test", "test/images"))
        print(f"测试集实际路径: {test_dir}")
        # 校验测试集是否存在
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"测试集目录不存在：{test_dir}\n请检查 data.yaml 中 'path' 和 'test' 配置！")

    # 校验模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}\n请检查 weights 文件夹！")

    # 加载模型
    model = YOLO(model_path)
    print(f"模型加载成功: {model_path}")

    # 测试集验证参数
    val_params = {
        "data": data_config_path,  # 数据集配置文件
        "split": "test",           # 明确使用测试集（对应 data.yaml 的 test 字段）
        "device": "0",             # GPU设备（无GPU则改 "cpu"）
        "save": True,              # 保存预测结果和评估图表
        "save_dir": result_save_path,  # 结果保存目录
        "plots": True,             # 生成PR曲线、混淆矩阵等
        "verbose": True            # 显示详细日志
    }

    # 执行验证
    print("开始测试集验证...")
    metrics = model.val(**val_params)

    # 保存评估指标到文件
    metrics_path = os.path.join(result_save_path, "test_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("### 测试集评估结果 ###\n")
        f.write(f"整体mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP75: {metrics.box.map75:.4f}\n")
        f.write(f"平均精确率 (P): {metrics.box.mp:.4f}\n")
        f.write(f"平均召回率 (R): {metrics.box.mr:.4f}\n\n")
        f.write("### 各类别指标 ###\n")
        for i, class_name in enumerate(metrics.names):
            f.write(f"{class_name}: mAP50-95={metrics.box.maps[i]:.4f}\n")

    print(f"验证完成！结果已保存至：{result_save_path}")

if __name__ == "__main__":
    validate_testset()