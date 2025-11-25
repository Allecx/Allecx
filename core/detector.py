# core/detector.py
import torch
from ultralytics import YOLO


class Detector:
    def __init__(self):
        self.model = None
        self.task = None
        self.names = None

    def load_model(self, model_path: str, model_format: str):
        """
        根据路径和格式加载 YOLO 模型
        注意：.pth 需要特殊处理，但 Ultralytics 一般只支持 .pt 和 .onnx
        """
        try:
            # 直接使用 YOLO 加载（自动识别 pt/onnx）
            self.model = YOLO(model_path)

            # 自动选择设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)

            # 获取任务类型和类别名称
            info = {
                'task': self.model.task,
                'device': device,
                'names': self.model.names
            }
            return True, info
        except Exception as e:
            return False, str(e)

    def infer(self, image, imgsz=640):
        """执行单帧推理"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        results = self.model(image, imgsz=imgsz)
        return results[0]  # 返回 Results 对象
