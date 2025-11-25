# core/visualizer.py
import cv2
from PIL import Image, ImageTk


class Visualizer:
    @staticmethod
    def resize_for_display(cv_img, target_width: int, target_height: int):
        """将 OpenCV 图像缩放到适合 Label 显示的大小，保持宽高比"""
        h, w = cv_img.shape[:2]
        ratio = min(target_width / w, target_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil_img)

    @staticmethod
    def format_info_text(det_info: dict) -> str:
        """
        格式化检测信息文本
        det_info = {
            'boxes': numpy.ndarray (n, 6) → [x1, y1, x2, y2, conf, cls_id],
            'names': dict {id: name}
        }
        """
        boxes = det_info.get('boxes', [])
        names = det_info.get('names', {})

        info = "检测结果:\n"
        class_count = {}

        if len(boxes) == 0:
            info += "未检测到任何目标\n"
        else:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]

                info += f"目标: {cls_name}, 置信度: {conf:.2f}, 位置: ({x1},{y1},{x2},{y2})\n"
                class_count[cls_name] = class_count.get(cls_name, 0) + 1

            info += "\n统计信息:\n"
            for name, count in class_count.items():
                info += f"{name}: {count} 个\n"

        return info
