# core/source.py
import cv2
from enum import Enum
from typing import Generator, Optional


class SourceType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"


class FrameSource:
    def __init__(self, source_type: SourceType, path_or_id=None):
        self.source_type = source_type
        self.path_or_id = path_or_id
        self.cap = None
        self.is_open = False

    def open(self) -> bool:
        if self.source_type == SourceType.CAMERA:
            self.cap = cv2.VideoCapture(0)
        elif self.source_type == SourceType.VIDEO:
            self.cap = cv2.VideoCapture(self.path_or_id)
        else:
            return False  # 图片不需要持续打开

        self.is_open = self.cap.isOpened()
        return self.is_open

    def frames(self) -> Generator[Optional[tuple], None, None]:
        """生成器模式返回每一帧"""
        if self.source_type == SourceType.IMAGE:
            yield None, cv2.imread(self.path_or_id)
        else:
            while self.is_open:
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield None, frame
            yield "end", None

    def release(self):
        if self.cap:
            self.cap.release()
        self.is_open = False