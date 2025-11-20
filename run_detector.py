#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5目标检测软件启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from yolo_detector import YOLODetectorApp
    import tkinter as tk
    from tkinter import messagebox
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有依赖项，运行: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = YOLODetectorApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"程序运行出错: {e}")
        messagebox.showerror("错误", f"程序运行出错: {e}")


if __name__ == "__main__":
    main()