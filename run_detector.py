#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv5目标检测软件启动脚本
"""

import sys
import os
import subprocess

def install_requirements():
    """安装依赖包"""
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_file):
        print("正在安装依赖包...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("依赖包安装完成！")
    else:
        print("未找到requirements.txt文件")

def main():
    """主函数"""
    try:
        # 尝试导入必要的库
        import tkinter as tk
        from yolo_detector import YOLODetectorApp
    except ImportError as e:
        print(f"导入库时出错: {e}")
        install_requirements()
        # 重新尝试导入
        try:
            import tkinter as tk
            from yolo_detector import YOLODetectorApp
        except ImportError as e:
            print(f"仍然无法导入库: {e}")
            print("请手动运行: pip install -r requirements.txt")
            return

    # 创建并运行GUI应用
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()