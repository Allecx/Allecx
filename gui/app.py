# gui/app.py
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import os
import queue

# --- 防止 seaborn 导入错误 ---
class DummySeaborn:
    pass
sys.modules['seaborn'] = DummySeaborn()

# ✅ 添加：导入 PIL 的 Image 模块（非常重要！）
from PIL import Image, ImageTk  # ← 改这里：之前可能只写了 ImageTk

from core.detector import Detector
from core.visualizer import Visualizer


class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("目标检测软件")
        self.root.geometry("1400x900")

        # 初始化变量（严格按照原始命名）
        self.model_path = tk.StringVar()
        self.current_model = None  # 用于存储 Detector 实例
        self.cap = None
        self.is_detecting = False
        self.model_type = tk.StringVar(value="pt")  # 默认格式 pt

        # 多线程相关
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.stop_thread = False
        self.thread = None

        # 存储图片引用防止被GC回收
        self.original_tk_image = None
        self.result_tk_image = None

        self.setup_ui()
        self.poll_results()  # 启动结果轮询

    def setup_ui(self):
        """构建用户界面（完全复刻原始布局）"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ========== 模型选择区域 ==========
        model_frame = ttk.LabelFrame(main_frame, text="模型选择", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(model_frame, text="选择模型", command=self.select_model).grid(row=0, column=2, padx=5)
        ttk.Button(model_frame, text="加载模型", command=self.load_model).grid(row=0, column=3, padx=5)

        ttk.Label(model_frame, text="模型格式:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(5, 0))
        model_type_combo = ttk.Combobox(model_frame, textvariable=self.model_type,
                                        values=["pt", "pth", "onnx"], state="readonly", width=10)
        model_type_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5, 0))

        # ========== 功能按钮区域 ==========
        func_frame = ttk.LabelFrame(main_frame, text="功能选择", padding="10")
        func_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(func_frame, text="图片检测", command=self.detect_image).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(func_frame, text="摄像头检测", command=self.detect_camera).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(func_frame, text="视频检测", command=self.detect_video).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(func_frame, text="停止检测", command=self.stop_detection).grid(row=0, column=3, padx=5, pady=5)

        # ========== 显示区域 ==========
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)

        left_frame = ttk.LabelFrame(display_frame, text="原图", padding="5")
        left_frame.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        self.original_label = ttk.Label(left_frame, text="请选择检测功能", background="black", foreground="white", anchor="center")
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        right_frame = ttk.LabelFrame(display_frame, text="检测结果", padding="5")
        right_frame.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        self.result_label = ttk.Label(right_frame, text="检测结果将在此显示", background="black", foreground="white", anchor="center")
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ========== 信息输出区域 ==========
        info_frame = ttk.LabelFrame(main_frame, text="检测信息", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

        self.info_text = tk.Text(info_frame, height=8)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)

    def poll_results(self):
        """从 output_queue 中轮询结果并更新 UI"""
        try:
            original_img, annotated_img, det_info = self.output_queue.get_nowait()
            self.display_image(original_img, self.original_label)
            self.display_image(annotated_img, self.result_label)
            self.display_detection_info(det_info)
        except queue.Empty:
            pass
        finally:
            self.root.after(30, self.poll_results)

    def inference_worker(self):
        """子线程中运行的推理函数"""
        while not self.stop_thread:
            try:
                frame = self.input_queue.get(timeout=1)
                if frame is None:
                    break

                # 执行推理
                result = self.current_model.infer(frame)
                rendered = result.plot()  # BGR numpy array

                # 提取信息
                det_info = {
                    'boxes': result.boxes.cpu().numpy(),
                    'names': result.names
                }

                # 放入输出队列
                try:
                    self.output_queue.put((frame.copy(), rendered, det_info), block=False)
                except queue.Full:
                    pass  # 丢弃旧帧

                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker] 推理出错: {e}")
                continue

    def start_inference_thread(self):
        """启动推理线程"""
        if self.thread and self.thread.is_alive():
            return
        self.stop_thread = False
        self.thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.thread.start()

    def stop_inference_thread(self):
        """停止推理线程"""
        self.stop_thread = True
        try:
            self.input_queue.put_nowait(None)
        except:
            pass
        if self.thread:
            self.thread.join(timeout=1)

    def select_model(self):
        current_type = self.model_type.get()
        extensions = {"pt": "*.pt", "pth": "*.pth", "onnx": "*.onnx"}
        file_types = [(f"{current_type.upper()} 模型", extensions.get(current_type, "*.*")), ("所有文件", "*.*")]

        file_path = filedialog.askopenfilename(title=f"选择 {current_type} 模型文件", filetypes=file_types)
        if file_path:
            self.model_path.set(file_path)

    def load_model(self):
        try:
            path = self.model_path.get()
            if not path:
                messagebox.showerror("错误", "请先选择模型文件")
                return
            if not os.path.exists(path):
                messagebox.showerror("错误", "模型文件不存在")
                return

            # 创建 Detector 并加载模型
            self.current_model = Detector()
            success, info_or_error = self.current_model.load_model(path, self.model_type.get())

            if success:
                task = info_or_error['task']
                device = info_or_error['device']
                model_name = os.path.basename(path)

                messagebox.showinfo(
                    "成功",
                    f"模型加载成功！\n名称: {model_name}\n任务: {task}\n设备: {device}"
                )
            else:
                messagebox.showerror("错误", f"模型加载失败:\n{info_or_error}")

        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败:\n{str(e)}")

    def detect_image(self):
        if not self.current_model:
            messagebox.showerror("错误", "请先加载模型")
            return

        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if not path:
            return

        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("无法读取图像")

            result = self.current_model.infer(img)
            self.display_image(img, self.original_label)
            self.display_image(result.plot(), self.result_label)
            self.display_detection_info({
                'boxes': result.boxes.cpu().numpy(),
                'names': result.names
            })
        except Exception as e:
            messagebox.showerror("错误", f"图片检测失败:\n{str(e)}")

    def detect_camera(self):
        if not self.current_model:
            messagebox.showerror("错误", "请先加载模型")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_detecting = True
        self.start_inference_thread()
        self.camera_capture_loop()

    def camera_capture_loop(self):
        if not self.is_detecting:
            return

        ret, frame = self.cap.read()
        if ret:
            try:
                if self.input_queue.empty():
                    self.input_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        if self.is_detecting:
            self.root.after(30, self.camera_capture_loop)

    def detect_video(self):
        if not self.current_model:
            messagebox.showerror("错误", "请先加载模型")
            return

        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if not path:
            return

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return

        self.is_detecting = True
        self.start_inference_thread()
        self.video_capture_loop()

    def video_capture_loop(self):
        if not self.is_detecting:
            return

        ret, frame = self.cap.read()
        if ret:
            try:
                if self.input_queue.empty():
                    self.input_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
        else:
            self.is_detecting = False
            self.cap.release()
            self.cap = None
            messagebox.showinfo("完成", "视频播放结束")
            return

        if self.is_detecting:
            self.root.after(1, self.video_capture_loop)

    def stop_detection(self):
        self.is_detecting = False
        self.stop_inference_thread()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.original_label.configure(image='', text="请选择检测功能", background="black", foreground="white")
        self.result_label.configure(image='', text="检测结果将在此显示", background="black", foreground="white")
        self.info_text.delete(1.0, tk.END)

        # 清除图像引用
        self.original_tk_image = None
        self.result_tk_image = None

    def display_image(self, img, label):
        """将 OpenCV 图像显示在 Label 上，使用 Visualizer 处理缩放"""
        try:
            # 使用 Visualizer 缩放并转为 Tkinter 可用格式
            photo_image = Visualizer.resize_for_display(img, label.winfo_width(), label.winfo_height())

            # 更新 Label 并保存引用
            label.configure(image=photo_image, text="", background="white")
            label.image = photo_image  # 防止被 GC 回收

        except Exception as e:
            print(f"[Display] 显示图像失败: {e}")

    def display_detection_info(self, det_info):
        """显示检测信息，使用 Visualizer 生成文本"""
        self.info_text.delete(1.0, tk.END)
        try:
            info_text = Visualizer.format_info_text(det_info)

            self.info_text.insert(tk.END, info_text)
        except Exception as e:
            # self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"解析结果失败: {e}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()