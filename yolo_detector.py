import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import threading
import time
import os
import queue  # 用于线程间通信


# --- 防止 seaborn 导入错误 ---
class DummySeaborn:
    pass


sys.modules['seaborn'] = DummySeaborn()

# --- 导入 ultralytics ---
from ultralytics import YOLO


class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5/YOLOv8 目标检测软件")
        self.root.geometry("1400x900")

        # 初始化变量
        self.model_path = tk.StringVar()
        self.current_model = None
        self.cap = None
        self.is_detecting = False
        self.model_type = tk.StringVar(value="pt")

        # 多线程相关
        self.input_queue = queue.Queue(maxsize=1)  # 输入帧队列
        self.output_queue = queue.Queue(maxsize=1)  # 输出结果队列
        self.stop_thread = False
        self.thread = None

        self.setup_ui()

    def setup_ui(self):
        """构建用户界面"""
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
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5,
                                                                            sticky=(tk.W, tk.E))
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
        self.original_label = ttk.Label(left_frame, text="请选择检测功能", background="black", foreground="white",
                                        anchor="center")
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        right_frame = ttk.LabelFrame(display_frame, text="检测结果", padding="5")
        right_frame.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        self.result_label = ttk.Label(right_frame, text="检测结果将在此显示", background="black", foreground="white",
                                      anchor="center")
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

        # 启动结果更新器（非阻塞）
        self.poll_results()

    def poll_results(self):
        """从 output_queue 中轮询结果，更新 UI"""
        try:
            result_data = self.output_queue.get_nowait()
            original_img, annotated_img, det_info = result_data

            self.display_image(original_img, self.original_label)
            self.display_image(annotated_img, self.result_label)
            self.display_detection_info(det_info)
        except queue.Empty:
            pass
        finally:
            self.root.after(30, self.poll_results)  # 每30ms检查一次结果

    def inference_worker(self):
        """运行在子线程中的推理函数"""
        while not self.stop_thread:
            try:
                frame = self.input_queue.get(timeout=1)
                if frame is None:
                    break

                # 执行推理
                results = self.current_model(frame, imgsz=640)
                result = results[0]

                # 获取渲染图像
                rendered = result.plot()  # BGR numpy array

                # 准备要返回的数据
                det_info = {
                    'boxes': result.boxes.cpu().numpy(),
                    'names': result.names
                }

                # 放入输出队列（非阻塞）
                try:
                    self.output_queue.put((frame.copy(), rendered, det_info), block=False)
                except queue.Full:
                    pass  # 丢弃旧结果，保证实时性

                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker] 推理出错: {e}")
                continue

    def start_inference_thread(self):
        """启动推理子线程"""
        if self.thread and self.thread.is_alive():
            return

        self.stop_thread = False
        self.thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.thread.start()

    def stop_inference_thread(self):
        """停止推理子线程"""
        self.stop_thread = True
        try:
            self.input_queue.put_nowait(None)  # 唤醒线程
        except:
            pass
        if self.thread:
            self.thread.join(timeout=1)

    def select_model(self):
        current_type = self.model_type.get()
        extensions = {"pt": "*.pt", "pth": "*.pth", "onnx": "*.onnx"}
        file_types = [(f"{current_type.upper()} 模型", extensions.get(current_type, "*.*"))]
        file_types.append(("所有文件", "*.*"))

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

            self.current_model = YOLO(path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.current_model.to(device)

            task = self.current_model.task
            model_name = os.path.basename(path)

            messagebox.showinfo(
                "成功",
                f"模型加载成功！\n名称: {model_name}\n任务: {task}\n设备: {device}"
            )
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

            results = self.current_model(img)
            result = results[0]

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
                # 尝试放入输入队列
                if self.input_queue.empty():
                    self.input_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # 忽略，等待下一帧

        if self.is_detecting:
            self.root.after(30, self.camera_capture_loop)  # 约30FPS采集

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

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        label_width = max(label.winfo_width(), 320)
        label_height = max(label.winfo_height(), 240)
        ratio = pil_img.width / pil_img.height
        label_ratio = label_width / label_height

        if ratio > label_ratio:
            new_width = label_width
            new_height = int(label_width / ratio)
        else:
            new_height = label_height
            new_width = int(label_height * ratio)

        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)
        label.configure(image=img_tk, text="", background="white")
        label.image = img_tk

    def display_detection_info(self, det_info):
        self.info_text.delete(1.0, tk.END)

        try:
            names = det_info['names']
            boxes = det_info['boxes']

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

            self.info_text.insert(tk.END, info)
        except Exception as e:
            self.info_text.insert(tk.END, f"解析结果失败: {e}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()
