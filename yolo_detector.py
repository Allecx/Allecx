import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import threading
import time
import os

class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 目标检测软件")
        self.root.geometry("1400x900")
        
        # 初始化变量
        self.model_path = tk.StringVar()
        self.current_model = None
        self.cap = None
        self.is_detecting = False
        self.model_type = tk.StringVar(value="pt")  # 当前模型类型
        
        self.setup_ui()
    
    def setup_ui(self):
        # 配置网格权重，使界面可以响应窗口大小变化
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 模型选择区域
        model_frame = ttk.LabelFrame(main_frame, text="模型选择", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(model_frame, text="选择模型", command=self.select_model).grid(row=0, column=2, padx=5)
        ttk.Button(model_frame, text="加载模型", command=self.load_model).grid(row=0, column=3, padx=5)
        
        # 模型类型选择
        ttk.Label(model_frame, text="模型格式:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(5,0))
        model_type_combo = ttk.Combobox(model_frame, textvariable=self.model_type, values=["pt", "pth", "onnx"], state="readonly", width=10)
        model_type_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5,0))
        
        # 功能选择区域
        func_frame = ttk.LabelFrame(main_frame, text="功能选择", padding="10")
        func_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(func_frame, text="图片检测", command=self.detect_image).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(func_frame, text="摄像头检测", command=self.detect_camera).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(func_frame, text="视频检测", command=self.detect_video).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(func_frame, text="停止检测", command=self.stop_detection).grid(row=0, column=3, padx=5, pady=5)
        
        # 显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 左侧显示 - 原图
        left_frame = ttk.LabelFrame(display_frame, text="原图", padding="5")
        left_frame.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        self.original_label = ttk.Label(left_frame, text="请选择检测功能", background="black", foreground="white", anchor="center")
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 右侧显示 - 检测结果
        right_frame = ttk.LabelFrame(display_frame, text="检测结果", padding="5")
        right_frame.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        self.result_label = ttk.Label(right_frame, text="检测结果将在此显示", background="black", foreground="white", anchor="center")
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 检测信息显示
        info_frame = ttk.LabelFrame(main_frame, text="检测信息", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        self.info_text = tk.Text(info_frame, height=8)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)
    
    def select_model(self):
        model_types = {
            "pt": [("PyTorch模型", "*.pt")],
            "pth": [("PyTorch模型", "*.pth")],
            "onnx": [("ONNX模型", "*.onnx")]
        }
        
        current_type = self.model_type.get()
        file_types = model_types.get(current_type, [("All files", "*.*")])
        file_types.append(("All files", "*.*"))
        
        file_path = filedialog.askopenfilename(
            title=f"选择{current_type.upper()}模型文件",
            filetypes=file_types
        )
        if file_path:
            self.model_path.set(file_path)
    
    def load_model(self):
        try:
            if not self.model_path.get():
                messagebox.showerror("错误", "请先选择模型文件")
                return
            
            model_path = self.model_path.get()
            model_ext = os.path.splitext(model_path)[1][1:]  # 获取扩展名 without dot
            
            if model_ext == "pt":
                # 加载PyTorch模型
                self.current_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            elif model_ext == "pth":
                # 加载PyTorch模型
                self.current_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            elif model_ext == "onnx":
                # 加载ONNX模型
                self.current_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            else:
                messagebox.showerror("错误", f"不支持的模型格式: {model_ext}")
                return
            
            messagebox.showinfo("成功", f"{model_ext.upper()}模型加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
    
    def detect_image(self):
        if not self.current_model:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        try:
            # 读取图片
            img = cv2.imread(file_path)
            original_img = img.copy()
            
            # 进行检测
            results = self.current_model(img)
            
            # 显示原图
            self.display_image(original_img, self.original_label)
            
            # 显示检测结果
            img_with_boxes = results.render()[0]
            self.display_image(img_with_boxes, self.result_label)
            
            # 显示检测信息
            self.display_detection_info(results)
            
        except Exception as e:
            messagebox.showerror("错误", f"图片检测失败: {str(e)}")
    
    def detect_camera(self):
        if not self.current_model:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        
        self.is_detecting = True
        self.camera_loop()
    
    def camera_loop(self):
        if not self.is_detecting:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # 进行检测
            results = self.current_model(frame)
            
            # 显示原图
            self.display_image(frame, self.original_label)
            
            # 显示检测结果
            img_with_boxes = results.render()[0]
            self.display_image(img_with_boxes, self.result_label)
            
            # 显示检测信息（更新频率可调整，避免信息刷新过快）
            if hasattr(self, '_last_info_update'):
                if time.time() - self._last_info_update > 1.0:  # 每秒更新一次信息
                    self.display_detection_info(results)
                    self._last_info_update = time.time()
            else:
                self.display_detection_info(results)
                self._last_info_update = time.time()
        
        # 继续循环
        if self.is_detecting:
            self.root.after(10, self.camera_loop)  # 约100 FPS
    
    def detect_video(self):
        if not self.current_model:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        # 打开视频文件
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return
        
        self.is_detecting = True
        self.video_loop()
    
    def video_loop(self):
        if not self.is_detecting:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # 进行检测
            results = self.current_model(frame)
            
            # 显示原图
            self.display_image(frame, self.original_label)
            
            # 显示检测结果
            img_with_boxes = results.render()[0]
            self.display_image(img_with_boxes, self.result_label)
            
            # 显示检测信息（更新频率可调整）
            if hasattr(self, '_last_info_update'):
                if time.time() - self._last_info_update > 0.5:  # 每0.5秒更新一次信息
                    self.display_detection_info(results)
                    self._last_info_update = time.time()
            else:
                self.display_detection_info(results)
                self._last_info_update = time.time()
        else:
            # 视频播放结束
            self.is_detecting = False
            if self.cap:
                self.cap.release()
            return
        
        # 继续循环
        if self.is_detecting:
            self.root.after(1, self.video_loop)  # 尽可能快地处理视频帧
    
    def stop_detection(self):
        self.is_detecting = False
        if self.cap:
            self.cap.release()
            self.cap = None
        # 重置显示标签
        self.original_label.configure(image='', text="请选择检测功能", background="black", foreground="white")
        self.result_label.configure(image='', text="检测结果将在此显示", background="black", foreground="white")
    
    def display_image(self, img, label):
        # 转换图像格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 获取标签的当前尺寸
        label_width = label.winfo_width()
        label_height = label.winfo_height()
        
        # 如果标签还没有尺寸（例如首次显示），则使用默认尺寸
        if label_width <= 1 or label_height <= 1:
            label_width, label_height = 640, 480
        
        # 调整图像大小以适应显示区域，保持宽高比
        img_ratio = img_pil.width / img_pil.height
        label_ratio = label_width / label_height
        
        if img_ratio > label_ratio:
            # 图像更宽，以宽度为基准
            new_width = label_width
            new_height = int(label_width / img_ratio)
        else:
            # 图像更高，以高度为基准
            new_height = label_height
            new_width = int(label_height * img_ratio)
        
        img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk, text="", background="white")
        label.image = img_tk  # 保持引用防止垃圾回收
    
    def display_detection_info(self, results):
        # 清空文本框
        self.info_text.delete(1.0, tk.END)
        
        # 获取检测结果
        detections = results.pandas().xyxy[0]
        
        # 显示检测信息
        info = "检测结果:\n"
        for index, row in detections.iterrows():
            info += f"目标: {row['name']}, 置信度: {row['confidence']:.2f}, 位置: ({int(row['xmin'])}, {int(row['ymin'])}, {int(row['xmax'])}, {int(row['ymax'])})\n"
        
        # 添加统计信息
        if len(detections) > 0:
            info += f"\n统计信息:\n"
            for obj in detections['name'].value_counts().iteritems():
                info += f"{obj[0]}: {obj[1]} 个\n"
        else:
            info += "\n未检测到任何目标\n"
        
        self.info_text.insert(tk.END, info)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()